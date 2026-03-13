import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# 从我们搭建的各个模块导入组件
from datasets.pose_dataset import PoseSequenceDataset  # 假设你的 dataset 类名为 PoseDataset
from models.struct_lnn import StructLNN  # 假设你的网络外壳类名为 StructLNN
from models.physics_loss import StructLNNLoss
from utils.metrics import SpareseKeyframeMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Struct-LNN 极度稀疏关键帧训练引擎")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重路径")
    return parser.parse_args()

def main():
    args = parse_args()

    # 硬件与配置环境初始化
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] 训练引擎已挂载至设备: {device}")

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0.0

    # 数据链路装载 (Data Pipeline)
    print("[Pipeline] 正在构建零拷贝 DataLoader...")
    train_dataset = PoseSequenceDataset(config, split='train')
    val_dataset = PoseSequenceDataset(config, split='valid')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True  # 锁页内存，加速 CPU 到 GPU 的数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 核心模型与物理损失初始化
    model = StructLNN(config).to(device)

    # 提取 Loss 配置
    bce_weight = config['training'].get('loss', {}).get('bce_weight', 1.0)
    physics_weight = config['training'].get('loss', {}).get('physics_penalty_weight', 0.5)
    criterion = StructLNNLoss(physics_weight=physics_weight).to(device)

    # 优化器与混合精度引擎 (AMP)
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-4)
    scaler = GradScaler()  # 自动缩放梯度，防止 float16 下的梯度下溢

    # 评估标尺
    tolerance = config['evaluation'].get('tolerance_windows', 3)
    val_metrics = SpareseKeyframeMetrics(tolerance=tolerance)

    # 训练主循环
    epochs = config['training']['epochs']
    print(f"[Engine] 开始训练，总 Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        train_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_data, batch_labels in pbar:
            # 数据送入 VRAM
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 开启 AMP 上下文
            with autocast():
                logits = model(batch_data)
                # 计算总损失 (融合分类与 1D TV 物理去抖)
                total_loss, focal_loss, phys_loss = criterion(logits, batch_labels)

            # 缩放反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}",
                              "Phys": f"{phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss:.4f}"})

        # 验证阶段
        model.eval()
        val_metrics.reset()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                with autocast():
                    logits = model(batch_data)
                    loss, _, _ = criterion(logits, batch_labels)

                val_loss_epoch += loss.item()
                # 更新容差匹配指标
                val_metrics.update(logits, batch_labels)

        # 计算当前 Epoch 的严格指标
        metrics_result = val_metrics.compute()
        current_f1 = metrics_result['F1_Score']

        print(f"--> Epoch {epoch + 1} Val Loss: {val_loss_epoch / len(val_loader):.4f} | "
              f"Val F1: {current_f1:.4f} (P: {metrics_result['Precision']:.4f}, R: {metrics_result['Recall']:.4f})")

        # 硬件级 Checkpoint 保存：仅保存表现最好（F1最高）的模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_path = os.path.join("checkpoints", f"{config['experiment_name']}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
            }, save_path)
            print(f"[Checkpoint] 最强硬件权重已保存 -> F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()