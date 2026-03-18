import os
import argparse
from gc import enable

import yaml
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

# 从系统搭建的各个模块导入组件
from datasets.pose_dataset import PoseSequenceDataset
from models.struct_lnn import StructLNN
from models.physics_loss import StructLNNLoss
from utils.metrics import SpareseKeyframeMetrics

def parse_args():
    parser = argparse.ArgumentParser(description="Struct-LNN 关键帧训练引擎")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重路径")
    return parser.parse_args()

def main():
    args = parse_args()

    # 硬件与配置环境初始化
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" 训练引擎已挂载至设备: {device}")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("已开启 cuDNN Benchmark 加速")

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0.0
    start_epoch = 0

    # 数据链路装载 (Data Pipeline)
    print("[ 正在构建 DataLoader...")
    train_dataset = PoseSequenceDataset(config, split='train')
    val_dataset = PoseSequenceDataset(config, split='valid')

    num_workers = config['training'].get('num_workers', min(8, multiprocessing.cpu_count() // 2))
    print(f" 启用的 DataLoader 工作线程数：{num_workers}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # 锁页内存，加速 CPU 到 GPU 的数据传输
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
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
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())  # 自动缩放梯度，防止 float16 下的梯度下溢

    epochs = config['training']['epochs']
    # 使用余弦退火策略，让模型在后期能够更平滑地收敛
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 模型训练中断时还能恢复权重逻辑以及训练进度
    if args.resume and os.path.isfile(args.resume):
        print(f" 正在从 {args.resume} 恢复检查点...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_f1 = checkpoint['best_f1']
        start_epoch = checkpoint['epoch'] + 1

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_f1' in checkpoint:
            best_f1 = checkpoint['best_f1']
        print(f" 恢复成功！将从第{start_epoch + 1} 个 Epoch 继续训练。历史最佳 F1：{best_f1:.4f}")

    # 评估标尺
    tolerance = config['evaluation'].get('tolerance_windows', 3)
    val_metrics = SpareseKeyframeMetrics(tolerance=tolerance)

    # 训练主循环
    epochs = config['training']['epochs']
    print(f"[Engine] 开始训练，总 Epochs: {epochs}")

    for epoch in range(start_epoch,epochs):
        model.train()
        train_loss_epoch = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_data, batch_labels in pbar:
            # 数据送入 VRAM
            batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True) # 释放显存，略微加速

            # 开启 AMP 上下文
            device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
            with autocast(device_type=device_type):
                logits = model(batch_data)
                # 计算总损失 (融合分类与物理去抖)
                total_loss, focal_loss, phys_loss = criterion(logits, batch_labels)

            # 缩放反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}",
                              "Phys": f"{phys_loss.item() if isinstance(phys_loss, torch.Tensor) else phys_loss:.4f}"})

        scheduler.step() # 触发学习率调度器

        # 验证阶段
        model.eval()
        val_metrics.reset()
        val_loss_epoch = 0.0

        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
                batch_labels = batch_labels.to(device, non_blocking=True)

                with autocast(device_type=device_type):
                    logits = model(batch_data)
                    loss, _, _ = criterion(logits, batch_labels)

                val_loss_epoch += loss.item()
                # 更新容差匹配指标
                val_metrics.update(logits, batch_labels)

        # 计算当前 Epoch 的严格指标
        metrics_result = val_metrics.compute()
        current_f1 = metrics_result['F1_Score']

        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_val_loss = val_loss_epoch / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"--> Epoch {epoch + 1} Val Loss: {val_loss_epoch / len(val_loader):.4f} | "
              f"Val F1: {current_f1:.4f} (P: {metrics_result['Precision']:.4f}, R: {metrics_result['Recall']:.4f})")

        # 硬件级 Checkpoint 保存：仅保存表现最好（F1最高）的模型
        if current_f1 > best_f1:
            best_f1 = current_f1
            save_path = os.path.join("checkpoints", f"{config['experiment_name']}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
            }, save_path)
            print(f"[Checkpoint] 最好硬件权重已保存 -> F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()