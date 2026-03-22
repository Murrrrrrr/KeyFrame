# 文件路径: tools/train_mlp.py
import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing

# 导入你的组件
from datasets.pose_dataset import PoseSequenceDataset
from models.physics_loss import StructLNNLoss
from utils.metrics import SpareseKeyframeMetrics
from models.mlp_baseline import MLPBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="MLP 基线模型训练引擎")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的权重路径")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 兼容处理：YAML 中写的是 data: 而 Dataset 脚本中读取的是 dataset:
    if 'data' in config and 'dataset' not in config:
        config['dataset'] = config['data']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] MLP 训练引擎挂载设备: {device} | 实验名称: {config.get('experiment_name')}")

    os.makedirs("checkpoints", exist_ok=True)
    best_f1 = 0.0
    start_epoch = 0

    train_dataset = PoseSequenceDataset(config, split='train')
    val_dataset = PoseSequenceDataset(config, split='valid')
    num_workers = config['training'].get('num_workers', min(8, multiprocessing.cpu_count() // 2))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # 实例化 MLP 基线模型，传入对齐好的 config
    model = MLPBaseline(config=config).to(device)

    # 精确解析损失函数权重
    loss_cfg = config['training'].get('loss', {})
    bce_weight = loss_cfg.get('bce_weight', 1.0)
    physics_weight = loss_cfg.get('physics_penalty_weight', 0.5)
    pos_weight = loss_cfg.get('pos_weight', 60.0)
    focal_gamma = loss_cfg.get('focal_gamma', 2.0)

    criterion = StructLNNLoss(physics_weight=physics_weight, pos_weight=pos_weight, gamma=focal_gamma).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])

    tolerance = config['evaluation'].get('tolerance_windows', 3)
    val_metrics = SpareseKeyframeMetrics(tolerance=tolerance, from_logits=True, threshold=0.3)

    epochs = config['training']['epochs']
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss_epoch = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")

        for batch_data, batch_labels in pbar:
            batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
            batch_labels = batch_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type):
                logits = model(batch_data)
                total_loss, focal_loss, phys_loss = criterion(logits, batch_labels)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_epoch += total_loss.item()
            pbar.set_postfix({"Loss": f"{total_loss.item():.4f}"})

        scheduler.step()

        # ---------------- 验证阶段 ----------------
        model.eval()
        val_metrics.reset()
        val_loss_epoch = 0.0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
                batch_labels = batch_labels.to(device, non_blocking=True)

                with autocast(device_type=device_type):
                    logits = model(batch_data)
                    loss, _, _ = criterion(logits, batch_labels)

                val_loss_epoch += loss.item()
                val_metrics.update(logits, batch_labels)

        metrics_result = val_metrics.compute()
        current_f1 = metrics_result['F1_Score']

        print(f"--> Epoch {epoch + 1} Val Loss: {val_loss_epoch / len(val_loader):.4f} | "
              f"Val F1: {current_f1:.4f} (P: {metrics_result['Precision']:.4f}, R: {metrics_result['Recall']:.4f})")

        # 核心：保存最佳权重
        if current_f1 > best_f1:
            best_f1 = current_f1
            # 使用 YAML 中的 experiment_name 作为前缀
            save_path = os.path.join("checkpoints", f"{config.get('experiment_name', 'mlp_baseline')}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
            }, save_path)
            print(f"[Checkpoint] MLP 最优权重已保存 -> F1: {best_f1:.4f} (Saved to {save_path})")


if __name__ == "__main__":
    main()