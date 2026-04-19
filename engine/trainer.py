import os
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from engine.evaluator import Evaluator

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, config):
        """
        初始化训练引擎
        :param model:
        :param train_loader:
        :param val_loader:
        :param optimizer:
        :param criterion:
        :param config:
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # 将模型和损失函数挂载到指定硬件
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 读取训练策略参数
        train_cfg = config.get('training', {})
        self.epochs = train_cfg.get('epochs', 30)
        self.start_epoch = 0

        # 混合精度加速引擎（AMP）
        self.scaler = GradScaler("cuda", enabled=self.device.type == 'cuda')

        # 实例化独立的验证器
        self.evaluator = Evaluator(
            model=self.model,
            dataloader=self.val_loader,
            criterion=self.criterion,
            config=self.config,
            device=self.device,
        )

        # Checkpoint 存储路径
        self.experiment_name = config.get('experiment_name', 'StructLNN_Experiment')
        self.save_dir = 'checkpoints'
        os.makedirs(self.save_dir, exist_ok=True)

        # 跟踪最佳指标
        self.best_f1 = 0.0

    def train_epoch(self, epoch):
        """单轮训练流水线"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        for batch_data, batch_label in pbar:
            # 数据转移到硬件（非阻塞模式）
            batch_data = tuple(item.to(self.device, non_blocking=True) for item in batch_data)
            batch_labels = batch_label.to(self.device, non_blocking=True)

            # 清空旧梯度
            self.optimizer.zero_grad(set_to_none=True)

            # 混合精度前向传播与计算损失
            with autocast(device_type=device_type):
                logits = self.model(*batch_data)
                loss, focal_loss, phys_loss = self.criterion(logits, batch_labels)

            # 反向传播与梯度缩放
            self.scaler.scale(loss).backward()

            # 梯度裁剪：防止 LNN 在长序列下梯度爆炸
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

            # 权重更新
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            # 实时更新进度条显示物理去抖和总损失
            phys_val = phys_loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Phys": f"{phys_val:.4f}"})
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, epoch, metrics, is_best):
        """保存模型状态， 为边缘端推理铺路"""
        save_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'metrics': metrics,
        }

        # 始终保存最新的权重
        last_path = os.path.join(self.save_dir, f"{self.experiment_name}_last.pth")
        torch.save(save_data, last_path)

        if is_best:
            best_path = os.path.join(self.save_dir, f"{self.experiment_name}_last_best.pth")
            torch.save(save_data, best_path)
            print(f" [Checkpoint] 最佳模型已更新并保存，F1 Score:{self.best_f1:.4f}")

    def run(self):
        """启动整个训练生命周期"""
        print(f" [engine] 开始训练流水线，总 Epochs: {self.epochs}")

        for epoch in range(self.start_epoch, self.epochs):
            # 训练阶段
            train_loss = self.train_epoch(epoch)

            # 评估阶段
            metrics_result = self.evaluator.evaluate(epoch=epoch+1)

            # 提取指标
            val_loss = metrics_result.get('Loss', 0.0)
            current_f1 = metrics_result.get('F1_score', 0.0)
            precision = metrics_result.get('Precision', 0.0)
            recall = metrics_result.get('Recall', 0.0)

            # 打印报表
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f" Epoch {epoch+1} Summary | LR：{current_lr:.2e}")
            print(f" Train Loss: {train_loss:.4f} | Val Loss：{val_loss:.4f}")
            print(f" Val F1：{current_f1:.4f} (Precision：{precision: .4f}, Recall：{recall: .4f})")

            # 判断并保存
            is_best = current_f1 > self.best_f1
            if is_best:
                self.best_f1 = current_f1

            self.save_checkpoint(epoch, metrics_result, is_best)