import os
import torch
from tqdm import tqdm


class StructLNNTrainer:
    """
    工业级跑姿模型训练引擎 (支持 AMP 混合精度 + 学习率调度 + F1分数监控)
    """

    def __init__(self, model, train_loader, valid_loader, optimizer, criterion, device,
                 save_dir, grad_clip=1.0, scheduler=None, use_amp=False, threshold=0.5):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_dir = save_dir
        self.grad_clip = grad_clip
        self.scheduler = scheduler
        self.threshold = threshold  # 用于计算 F1 的激活阈值

        self.use_amp = use_amp and (device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.best_valid_loss = float('inf')
        # 新增：记录最佳 F1 分数
        self.best_valid_f1 = 0.0

        os.makedirs(self.save_dir, exist_ok=True)

    def _calculate_confusion_matrix(self, logits, labels):
        """
        内部辅助函数：计算 TP, FP, FN
        注意：因为 labels 是高斯软标签，所以也需要 thresholding 转化为 0/1 才能算 F1
        """
        # 将输出 Logits 转化为 0~1 的概率
        probs = torch.sigmoid(logits)

        # 二值化预测和真实标签
        preds_bin = (probs > self.threshold).float()
        labels_bin = (labels > self.threshold).float()

        # 计算 True Positives, False Positives, False Negatives
        # 沿着 batch 和 seq_len 维度求和，保留类别维度
        tp = (preds_bin * labels_bin).sum(dim=(0, 1))
        fp = (preds_bin * (1.0 - labels_bin)).sum(dim=(0, 1))
        fn = ((1.0 - preds_bin) * labels_bin).sum(dim=(0, 1))

        return tp, fp, fn

    def _compute_f1_score(self, tp, fp, fn, eps=1e-7):
        """计算宏平均 (Macro) F1 分数"""
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_per_class = 2 * (precision * recall) / (precision + recall + eps)

        # 返回所有 5 个动作类别的平均 F1
        return f1_per_class.mean().item()

    def train_one_epoch(self, epoch, total_epochs):
        self.model.train()
        train_loss = 0.0

        # 累加器，用于计算整个 Epoch 的 F1
        total_tp = torch.zeros(self.model.event_head[-1].out_features, device=self.device)
        total_fp = torch.zeros_like(total_tp)
        total_fn = torch.zeros_like(total_tp)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
        current_lr = self.optimizer.param_groups[0]['lr']

        for features, labels in pbar:
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(features)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()

            # 统计混淆矩阵用于 F1 计算
            with torch.no_grad():
                tp, fp, fn = self._calculate_confusion_matrix(logits, labels)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = train_loss / len(self.train_loader)
        epoch_f1 = self._compute_f1_score(total_tp, total_fp, total_fn)

        return avg_loss, epoch_f1

    def validate(self, epoch, total_epochs):
        self.model.eval()
        valid_loss = 0.0

        total_tp = torch.zeros(self.model.event_head[-1].out_features, device=self.device)
        total_fp = torch.zeros_like(total_tp)
        total_fn = torch.zeros_like(total_tp)

        pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch}/{total_epochs} [Valid]", leave=False)
        with torch.no_grad():
            for features, labels in pbar:
                features, labels = features.to(self.device), labels.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(features)
                    loss = self.criterion(logits, labels)

                valid_loss += loss.item()

                tp, fp, fn = self._calculate_confusion_matrix(logits, labels)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = valid_loss / len(self.valid_loader)
        epoch_f1 = self._compute_f1_score(total_tp, total_fp, total_fn)

        return avg_loss, epoch_f1

    def fit(self, epochs):
        print("\n" + "🔥 " * 25)
        print("启动 Struct-LNN 训练引擎...")
        print(f"[*] 设备: {self.device} | AMP: {'开启' if self.use_amp else '关闭'}")
        print("🔥 " * 25 + "\n")

        for epoch in range(1, epochs + 1):
            train_loss, train_f1 = self.train_one_epoch(epoch, epochs)
            valid_loss, valid_f1 = self.validate(epoch, epochs)

            # 打印日志时加上 F1 分数！
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | "
                  f"Train Loss: {train_loss:.4f} - F1: {train_f1:.4f} | "
                  f"Valid Loss: {valid_loss:.4f} - F1: {valid_f1:.4f}")

            # 我们可以改为根据 F1 分数来保存最佳模型 (F1 越高越好)
            # 或者您可以保留原有的根据 valid_loss 越低越好来保存。这里我修改为综合判断，以 F1 优先。
            if valid_f1 >= self.best_valid_f1:
                self.best_valid_f1 = valid_f1
                self.best_valid_loss = valid_loss  # 顺便记录此时的 loss

                save_path = os.path.join(self.save_dir, "struct_lnn_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"   🌟 验证集 F1 刷新记录 ({self.best_valid_f1:.4f})! 模型已保存至: {save_path}")

        print("\n🎉 训练全部完成！")
        print(f"🥇 最佳验证集 F1 Score: {self.best_valid_f1:.4f} (此时 Loss: {self.best_valid_loss:.4f})")