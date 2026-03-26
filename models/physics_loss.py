import torch
import torch.nn as nn
import torch.nn.functional as F

class StructLNNLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, physics_weight=0.1, min_step_frames=5, pos_weight=60.0):
        """
        联合损失函数： Focal Loss + 物理常识惩罚
        :param alpha: Focal Loss 正负样本平衡系数
        :param gamma: Focal Loss 难易样本聚焦系数
        :param physics_weight: 物理惩罚项权重
        :param min_step_frames: 物理极限约束，两次关键帧之间的最小帧间隔
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.physics_weight = physics_weight
        self.min_step_frames = min_step_frames
        self.pos_weight = pos_weight

    def forward(self, logits, targets, dt=None):
        """
        :param logits: [Batch, SeqLen, 1] 模型的原始输出（未经过 Sigmoid）
        :param targets:  [Batch, SeqLen, 1] 真实高斯软标签
        :param dt: [Batch, SeqLen, 1] 物理时间间隔，用于未来的积分约束
        """
        # 核心分类：BCE Focal Loss
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight_tensor)
        probs = F.sigmoid(logits)

        # 使用预测概率与高斯标签之间的绝对差值作为难易囊本的聚焦权重
        focal_weight = torch.abs(targets - probs) ** self.gamma

        # 类别平衡稀疏 alpha
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = (alpha_weight * focal_weight * bce_loss).mean()

        # 物理常识惩罚
        # 惩罚在极短窗口内出现的连续高概率激活（违背了人体运动学以及传感器物理规律）
        physics_loss = 0.0
        if self.physics_weight > 0:
            # 将 [Batch, SeqLen, 5] 转置为 [Batch, Channels(5), SeqLen]
            probs_transposed = probs.transpose(1, 2)

            # 使用 1D 平均池化计算局部窗口内的“概率积分”
            window_sum = F.avg_pool1d(
                probs_transposed,
                kernel_size=self.min_step_frames,
                stride=1,
                padding=self.min_step_frames // 2
            ) * self.min_step_frames

            # 转置回原来的形状
            window_sum = window_sum.transpose(1, 2)

            # 物理含义：如果某个独立通道在 min_step_frames 内的概率和超过 1.0 ，则产生惩罚
            physics_penalty = F.relu(window_sum - 1.0)
            physics_loss = physics_penalty.mean() * self.physics_weight

        # 总线聚合
        total_loss = focal_loss + physics_loss

        return total_loss, focal_loss, physics_loss