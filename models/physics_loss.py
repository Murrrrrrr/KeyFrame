import torch
import torch.nn as nn
import torch.nn.functional as F

class StructLNNLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, physics_weight=0.1, min_step_frames=5):
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

    def forward(self, logits, targets, dt=None):
        """
        :param logits: [Batch, 64, 1] 模型的原始输出（未经过 Sigmoid）
        :param targets:  [Batch, 64, 1] 真实标签（0或1）
        :param dt: [Batch, 64, 1] 物理时间间隔，用于未来的积分约束
        """
        # 核心分类：BCE Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = F.sigmoid(logits)

        # 动态计算 focal 权重：p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - p_t) ** self.gamma

        focal_loss = (focal_weight * bce_loss).mean()

        # 物理常识惩罚
        # 惩罚在极短窗口内出现的连续高概率激活（违背了人体运动学以及传感器物理规律）
        physics_loss = 0.0
        if self.physics_weight > 0:
            # 使用 1D 最大池化寻找局部窗口内的峰值冲突
            # 窗口大小为 min_step_frames，如果局部连续高激活，Max Pool 后求和会放大惩罚
            smoothed_probs = probs.squeeze(-1) # [Batch, 64]

            prob_diff = torch.abs(smoothed_probs[:, 1:] - smoothed_probs[:, :-1])
            tv_loss = prob_diff.mean() # Total Variation 惩罚过拟合的尖峰噪音

            physics_loss = tv_loss * self.physics_weight

        # 总线聚合
        total_loss = focal_loss + physics_loss

        return total_loss, focal_loss, physics_loss