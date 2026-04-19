import torch
import torch.nn as nn
import torch.nn.functional as F

class StructLNNLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, physics_weight=1.0, min_step_frames=5, pos_weight=60.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.physics_weight = physics_weight
        self.min_step_frames = min_step_frames
        self.pos_weight = pos_weight

    def forward(self, logits, targets, dt=None):
        # 核心损失函数：BCE Focal Loss
        pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=pos_weight_tensor)
        probs = torch.sigmoid(logits)

        # 难易样本聚焦权重
        focal_weight = torch.abs(targets - probs)  ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = (alpha_weight * focal_weight * bce_loss).mean()

        # 物理常识惩罚
        physics_loss = 0.0
        if self.physics_weight > 0:
            probs_transposed = probs.transpose(1, 2)
            window_sum = F.avg_pool1d(
                probs_transposed,
                kernel_size=self.min_step_frames,
                stride=1,
                padding=self.min_step_frames // 2
            ) * self.min_step_frames
            window_sum = window_sum.transpose(1, 2)

            physics_penalty = F.relu(window_sum - 1.0)
            physics_loss = physics_penalty.mean() * self.physics_weight

        total_loss = focal_loss + physics_loss
        return total_loss, focal_loss, physics_loss