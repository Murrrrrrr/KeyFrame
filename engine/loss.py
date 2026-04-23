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
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)

        # 动态自适应权重
        weight_mask = torch.where(targets > 0.05, self.pos_weight, 1.0)
        weighted_bce = bce_loss * weight_mask

        # 分组归约
        loss_per_batch = weighted_bce.sum(dim=[1, 2]) / targets.size(1)
        main_loss = loss_per_batch.mean()

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

            physics_penalty = F.relu(window_sum - 2.0)
            physics_loss = (physics_penalty.sum(dim=[1,2]) / targets.size(1)).mean() * self.physics_weight

        total_loss = main_loss + physics_loss
        return total_loss, main_loss, physics_loss