import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftFocalLoss(nn.Module):
    """
    针对高斯软标签定制的 Focal Loss
    核心逻辑：降低大量“纯背景帧(0)”的权重，强迫模型关注那些“有动作发生的帧(>0)”
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(SoftFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. 将 logits 转换为概率 p (0~1)
        probs = torch.sigmoid(logits)

        # 2. 计算基础的 BCE Loss (不求均值)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 3. 构建动态权重 (Focal 机制)
        # 目标越接近 1，如果预测概率很低，权重就越大
        # 我们用绝对误差 |targets - probs| 作为调制系数
        pt_diff = torch.abs(targets - probs)
        focal_weight = self.alpha * (pt_diff ** self.gamma)

        # 4. 应用权重
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def build_criterion(config):
    """
    损失函数工厂模式 (Factory Pattern)
    根据 YAML 配置，动态选择和初始化损失函数
    """
    loss_type = config.get('train', {}).get('loss_type', 'bce')

    if loss_type == 'focal':
        print("[*] 选用损失函数: Soft Focal Loss (解决正负样本极度不平衡)")
        alpha = config.get('train', {}).get('focal_alpha', 0.25)
        gamma = config.get('train', {}).get('focal_gamma', 2.0)
        return SoftFocalLoss(alpha=alpha, gamma=gamma)

    elif loss_type == 'bce_weighted':
        print("[*] 选用损失函数: Weighted BCE Loss")
        # 假设正样本比负样本少 10 倍，强行给正样本加权
        pos_weight = torch.tensor([10.0])  # 可以根据实际统计修改
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    else:
        print("[*] 选用损失函数: 标准 BCEWithLogitsLoss")
        return nn.BCEWithLogitsLoss()