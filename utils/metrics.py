""" 训练评价指标 """
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict

class SparseKeyframeMetrics:
    """
    稀疏关键帧评估器
    """
    def __init__(self, tolerance: int=3, threshold: float=0.5, min_step_frames: int=5, from_logits: bool=False, num_classes: int=5):
        """
        :param tolerance: 容差匹配窗口（默认正负3帧，即 delta_t 容忍度）
        :param threshold: 激活概率的绝对阈值
        :param min_step_frames: 两次极值之间的最小物理帧间隔 (需与 physics_loss 保持一致)
        """
        self.tolerance = tolerance
        self.threshold = threshold
        self.min_step_frames = min_step_frames
        self.from_logits = from_logits
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """每个 Epoch 开始前调用"""
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

    @torch.no_grad()
    def update(self, logits_or_probs: torch.Tensor, targets: torch.Tensor):
        """
        接收当前 Batch 的预测与真实标签并累加统计。
        :param logits_or_probs: [Batch, 64, 1] 或 [Batch, 64]
        :param targets: [Batch, 64, 1] 或 [Batch, 64] 二值标签
        """
        # 统一张量维度与数值域
        if self.from_logits:
            probs = torch.sigmoid(logits_or_probs)
        else:
            probs = logits_or_probs

        # 转换前: [Batch, SeqLen, NumClasses]
        # 转换后: [Batch, NumClasses, SeqLen]
        probs = probs.transpose(1, 2)
        targets = targets.transpose(1, 2)

        # 确保感受野为奇数
        kernel_size = self.min_step_frames + (1 if self.min_step_frames % 2 == 0 else 0)
        pad = kernel_size // 2

        # 核心逻辑 A：对预测概率 (Probs) 进行多通道 NMS 寻峰
        max_vals = F.max_pool1d(probs, kernel_size=kernel_size, stride=1, padding=pad)
        if max_vals.shape[-1] != probs.shape[-1]:
            max_vals = max_vals[..., :probs.shape[-1]]

        is_max = (probs == max_vals) & (probs >= self.threshold)
        shifted_right = torch.cat([torch.zeros_like(probs[:, :, :1]), probs[:, :, :-1]], dim=-1)
        shifted_left = torch.cat([probs[:, :, 1:], torch.zeros_like(probs[:, :, -1:])], dim=-1)

        peak_mask = is_max & (probs > shifted_right) & (probs >= shifted_left)

        # 核心逻辑 B：对真实高斯软标签 (Targets) 进行同等 NMS 寻峰
        # 防止连续的 >0.5 概率导致 GT 帧数成倍增加
        target_max_vals = F.max_pool1d(targets, kernel_size=kernel_size, stride=1, padding=pad)
        if target_max_vals.shape[-1] != targets.shape[-1]:
            target_max_vals = target_max_vals[..., :targets.shape[-1]]

        # 注意这里 threshold 可以设低一点，只要它是局部极值即可
        target_is_max = (targets == target_max_vals) & (targets >= 0.5)
        target_shifted_right = torch.cat([torch.zeros_like(targets[:, :, :1]), targets[:, :, :-1]], dim=-1)
        target_shifted_left = torch.cat([targets[:, :, 1:], torch.zeros_like(targets[:, :, -1:])], dim=-1)

        target_peak_mask = target_is_max & (targets > target_shifted_right) & (targets >= target_shifted_left)

        # 双重循环：在 Batch 和 Channel 维度上分别匹配
        batch_size = probs.size(0)
        num_channels = probs.size(1)

        for i in range(batch_size):
            for c in range(num_channels):
                # 提取当前 Batch、当前 Channel 下的波峰帧索引
                pred_indices = torch.nonzero(peak_mask[i, c]).squeeze(-1).cpu().tolist()
                gt_indices = torch.nonzero(target_peak_mask[i, c]).squeeze(-1).cpu().tolist()

                # 兼容处理：确保是 list
                if isinstance(pred_indices, int): pred_indices = [pred_indices]
                if isinstance(gt_indices, int): gt_indices = [gt_indices]

                # 独立匹配当前通道
                tp, fp, fn = self._match_1d_greedy(pred_indices, gt_indices)
                self.total_tp += tp
                self.total_fp += fp
                self.total_fn += fn

    def _match_1d_greedy(self, pred_idx: List[int], gt_idx: List[int]) -> Tuple[int, int, int]:
        """
        基于帧容差窗口的一维贪心二分图匹配。
        保证 1-to-1 映射，避免逻辑抖动导致的重复匹配。
        """
        tp = 0
        matched_preds = set()
        matched_gts = set()

        matches = []
        for p in pred_idx:
            for g in gt_idx:
                dist = abs(p - g)
                if dist <= self.tolerance:
                    matches.append((dist, p, g))

        # 贪心策略：优先匹配物理距离（时间）最接近的帧
        matches.sort(key=lambda x: x[0])

        for dist, p, g in matches:
            if p not in matched_preds and g not in matched_gts:
                tp += 1
                matched_preds.add(p)
                matched_gts.add(g)

        # 统计假阳性和假阴性
        fp = len(pred_idx) - len(matched_preds)
        fn = len(gt_idx) - len(matched_gts)

        return tp, fp, fn

    def compute(self) -> Dict[str, float]:
        """
        计算并返回最终指标，可直接送入 WandB 或 TensorBoard
        """
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0

        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1_score,
            "TP": self.total_tp,
            "FP": self.total_fp,
            "FN": self.total_fn
        }