import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict

class SpareseKeyframeMetrics:
    """
    稀疏关键帧评估器
    """
    def __init__(self, tolerance: int=3, threshold: float=0.5, min_step_frames: int=5, from_logits: bool=False):
        """
        :param tolerance: 容差匹配窗口（默认正负3帧，即 delta_t 容忍度）
        :param threshold: 激活概率的绝对阈值
        :param min_step_frames: 两次极值之间的最小物理帧间隔 (需与 physics_loss 保持一致)
        """
        self.tolerance = tolerance
        self.threshold = threshold
        self.min_step_frames = min_step_frames
        self.from_logits = from_logits
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

        probs = probs.squeeze(-1).view(targets.size(0), -1)  # [Batch, 64]
        targets = targets.squeeze(-1).view(targets.size(0), -1)  # [Batch, 64]

        # PyTorch Native 1D NMS (非极大值抑制) 提取波峰
        # 利用 min_step_frames 作为 1D Pooling 的感受野，呼应 physics_loss 中的物理约束
        kernel_size = self.min_step_frames + (1 if self.min_step_frames % 2 == 0 else 0)  # 确保感受野为奇数
        pad = kernel_size // 2

        probs_unsqueeze = probs.unsqueeze(1)  # [Batch, 1, 64]

        # 通过 MaxPool1d 滑动窗口寻找局部极值
        max_vals = F.max_pool1d(probs_unsqueeze, kernel_size=kernel_size, stride=1, padding=pad)

        # 截断由于 padding 可能导致的多余长度
        if max_vals.shape[-1] != probs.shape[-1]:
            max_vals = max_vals[..., :probs.shape[-1]]

        # 波峰触发条件：当前点是局部最大值 AND 概率超过硬件预设阈值
        peak_mask = (probs_unsqueeze == max_vals) & (probs_unsqueeze >= self.threshold)
        peak_mask = peak_mask.squeeze(1)  # [Batch, 64]

        # CPU/NumPy 离线贪心容差匹配
        batch_size = probs.size(0)
        target_mask = targets > 0.5

        batch_size = probs.size(0)
        for i in range(batch_size):
            # 在gpu 端找出True 的位置，再 .cpu().tolist()，数据量从 Batch*64 锐减到几个数字
            pred_indices = torch.nonzero(peak_mask[i]).squeeze(-1).cpu().tolist()
            gt_indices = torch.nonzero(target_mask[i]).squeeze(-1).cpu().tolist()

            # 兼容处理：如果只有一个极值点，torch.nonzero 返回的 list 可能是单个数字，转为list
            if isinstance(pred_indices, int): pred_indices = [pred_indices]
            if isinstance(gt_indices, int): gt_indices = [gt_indices]

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

        # 计算所有可能的 pred-gt 距离
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