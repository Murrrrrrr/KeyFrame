import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict

class SparseKeyframeMetrics:
    """
    稀疏关键帧评估器 (5分类升级版)
    """
    def __init__(self, tolerance: int=3, threshold: float=0.5, min_step_frames: int=5, from_logits: bool=False, num_classes: int=5):
        self.tolerance = tolerance
        self.threshold = threshold
        self.min_step_frames = min_step_frames
        self.from_logits = from_logits
        self.num_classes = num_classes
        # 定义类别名称映射，方便最后输出
        self.class_names = ["Left_HS", "Left_TO", "Right_HS", "Right_TO", "Toe_Max"]
        self.reset()

    def reset(self):
        """每个 Epoch 开始前调用"""
        # 将全局统计改为字典，按类别通道索引 (0 到 num_classes-1) 独立统计
        self.tp_dict = {c: 0 for c in range(self.num_classes)}
        self.fp_dict = {c: 0 for c in range(self.num_classes)}
        self.fn_dict = {c: 0 for c in range(self.num_classes)}

    @torch.no_grad()
    def update(self, logits_or_probs: torch.Tensor, targets: torch.Tensor):
        if self.from_logits:
            probs = torch.sigmoid(logits_or_probs)
        else:
            probs = logits_or_probs

        probs = probs.transpose(1, 2)
        targets = targets.transpose(1, 2)

        kernel_size = self.min_step_frames + (1 if self.min_step_frames % 2 == 0 else 0)
        pad = kernel_size // 2

        # 预测概率 NMS 寻峰
        max_vals = F.max_pool1d(probs, kernel_size=kernel_size, stride=1, padding=pad)
        if max_vals.shape[-1] != probs.shape[-1]:
            max_vals = max_vals[..., :probs.shape[-1]]

        is_max = (probs == max_vals) & (probs >= self.threshold)
        shifted_right = torch.cat([torch.zeros_like(probs[:, :, :1]), probs[:, :, :-1]], dim=-1)
        shifted_left = torch.cat([probs[:, :, 1:], torch.zeros_like(probs[:, :, -1:])], dim=-1)
        peak_mask = is_max & (probs > shifted_right) & (probs >= shifted_left)

        # 真实标签 NMS 寻峰
        target_max_vals = F.max_pool1d(targets, kernel_size=kernel_size, stride=1, padding=pad)
        if target_max_vals.shape[-1] != targets.shape[-1]:
            target_max_vals = target_max_vals[..., :targets.shape[-1]]

        target_is_max = (targets == target_max_vals) & (targets >= 0.5)
        target_shifted_right = torch.cat([torch.zeros_like(targets[:, :, :1]), targets[:, :, :-1]], dim=-1)
        target_shifted_left = torch.cat([targets[:, :, 1:], torch.zeros_like(targets[:, :, -1:])], dim=-1)
        target_peak_mask = target_is_max & (targets > target_shifted_right) & (targets >= target_shifted_left)

        batch_size = probs.size(0)
        num_channels = probs.size(1)

        for i in range(batch_size):
            for c in range(num_channels):
                pred_indices = torch.nonzero(peak_mask[i, c]).squeeze(-1).cpu().tolist()
                gt_indices = torch.nonzero(target_peak_mask[i, c]).squeeze(-1).cpu().tolist()

                if isinstance(pred_indices, int): pred_indices = [pred_indices]
                if isinstance(gt_indices, int): gt_indices = [gt_indices]

                tp, fp, fn = self._match_1d_greedy(pred_indices, gt_indices)

                # 【核心修复点】按通道 c 独立累加，而不是混入 total_tp
                self.tp_dict[c] += tp
                self.fp_dict[c] += fp
                self.fn_dict[c] += fn

    def _match_1d_greedy(self, pred_idx: List[int], gt_idx: List[int]) -> Tuple[int, int, int]:
        tp = 0
        matched_preds = set()
        matched_gts = set()

        matches = []
        for p in pred_idx:
            for g in gt_idx:
                dist = abs(p - g)
                if dist <= self.tolerance:
                    matches.append((dist, p, g))

        matches.sort(key=lambda x: x[0])

        for dist, p, g in matches:
            if p not in matched_preds and g not in matched_gts:
                tp += 1
                matched_preds.add(p)
                matched_gts.add(g)

        fp = len(pred_idx) - len(matched_preds)
        fn = len(gt_idx) - len(matched_gts)

        return tp, fp, fn

    def compute(self) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        返回嵌套字典，包含所有独立类别的指标，以及 Macro 宏观指标
        """
        result = {}

        total_tp = sum(self.tp_dict.values())
        total_fp = sum(self.fp_dict.values())
        total_fn = sum(self.fn_dict.values())

        # 计算全局宏观指标
        macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        macro_f1 = 2 * (macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

        result['Macro'] = {
            "Precision": macro_p, "Recall": macro_r, "F1_Score": macro_f1,
            "TP": total_tp, "FP": total_fp, "FN": total_fn
        }

        # 计算每个类别的独立指标
        for c in range(self.num_classes):
            tp, fp, fn = self.tp_dict[c], self.fp_dict[c], self.fn_dict[c]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

            # 使用真实类别名作为 key
            class_name = self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
            result[class_name] = {
                "Precision": p, "Recall": r, "F1_Score": f1,
                "TP": tp, "FP": fp, "FN": fn
            }

        # 兼容 train.py 训练阶段读取标量指标（保证不破坏原有训练代码）
        result['F1_Score'] = macro_f1
        result['Precision'] = macro_p
        result['Recall'] = macro_r

        return result