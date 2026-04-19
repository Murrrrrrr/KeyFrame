import torch
from torch.amp import autocast
from tqdm import tqdm

from utils.metrics import SparseKeyframeMetrics

class Evaluator:
    def __init__(self, model, dataloader, criterion, config, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

        # 从评估配置初始化指标计算器
        eval_cfg = config.get("evaluation", {})
        tolerance = eval_cfg.get("tolerance_windows", 3)
        num_classes = config.get('model', {}).get('num_classes', 5)

        self.metrics = SparseKeyframeMetrics(
            tolerance=tolerance,
            from_logits=True,
            threshold=0.3,
            num_classes=num_classes
        )

        @torch.no_grad()
        def evaluate(self, epoch=None, prefix="Val"):
            """
            执行一次完整的评估
            """
            self.model.eval()
            self.metrics.reset()
            total_loss = 0.0

            desc = f"Epoch {epoch} [{prefix}]" if epoch is not None else f"[{prefix}] Evaluation"
            pbar = tqdm(self.dataloader, desc=desc)

            device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

            for batch_data, batch_labels in pbar:
                # 将训练数据batch送入设备
                batch_data = tuple(item.to(self.device, non_blocking=True) for item in batch_data)
                batch_labels = batch_labels.to(self.device, non_blocking=True)

                # AMP 混合精度推理
                with autocast(device_type=device_type):
                    logits = self.model(*batch_data)
                    loss, _, _ = self.criterion(logits, batch_labels)

                total_loss += loss.item()
                # 更新容差匹配指标
                self.metrics.updata(logits, batch_labels)
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.dataloader)
            metrics_result = self.metrics.compute()

            # 将 avg_loss 加入放回结果中，方便后续统一打印
            metrics_result['Loss'] = avg_loss
            return metrics_result