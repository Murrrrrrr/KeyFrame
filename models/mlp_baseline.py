# 文件路径: models/mlp_baseline.py
import torch
import torch.nn as nn
from models.heads.event_head import EventHead


class MLPBaseline(nn.Module):
    """
    MLP 基线靶子模型 (无时序记忆，无物理时间感知)
    用于证明跑姿关键帧提取必须依赖时序动态演化。
    """

    def __init__(self, config=None, input_dim=67, hidden_dim=128, num_classes=1):
        super(MLPBaseline, self).__init__()

        if config is not None:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_dim = model_cfg.get('hidden_dim', hidden_dim)
            num_classes = model_cfg.get('num_classes', num_classes)

        self.hidden_dim = hidden_dim

        # 纯前馈感知机网络，用来替代 CfC 液态核心
        self.mlp_core = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 挂载你原本写好的降维回归任务头
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, batch_data):
        """
        前向传播
        """
        x, dt = batch_data

        # 【致命弱点注入】在这里，MLP 完全忽略了物理时间戳 dt
        # 它只是在特征维度（最后一维）上做非线性映射，毫无上下文概念
        sequence_output = self.mlp_core(x)

        # 回归输出
        logits = self.event_head(sequence_output)
        return logits


# 尺寸自检
if __name__ == "__main__":
    dummy_x = torch.randn(32, 64, 67)
    dummy_dt = torch.randn(32, 64, 1)  # MLP 实际上不会用到它，但为了接口统一保留
    model = MLPBaseline(input_dim=67, hidden_dim=128, num_classes=1)
    out = model((dummy_x, dummy_dt))
    print(f"MLP 输出维度: {out.shape} (预期: [32, 64, 1])")