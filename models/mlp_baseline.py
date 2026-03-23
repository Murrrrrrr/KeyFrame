import torch
import torch.nn as nn
from models.heads.event_head import EventHead


class MLPBaseline(nn.Module):
    """
    MLP 基线靶子模型 (无时序记忆，无物理时间感知)
    支持根据 yaml 动态生成层数，完全对齐消融实验配置。
    """

    def __init__(self, config=None, input_dim=67, hidden_size=64, num_layers=2, output_dim=1):
        super(MLPBaseline, self).__init__()

        # 从 YAML 配置中安全解析参数，修复了变量名不匹配的问题
        if config is not None:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_size = model_cfg.get('hidden_size', hidden_size)  # 对应 yaml 的 hidden_size
            num_layers = model_cfg.get('num_layers', num_layers)  # 对应 yaml 的 num_layers
            output_dim = model_cfg.get('output_dim', output_dim)  # 对应 yaml 的 output_dim

        self.hidden_size = hidden_size

        # 动态构建 MLP 核心结构
        layers = []
        current_in_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            current_in_dim = hidden_size  # 后续隐藏层的输入维度均为 hidden_size

        self.mlp_core = nn.Sequential(*layers)

        # 挂载降维回归任务头
        self.event_head = EventHead(hidden_dim=hidden_size, num_classes=output_dim)

    def forward(self, batch_data):
        """
        前向传播
        """
        x, dt = batch_data

        # 【致命弱点注入】MLP 完全忽略了物理时间戳 dt
        # 它只是在特征维度上做非线性映射，毫无上下文概念
        sequence_output = self.mlp_core(x)

        # 回归输出
        logits = self.event_head(sequence_output)
        return logits

# 尺寸自检
if __name__ == "__main__":
    dummy_x = torch.randn(32, 64, 67)
    dummy_dt = torch.randn(32, 64, 1)
    model = MLPBaseline()
    out = model((dummy_x, dummy_dt))
    print(f"MLP 输出维度: {out.shape} (预期: [32, 64, 1])")