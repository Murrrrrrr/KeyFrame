import torch
import torch.nn as nn
from models.heads.event_head import EventHead


class BaselineTransformer(nn.Module):
    """
    离散时间基线模型：Transformer Encoder
    用于评估纯自注意力机制在处理高频骨骼动作序列时的表现，对比其与 LNN 的算力消耗。
    """

    def __init__(self, config=None):
        super(BaselineTransformer, self).__init__()

        model_cfg = config.get('model', {}) if config else {}
        input_dim = model_cfg.get('input_dim', 43)
        hidden_dim = model_cfg.get('hidden_size', 64)
        num_classes = model_cfg.get('num_classes', 5)
        num_layers = model_cfg.get('num_layers', 2)
        nhead = model_cfg.get('nhead', 4)  # 注意力头数

        # 1. 维度对齐投影层 (Transformer 严格要求 d_model 必须能被 nhead 整除)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2. 构建 Transformer 编码器主干
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 同样的复用标准输出头
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x, dt=None):
        """
        前向传播总线
        """
        # 🌟 同样优雅地忽略物理时间戳 dt

        # 先升维到 hidden_dim (比如从 43 维变到 64 维)
        x_proj = self.input_projection(x)

        # 自注意力编码
        out = self.transformer(x_proj)

        # 事件分类预测
        logits = self.event_head(out)
        return logits