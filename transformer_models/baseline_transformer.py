import torch
import torch.nn as nn
import math
from models.heads.event_head import EventHead


class PositionalEncoding(nn.Module):
    """经典的固定位置编码（体现离散模型的缺陷）"""

    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x shape: [Batch, SeqLen, d_model]
        seq_len = x.size(1)
        # 将位置编码加到输入特征上（只在时间维度截取对应的长度）
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x


class BaselineTransformer(nn.Module):
    """
    基线对比网络：自注意力机制 Transformer
    用于证明即使是目前最强的固定位置编码模型，在面对丢帧抖动时也不如连续时间模型 LNN
    """

    def __init__(self, config=None, input_dim=43, hidden_dim=64, num_classes=5, num_layers=2):
        super(BaselineTransformer, self).__init__()

        if config is not None and 'model' in config:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_dim = model_cfg.get('hidden_size', hidden_dim)
            num_classes = model_cfg.get('num_classes', num_classes)
            num_layers = model_cfg.get('num_layers', num_layers)

        self.input_norm = nn.LayerNorm(input_dim)

        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 固定位置编码
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim)

        # Transformer 编码器核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # 4个注意力头
            dim_feedforward=hidden_dim * 2,
            batch_first=True
        )
        self.transformer_core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, batch_data):
        x, dt = batch_data  # 同样，Transformer 只能无视真实的 dt

        x_normlized = self.input_norm(x)
        x_feature = self.feature_projection(x_normlized)

        # 加入离散位置编码
        x_feature = self.pos_encoder(x_feature)

        # 自注意力时序推演
        tf_out = self.transformer_core(x_feature)

        # 降维回归
        logits = self.event_head(tf_out)
        return logits