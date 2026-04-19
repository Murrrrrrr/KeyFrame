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
    结合因果掩码 (Causal Mask) 模拟真实的流式实时推理环境
    """
    def __init__(self, config=None, input_dim=43, hidden_dim=64, num_classes=5, num_layers=2):
        super(BaselineTransformer, self).__init__()

        # 动态解析配置参数
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
            batch_first=True  # 保持我们的数据维度为 [Batch, Seq, Feature]
        )
        self.transformer_core = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出头
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x, dt):
        """
        前向传播
        :param x: [Batch, SeqLen, Features]
        :param dt: Transformer 虽然接收 dt，但不会利用它，体现了离散时间的局限性
        """
        seq_len = x.size(1)
        device = x.device

        # 1. 归一化与特征投影
        x_normlized = self.input_norm(x)
        x_feature = self.feature_projection(x_normlized)

        # 2. 加入离散位置编码
        # x_feature = self.pos_encoder(x_feature)

        # 3. 【方案一：架构对齐】生成因果注意力掩码 (Causal Mask)
        # 生成一个 [seq_len, seq_len] 的矩阵，对角线及左下角为 0，右上角（未来时间步）为 -inf
        # 迫使 Transformer 变成单向自回归模型，无法作弊看到未来的帧
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device) * float('-inf'),
            diagonal=1
        )

        # 4. 自注意力时序推演 (带上 mask)
        # 在 batch_first=True 的情况下，mask 直接作用在序列的时间步上
        tf_out = self.transformer_core(x_feature, mask=causal_mask)

        # 5. 降维回归
        logits = self.event_head(tf_out)

        return logits