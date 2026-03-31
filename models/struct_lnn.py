import torch
import torch.nn as nn

from models.backbones.cfc_core import CfCCell
from models.heads.event_head import EventHead

class StructLNN(nn.Module):
    """
    Struct-LNN 核心决策网络
    整合 67 维多模态物理先验特征，经由液态连续时间网络，输出关键帧预测
    """
    def __init__(self, config=None, input_dim=67, hidden_dim=128, num_classes=1, time_scale=10.0):
        super(StructLNN, self).__init__()

        # 从 config 字典传入，方便后续在 train.py 统一管理
        if config is not None and 'model' in config:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_dim = model_cfg.get('hidden_size', hidden_dim)
            num_classes = model_cfg.get('num_classes', num_classes)
            time_scale = model_cfg.get('time_scale', time_scale)

        self.hidden_dim = hidden_dim
        self.time_scale = time_scale
        self.input_norm = nn.LayerNorm(input_dim)

        # 物理特征投影层（感知层融合）
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 动力学演化主干（决策层流体态）
        self.rnn_cell = CfCCell(input_dim=hidden_dim, hidden_dim=hidden_dim)

        # 任务输出头（执行层）
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x,dt):
        """
        前向传播总线
        :param x: [Batch, seq_len, 61]
        :param dt: [Batch, seq_len, 1]
        """
        batch_size ,seq_len, _ = x.size()
        # 保留 M-Zeni 等物理信号波形的前提下，将其拉入神经网络适宜的数值区间
        x_normalized = self.input_norm(x)

        # 特征升维
        x_feature = self.feature_projection(x_normalized)

        # 隐状态初始化（硬件端初始化为0）
        hx = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []

        # 时序推进（沿时间抽滑窗）
        for t in range(seq_len):
            x_t = x_feature[:, t, :]
            dt_t = dt[:, t, :] * self.time_scale
            hx = self.rnn_cell(x_t, hx, dt_t)
            outputs.append(hx.unsqueeze(1))

        # 内存连续化拼接
        sequence_output = torch.cat(outputs, dim=1)

        # 降维回归
        logits = self.event_head(sequence_output)
        return logits