import torch
import torch.nn as nn

from models.backbones.cfc_core import CfCCell
from models.heads.event_head import EventHead

class StructLNN(nn.Module):
    """
    Struct-LNN 核心决策网络
    整合多模态物理先验特征（空间/速度/M-Zeni）,经由液态连续事件网络处理，输出关键帧预测
    """
    def __init__(self, config=None):
        super(StructLNN, self).__init__()

        # 从 YAML 配置安全解析模型超参数
        model_cfg = config.get("model", {}) if config else {}
        self.input_dim = model_cfg.get("input_dim", 43)
        self.hidden_dim = model_cfg.get("hidden_dim", 64)
        self.num_classes = model_cfg.get("num_classes", 5)
        self.time_scale = model_cfg.get("time_scale", 10.0)

        # 物理特征投影层（感知层融合与升维）
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )

        # 实例化连续时间 RNN 核心模块
        self.rnn_cell = CfCCell(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim)

        # 实例化任务接码头
        self.event_head = EventHead(hidden_dim=self.hidden_dim, num_classes=self.num_classes)

    def forward(self, x, dt):
        """
        前向传播总线
        :param x: 多模态物理特征 [Batch, seq_len, input_dim]
        :param dt: 物理硬件时间间隔 [Batch, seq_len，1]
        """
        batch_size, seq_len, _= x.size()

        # 保留 M-Zeni 等物理信号波形的前提下，将其拉入神经网络适宜的数值区间
        x_normalized = self.input_norm(x)
        x_feature = self.feature_projection(x_normalized)

        # 隐状态初始化
        hx = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []

        # 时序推进（沿时间轴滑窗执行连续时间积分）
        for t in range(seq_len):
            x_t = x_feature[:, t, :]

            # 将基础的 dt 乘上 time_scale, 将微秒级的物理时间映射到神经元易于学习的尺度
            dt_t = dt[:, t, :] * self.time_scale

            hx = self.rnn_cell(x_t, hx, dt_t)
            outputs.append(hx.unsqueeze(1))

        # 内存连续化拼接：[Batch, seq_len, hidden_dim]
        sequence_output = torch.cat(outputs, dim = 1)

        # 降维回归，得出全序列的关键帧 Logits
        logits = self.event_head(sequence_output)

        return logits