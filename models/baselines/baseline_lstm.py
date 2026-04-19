import torch
import torch.nn as nn
from models.heads.event_head import EventHead


class BaselineLSTM(nn.Module):
    """
    离散时间基线模型：标准 LSTM
    用于评估传统 RNN 在未引入连续时间动力学 (dt) 时，面对传感器丢帧和时钟抖动的表现差距。
    """

    def __init__(self, config=None):
        super(BaselineLSTM, self).__init__()

        # 1. 统一从 config 字典解析参数
        model_cfg = config.get('model', {}) if config else {}
        input_dim = model_cfg.get('input_dim', 43)
        hidden_dim = model_cfg.get('hidden_size', 64)
        num_classes = model_cfg.get('num_classes', 5)
        num_layers = model_cfg.get('num_layers', 2)

        # 2. 构建离散时间的主干网络
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        # 3. 复用我们之前为 Struct-LNN 写的标准化输出头，保证消融实验的绝对公平
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x, dt=None):
        """
        前向传播总线
        :param x: [Batch, seq_len, input_dim] 多模态物理特征
        :param dt: [Batch, seq_len, 1] 物理时间步长
        """
        # 🌟 核心兼容：接收 Trainer 传来的 dt，但完全不使用它。
        # 这就是多态的魅力，LSTM 依然按照“等间距采样”的错误假设进行运算。

        out, _ = self.lstm(x)  # LSTM 只吃特征 x
        logits = self.event_head(out)  # 直接映射到对应的分类 logits
        return logits