import torch.nn as nn

class EventHead(nn.Module):
    """
    事件检测输出头
    统一的输出接口，不论底层是什么模型
    """
    def __init__(self, hidden_dim, num_classes):
        super(EventHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(), # SiLU 激活函数
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        :param x: 时序骨干网络输出的隐藏状态序列 [Batch, Seq_len, hidden_dim]
        :return: 未经过 Sigmoid 的原始 Logits [Batch, Seq_len, num_classes]
        """
        return self.head(x)