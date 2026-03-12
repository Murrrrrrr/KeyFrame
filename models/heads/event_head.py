import torch.nn as nn

class EventHead(nn.Module):
    """
    关键帧事件检测头
    将高维潜空间状态降维回归至 1D Logits
    """
    def __init__(self, hidden_dim, num_classes=1, dropout_rate=0.2):
        super(EventHead, self).__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(), # 平滑激活，防止突变梯度
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, sequence_output):
        """
        :param sequence_output: [Batch, seq_len, hidden_dim] 时序特征张量
        :return: logits: [Batch, seq_len, num_classes] 预测的概率分布
        """
        return self.head(sequence_output)