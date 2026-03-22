import torch
import torch.nn as nn

class CfCCell(nn.Module):
    """
    闭式连续时间核心单元（Closed-form Continuous Core）
    负责处理非均匀采样和物理时钟抖动 (Jitter)
    """
    def __init__(self, input_dim, hidden_dim):
        super(CfCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 神经主干 --计算稳态目标
        self.bb_x = nn.Linear(input_dim, hidden_dim)
        self.bb_h = nn.Linear(hidden_dim, hidden_dim)

        # 时间门控机制 --计算衰减率与偏移
        self.time_scale_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.time_shift_fc = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, hx, dt):
        """
        :param x: [Batch, input_dim] 当前帧特征
        :param hx: [Batch, hidden_dim] 上一帧隐藏状态
        :param dt: [Batch,1] 硬件物理时间间隔 \Delta t
        :return: [Batch, hidden_dim]
        """
        # 计算隐藏状态的稳态候选值
        state_update = torch.tanh(self.bb_x(x) + self.bb_h(hx))

        # 拼接特征用于计算 ODE 时间参数
        xh_cat = torch.cat([x, hx], dim=-1)
        time_scale = torch.sigmoid(self.time_scale_fc(xh_cat))
        time_shift = self.time_shift_fc(xh_cat)

        # CfC 闭式连续时间求解核心公式
        exponent = -time_scale * dt - time_shift
        exponent = torch.clamp(exponent, max=0.0)
        decay = torch.exp(exponent) # 保证 decay 永远在 (0, 1] 之间
        h_new = hx * decay + state_update * (1.0 - decay)

        return h_new