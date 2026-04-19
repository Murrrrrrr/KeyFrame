import torch
import torch.nn as nn

class CfCCell(nn.Module):
    """
    闭式连续时间单元 (Closed-form Continuous-time Cell)
    液态神经网络的核心动力学推演模块，能够处理非均匀采样的时间序列
    """
    def __init__(self, input_dim, hidden_dim):
        super(CfCCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 时间常数门控网络
        # 负责根据当前输入和历史状态，动态推算时间缩放因子 tau
        self.time_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Sigmoid(), # 限制时间门控在 （0，1） 之间
        )

        # 状态推演主干网络
        self.state_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh() # Tanh 激活函数
        )

    def forward(self, x, hx, dt):
        """
        :param x: 当前时刻的输入特征 [Batch, input_dim]
        :param hx: 上一时刻的隐状态 [Batch, hidden_dim]
        :param dt: 距离上一帧的物理时间间隔 [Batch, 1]
        """
        # 融合输入特征与历史状态
        combined = torch.cat([x, hx], dim=-1)

        # 动态时间常数推断
        tau = self.time_net(combined)

        # 计算基于真实物理时间的时间衰减因子 （Liquid 核心机制）
        # e^(-dt * tau)
        decay = torch.exp(-dt * tau)

        # 计算连续时间推演的目标状态
        target_state = self.state_net(combined)

        # 闭式状态更新
        # 新状态 = 旧状态的自然衰减 + 目标状态随时间的积累补偿
        new_hx = hx * decay + target_state * (1.0 - decay)
        return new_hx