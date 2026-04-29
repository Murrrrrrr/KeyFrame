import torch
import torch.nn as nn


class CfCCell(nn.Module):
    """
    闭式连续时间神经网络 (CfC - Closed-form Continuous-time) 核心单元。
    相比于传统 LSTM/GRU，CfC 通过常微分方程的闭式解来更新隐藏状态，
    对高频采样 (120FPS) 和不规则时间间隔 (丢帧/抖动) 具有极强的物理鲁棒性。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super(CfCCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 1. 时间门控网络 (Time-constant gating)
        # 决定当前输入和历史状态应该以多快的速度“遗忘”或“更新”
        self.time_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 引入 LayerNorm 加速收敛
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # 输出范围 (0, 1)，代表时间衰减率
        )

        # 2. 目标状态网络 (Target state representation)
        # 决定如果没有时间流逝，系统最终应该趋向的理想状态
        self.target_state = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()  # 输出范围 (-1, 1)，符合 RNN 隐藏状态的惯例
        )

    def forward(self, x: torch.Tensor, hx: torch.Tensor, delta_t: float = 1.0):
        """
        :param x: 当前帧输入特征 [Batch, Input_Size]
        :param hx: 上一时刻隐藏状态 [Batch, Hidden_Size]
        :param delta_t: 时间间隔 (120FPS下连续帧通常设为 1.0)
        :return: 新的隐藏状态 new_hx
        """
        # 拼接当前输入与历史状态
        cat_input = torch.cat([x, hx], dim=-1)

        # 计算门控常数和目标状态
        tau_inv = self.time_gate(cat_input)
        target_h = self.target_state(cat_input)

        # 核心动力学方程 (Closed-form 近似)
        # 随时间指数衰减的因子：时间隔越久，衰减越厉害
        decay_factor = torch.exp(-tau_inv * delta_t)

        # 液态更新：当前状态 = 历史的残存 + 目标状态的注入
        new_hx = hx * decay_factor + target_h * (1.0 - decay_factor)

        return new_hx