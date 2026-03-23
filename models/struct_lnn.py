from distutils.command.config import config

import torch
import torch.nn as nn

from models.backbones.cfc_core import CfCCell
from models.heads.event_head import EventHead

class StructLNN(nn.Module):
    """
    Struct-LNN 核心决策网络
    整合 67 维多模态物理先验特征，经由液态连续时间网络，输出关键帧预测
    """
    def __init__(self, config=None, input_dim=67, hidden_dim=128, num_classes=1):
        super(StructLNN, self).__init__()

        # 从 config 字典传入，方便后续在 train.py 统一管理
        if config is not None:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_dim = model_cfg.get('hidden_dim', hidden_dim)
            num_classes = model_cfg.get('num_classes', num_classes)

        self.hidden_dim = hidden_dim
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

    def forward(self, batch_data):
        """
        前向传播总线
        :param x: [Batch, seq_len, 61]
        :param dt: [Batch, seq_len, 1]
        """
        x, dt = batch_data
        batch_size ,seq_len, _ = x.size()
        # 保留 M-Zeni 等物理信号波形的前提下，将其拉入神经网络适宜的数值区间
        x_normlized = self.input_norm(x)

        # 特征升维
        x_feature = self.feature_projection(x_normlized)

        # 隐状态初始化（硬件端初始化为0）
        hx = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs = []

        # 时序推进（沿时间抽滑窗）
        for t in range(seq_len):
            x_t = x_feature[:, t, :]
            dt_t = dt[:, t, :] * 10
            hx = self.rnn_cell(x_t, hx, dt_t)
            outputs.append(hx.unsqueeze(1))

        # 内存连续化拼接
        sequence_output = torch.cat(outputs, dim=1)

        # 降维回归
        logits = self.event_head(sequence_output)
        return logits

# 硬件连通性自检
if __name__ == "__main__":
    print("正在验证 StructLNN 软硬件协同架构维度...")
    # 模拟从 pose_dataset.py 传过来的一个 Batch
    dummy_x = torch.randn(32, 64, 67)  # [Batch, SeqLen, Features]
    dummy_dt = torch.abs(torch.randn(32, 64, 1)) * 0.00833  # 模拟 50FPS 左右的 dt

    model = StructLNN(input_dim=67, hidden_dim=128, num_classes=1)

    # 将模型和数据推入显存（如果有GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dummy_x, dummy_dt = dummy_x.to(device), dummy_dt.to(device)

    out_logits = model((dummy_x, dummy_dt))

    print(f"输入 X 维度: {dummy_x.shape}")
    print(f"输入 dt 维度: {dummy_dt.shape}")
    print(f"最终输出 Logits 维度: {out_logits.shape} (预期: [32, 64, 1])")
    print("架构组装完美！")