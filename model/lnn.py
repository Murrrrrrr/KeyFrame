import torch
import torch.nn as nn
from model.backbones.cfc_core import CfCCell


class StructLNN(nn.Module):
    """
    融合物理约束的液态神经网络 (Struct-LNN)
    完整流水线：43维输入特征 -> 投影降噪 -> CfC动力学主干 -> 5分类软标签输出
    """

    def __init__(self, input_dim: int = 43, hidden_dim: int = 64, num_classes: int = 5):
        super(StructLNN, self).__init__()
        self.hidden_dim = hidden_dim

        # 1. 空间特征投影层 (Spatial Projection)
        # 将 43 维的杂揉特征升维/降维到统一的隐空间，提取高级语义
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)  # 增加正则化，防止过拟合
        )

        # 2. 时序动力学主干 (Temporal Backbone)
        self.rnn_cell = CfCCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # 3. 任务输出头 (Event Head)
        # 将 CfC 的隐藏状态解码为 5 个物理事件的 Logits
        self.event_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
            # 注意：此处不加 Sigmoid！因为 PyTorch 的 BCEWithLogitsLoss 自带了更稳定的 Sigmoid 运算。
        )

    def forward(self, x: torch.Tensor, delta_t: float = 1.0):
        """
        :param x: 视频时序特征张量 [Batch, Seq_Len, Feature_Dim] (例如: [16, 120, 43])
        :param delta_t: 物理时间步长
        :return: 逐帧预测结果 [Batch, Seq_Len, Num_Classes]
        """
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态 (全零向量)
        hx = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 存储所有时间步的输出
        outputs = []

        # 沿时间轴逐帧展开推理
        for t in range(seq_len):
            # 取出第 t 帧的特征 [Batch, 43]
            xt = x[:, t, :]

            # 投影到隐空间
            pt = self.feature_proj(xt)

            # CfC 动力学更新状态
            hx = self.rnn_cell(pt, hx, delta_t)

            # 通过事件头输出当前帧的 5 通道分类 Logits
            out_t = self.event_head(hx)
            outputs.append(out_t)

        # 将所有帧的结果沿 Seq_Len 维度拼接 [Batch, Seq_Len, Num_Classes]
        return torch.stack(outputs, dim=1)


# ================= 测试脚手架 =================
if __name__ == "__main__":
    # 模拟我们 DataLoader 吐出的数据: Batch=16, 120帧, 43维特征
    dummy_input = torch.randn(16, 120, 43)

    # 实例化网络
    model = StructLNN(input_dim=43, hidden_dim=64, num_classes=5)

    # 前向传播测试
    logits = model(dummy_input)

    print("-" * 50)
    print(f"模型组装成功！")
    print(f"输入数据维度: {dummy_input.shape}")
    print(f"网络参数总量: {sum(p.numel() for p in model.parameters() if p.requires_grad)} 个")
    print(f"输出 Logits 维度: {logits.shape}")  # 预期: [16, 120, 5]
    print("-" * 50)