import torch
import torch.nn as nn

# 复用你写好的事件解码头
from models.heads.event_head import EventHead


class BaselineLSTM(nn.Module):
    """
    基线对比网络：传统离散时间 LSTM
    用于证明连续时间建模 (LNN) 在非均匀采样和抖动环境下的优越性
    """

    def __init__(self, config=None, input_dim=43, hidden_dim=64, num_classes=1, num_layers=1):
        super(BaselineLSTM, self).__init__()

        # 从 config 中安全解析参数 (与 StructLNN 保持一致)
        if config is not None and 'model' in config:
            model_cfg = config.get('model', {})
            input_dim = model_cfg.get('input_dim', input_dim)
            hidden_dim = model_cfg.get('hidden_size', hidden_dim)  # 注意 yaml 中是 hidden_size
            num_classes = model_cfg.get('num_classes', num_classes)
            num_layers = model_cfg.get('num_layers', num_layers)

        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)

        # 1. 特征投影层 (与 StructLNN 保持一致的感知层融合)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        # 2. 传统离散时间 RNN 核心 (替换掉 CfCCell)
        # batch_first=True 表示输入张量形状为 (batch, seq, feature)
        self.lstm_core = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 3. 任务输出头 (执行层，完全复用)
        self.event_head = EventHead(hidden_dim=hidden_dim, num_classes=num_classes)

    def forward(self, x,dt):
        """
        前向传播
        :param batch_data: 包含 (x, dt) 的元组，为了保持接口一致
        """
        # 注意：在这里提取了 dt，但 LSTM 根本不使用它！这正是离散模型的致命缺陷。

        # 归一化处理
        x_normlized = self.input_norm(x)

        # 特征升维 [Batch, SeqLen, HiddenDim]
        x_feature = self.feature_projection(x_normlized)

        # LSTM 时序推演
        # lstm_out 的形状为 [Batch, SeqLen, HiddenDim]
        lstm_out, _ = self.lstm_core(x_feature)

        # 降维回归为 logits [Batch, SeqLen, NumClasses]
        logits = self.event_head(lstm_out)

        return logits


# 架构连通性自检
if __name__ == "__main__":
    print("正在验证 BaselineLSTM 基线模型维度...")
    dummy_x = torch.randn(32, 64, 43)  # [Batch, SeqLen, Features]
    dummy_dt = torch.randn(32, 64, 1)  # 模拟 dt，但 LSTM 内部会丢弃它

    model = BaselineLSTM(input_dim=43, hidden_dim=64, num_classes=5)

    out_logits = model((dummy_x, dummy_dt))

    print(f"输入 X 维度: {dummy_x.shape}")
    print(f"最终输出 Logits 维度: {out_logits.shape} (预期: [32, 64, 5])")
    print("基线模型构建成功，可与 StructLNN 进行对比训练！")