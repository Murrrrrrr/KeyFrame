"""在线计算 feature """
import torch
import torch.nn as nn

from utils.physics_utils import compute_kinematics_derivative, ema_lowpass_filter_tensor

class PhysicsPriorExtractor(nn.Module):
    def __init__(self,pelvis_idx,
                 left_ankle_idx, right_ankle_idx,
                 left_toe_idx, right_toe_idx,
                 left_heel_idx, right_heel_idx,
                 smooth_velocity=True,
                 toe_weight=0.5, ankle_weight=0.8,
                 bidirectional_ema=True, max_dt=0.1):
        """
        M-Zeni (骨盆质心 + 双踝联合约束)的 43 维特征提取
        :param pelvis_idx: 骨盆索引 - 经典 Zeni 算法参考系原点
        :param left_heel_idx, right_heel_idx: 足跟索引 - 用于捕捉 Heel Strike（HS）
        :param left_toe_idx, right_toe_idx: 足尖索引 - 用于捕捉 （TO）
        :param left_ankle_idx, right_ankle_idx: 踝关节索引 - 用户的创新刚性约束
        :param smooth_velocity: 是否对速度计算进行一阶 EMA 低通滤波 (抗边缘设备时钟抖动)
        :param toe_weight: 足尖位移的权重 (默认 0.5)
        :param ankle_weight: 踝关节刚性约束的权重 (默认 0.8，因为踝关节在视觉上最稳定)
        """
        super().__init__()
        self.pelvis_idx = pelvis_idx
        self.left_ankle_idx = left_ankle_idx
        self.right_ankle_idx = right_ankle_idx
        self.left_toe_idx = left_toe_idx
        self.right_toe_idx = right_toe_idx
        self.left_heel_idx = left_heel_idx
        self.right_heel_idx = right_heel_idx

        self.smooth_velocity = smooth_velocity
        self.toe_weight = toe_weight
        self.ankle_weight = ankle_weight
        self.bidirectional_ema = bidirectional_ema
        self.max_dt = max_dt

    def forward(self, pose_seq, dt_seq):
        """
        :param pose_seq: [Batch, 64, 10, 3] 原始下肢 3D 坐标
        :param dt_seq: [Batch, 64, 1] 硬件采样物理时间间隔
        :return: [Batch, 64, 61] 融合完整物理先验的特征张量（30维空间 + 30维速度 + 1维 M-Zeni）
        """
        batch_size, seq_len, num_kpts, dims = pose_seq.shape

        # 空间坐标展平
        spatial_features = pose_seq.view(batch_size, seq_len, num_kpts * dims)

        # 一阶运动学差分
        velocity_features = compute_kinematics_derivative(spatial_features, dt_seq, max_dt=self.max_dt)
        if self.smooth_velocity:
            velocity_features = ema_lowpass_filter_tensor(velocity_features, alpha=0.7, bidirectional=self.bidirectional_ema)

        # 提取相关的物理节点的3D坐标  [Batch, SeqLen, 3]
        pelvis = pose_seq[:, :, self.pelvis_idx, :]
        ankle_l, ankle_r = pose_seq[:, :, self.left_ankle_idx, :], pose_seq[:, :, self.right_ankle_idx, :]

        # 动态估计每个 Batch 序列的运动前进方向 [Batch, 1, 3]
        pelvis_vel = pelvis[:, 1:, :] - pelvis[:, :, -1, :]
        forward_dir = torch.mean(pelvis_vel, dim=1, keepdim=True)
        # 归一化方向向量
        forward_norm = torch.norm(forward_dir, p=2, dim=-1, keepdim=True) + 1e-6
        forward_dir = forward_dir / forward_norm

        # 投影辅助函数：计算相对坐标在前进方向上的投影
        def project_forward(node, origin, direction):
            relative_pos = node - origin
            return torch.sum(relative_pos * direction, dim=-1, keepdim=True)
        dist_ankle_l = project_forward(ankle_l, pelvis, forward_dir)
        dist_ankle_r = project_forward(ankle_r, pelvis, forward_dir)
        mzeni_signal = (dist_ankle_l - dist_ankle_r) + 1e-6

        # 多模态总线拼接 -> [Batch, SeqLen, 61]
        fused_features = torch.cat([
            spatial_features,  # 30维: 绝对位置状态
            velocity_features,  # 30维: 瞬时运动学状态
            mzeni_signal  # 1维: 融合了 Zeni 理论与踝关节约束的全局时钟
        ], dim=-1)

        return fused_features