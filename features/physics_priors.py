import torch
import torch.nn as nn

class PhysicsPriorExtractor(nn.Module):
    def __init__(self,pelvis_idx,
                 left_heel_idx, right_heel_idx,
                 left_toe_idx, right_toe_idx,
                 left_ankle_idx, right_ankle_idx,
                 smooth_velocity=True,
                 toe_weight=0.5, ankle_weight=0.8):
        """
        M-Zeni (骨盆质心 + 足跟 + 足尖 + 双踝联合约束)的 61 维特征提取
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
        self.left_heel_idx = left_heel_idx
        self.right_heel_idx = right_heel_idx
        self.left_toe_idx = left_toe_idx
        self.right_toe_idx = right_toe_idx
        self.left_ankle_idx = left_ankle_idx
        self.right_ankle_idx = right_ankle_idx

        self.smooth_velocity = smooth_velocity
        self.toe_weight = toe_weight
        self.ankle_weight = ankle_weight

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
        velocity_features = torch.zero_like(spatial_features)
        dt_safe = torch.clamp(dt_seq, min=1e-4) #防止硬件时钟断流导致除零
        vel = (spatial_features[:, 1:, :] - spatial_features[:, :-1, :]) / dt_safe[:, 1:, :]

        if self.smooth_velocity:
            # EMA(指数移动平均) 抗混叠滤波， 消除单目相机的果冻效应毛刺
            alpha = 0.7
            smoothed_vel = torch.zeros_like(vel)
            smoothed_vel[:, 0, :] = vel[:, 0, :]
            for t in range(1, seq_len - 1):
                smoothed_vel[:, t, :] = alpha * vel[:, t, :] + (1 - alpha) * smoothed_vel[:, t - 1, :]
            velocity_features[:, 1:, :] = smoothed_vel
        else:
            velocity_features[:, 1:, :] = vel

        # 零阶保持器填补第一帧
        velocity_features[:, 0, :] = velocity_features[:, 1, :]

        # M-Zeni 物理先验约束 -> [Batch, 64, 1]
        # 提取相关物理节点的 3D 坐标
        pelvis = pose_seq[:, :, self.pelvis_idx, :]

        heel_l = pose_seq[:, :, self.left_heel_idx, :]
        heel_r = pose_seq[:, :, self.right_heel_idx, :]

        toe_l = pose_seq[:, :, self.left_toe_idx, :]
        toe_r = pose_seq[:, :, self.right_toe_idx, :]

        ankle_l = pose_seq[:, :, self.left_ankle_idx, :]
        ankle_r = pose_seq[:, :, self.right_ankle_idx, :]

        # 计算各节点相对于骨盆(质心)的动态欧氏距离
        dist_heel_l = torch.norm(heel_l - pelvis, p=2, dim=-1, keepdim=True)
        dist_heel_r = torch.norm(heel_r - pelvis, p=2, dim=-1, keepdim=True)

        dist_toe_l = torch.norm(toe_l - pelvis, p=2, dim=-1, keepdim=True)
        dist_toe_r = torch.norm(toe_r - pelvis, p=2, dim=-1, keepdim=True)

        dist_ankle_l = torch.norm(ankle_l - pelvis, p=2, dim=-1, keepdim=True)
        dist_ankle_r = torch.norm(ankle_r - pelvis, p=2, dim=-1, keepdim=True)

        # 构建步态相位流形信号 (Gait Phase Manifold)
        # 这是一个左右脚交替主导的差分合成信号，对 CfC 网络来说是完美的相位时钟
        mzeni_signal = (dist_heel_l - dist_heel_r) + \
                       self.toe_weight * (dist_toe_l - dist_toe_r) + \
                       self.ankle_weight * (dist_ankle_l - dist_ankle_r)

        # 添加极小扰动防止数值下溢
        mzeni_signal = mzeni_signal + 1e-6

        # 多模态总线拼接 -> 
        fused_features = torch.cat([
            spatial_features,  # 30维: 绝对位置状态
            velocity_features,  # 30维: 瞬时运动学状态
            mzeni_signal  # 1维: 融合了 Zeni 理论与踝关节约束的全局时钟
        ], dim=-1)

        return fused_features