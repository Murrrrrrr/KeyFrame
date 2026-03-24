from os import name

import numpy as np
import torch
from scipy.signal import find_peaks

# H36M 拓扑关键点
PELVIS_IDX = 0
R_ANKLE_IDX = 3
L_ANKLE_IDX = 6
R_HEEL_IDX, R_TOE_IDX = 4, 5
L_HEEL_IDX, L_TOE_IDX = 9, 10

def ema_lowpass_filter_tensor(signal, alpha=0.7):
    """
    一阶 EMA 低通滤波
    用于消除单目相机的果冻效应毛刺和硬件时钟抖动
    :param signal: [Batch, SeqLen, FeatureDim] 的张量
    :param alpha: 滤波系数，越大越平滑但延迟越高
    """
    smoothed_signal = torch.zeros_like(signal)
    smoothed_signal[:, 0, :] = signal[:, 0, :]
    for t in range(1, signal.size(1)):
        smoothed_signal[:, t, :] = alpha * signal[:, t, :] + (1 - alpha) * smoothed_signal[:, t - 1, :]
    return smoothed_signal

def compute_kinematics_derivative(spatial_features, dt_seq, min_dt=1e-4):
    """
    计算基于物理时间的运动学差分
    :param spatial_features: 空间坐标特征 [Batch, SeqLen, Dim]
    :param dt_seq: 物理时间间隔 [Batch, SeqLen, 1]
    :param min_di: 防止硬件时钟断流导致除零的安全阈值
    """
    velocity = torch.zeros_like(spatial_features)
    dt_safe = torch.clamp(dt_seq, min=min_dt)
    # 前向差分计算
    vel_diff = (spatial_features[:, 1:, :] - spatial_features[:, :-1, :]) / dt_safe[:, 1:, :]
    velocity[:, 1:, :] = vel_diff
    # 零阶保持器填补第一帧
    velocity[:, 0, :]= velocity[:, 1, :]
    return velocity

def robust_normalize_phase(signal):
    """ 鲁棒相位归一化，映射到 [-1, 1] """
    # 使用分位数代替min/max，防止单帧飞点毁掉整个序列
    p_min = np.percentile(signal, 2)
    p_max = np.percentile(signal, 98)

    # 截断奇异点
    signal_clipped = np.clip(signal, p_min, p_max)

    if p_max - p_min==0:
        return signal_clipped - p_min

    # 将信号映射到 [-1, 1]，0 代表双腿交汇的对称状态
    norm_signal = 2.0 * (signal_clipped - p_min) / (p_max - p_min) - 1.0
    return norm_signal

def estimate_forward_direction(pelvis_pos):
    """
    根据骨盆轨迹估计运动前进方向
    :param pelvis_pos:
    :return:
    """
    velocity = np.diff(pelvis_pos, axis=0)
    forward = np.mean(velocity, axis=0)
    norm = np.linalg.norm(forward)
    if norm < 1e-6:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / norm
    return forward

def project_forward_distance(foot_pos, pelvis_pos, forward_dir):
    """
    前进方向投影距离
    :param foot_pos:
    :param pelvis_pos:
    :param forward_dir:
    :return:
    """
    relative = foot_pos - pelvis_pos
    return np.dot(relative, forward_dir)

def extract_m_zeni(keypoint_3d, min_frames_between_steps=35, prominence=0.05, cross_limb_weight=0.5, tolerance=18):
    """
    基于 M-Zeni 物理规则提取跑姿关键帧 （剥离归一化，直接计算）
    只适配于 H36M 17关节骨架格式
    :param keypoint_3d: 形状为 (Frames, Joints, 3) 的 3D 骨架坐标。
    :param min_frames_between_steps: 寻峰的最小帧间隔（控制容差）
    :param prominence: 寻峰的最小突起度， 过滤微小抖动噪声
    :param cross_limb_weight: 交叉肢体权重，用于缩放距离特征
    :return:
        dict: 包含 “Left_Extreme” 和 "Right_Extreme" 的字典，值为帧索引数组。
    """
    frames = keypoint_3d.shape[0]
    if frames == 0:
        return {}

    # 提取骨盆和左右脚踝的原始三维坐标
    pelvis_pos = keypoint_3d[:, PELVIS_IDX, :]
    l_ankle_pos = keypoint_3d[:, L_ANKLE_IDX, :]
    r_ankle_pos = keypoint_3d[:, R_ANKLE_IDX, :]

    # 估计前进方向
    forward_dir = estimate_forward_direction(pelvis_pos)

    # Zeni 距离
    l_ankle_dist = project_forward_distance(l_ankle_pos, pelvis_pos, forward_dir)
    r_ankle_dist = project_forward_distance(r_ankle_pos, pelvis_pos, forward_dir)

    # 寻找极值
    # HS 最大值
    l_hs, _ = find_peaks(
        l_ankle_dist,
        distance=min_frames_between_steps,
        prominence=prominence,
    )
    r_hs, _ = find_peaks(
        r_ankle_dist,
        distance=min_frames_between_steps,
        prominence=prominence,
    )
    # TO 最小值
    l_to, _ = find_peaks(
        -l_ankle_dist,
        distance=min_frames_between_steps,
        prominence=prominence,
    )
    r_to, _ = find_peaks(
        -r_ankle_dist,
        distance=min_frames_between_steps,
        prominence=prominence,
    )

    # m-zeni 创新足尖规则
    toe_distance = np.linalg.norm(
        l_ankle_pos - r_ankle_pos,
        axis=1
    )

    toe_min_distance = max(1, min_frames_between_steps // 2)
    toe_max_peaks, _ = find_peaks(
        toe_distance,
        distance=toe_min_distance,
        prominence=prominence * cross_limb_weight,
    )
    def validate(peaks, name="Event"):
        if len(peaks) == 0 or len(toe_max_peaks) == 0:
            return np.array([])
        valid = []
        for p in peaks:
            if np.any(np.abs(toe_max_peaks - p) <= tolerance):
                valid.append(p)
        print(f"[{name}] 验证前：{len(peaks)}帧，验证后：{len(valid)}帧")
        return np.array(valid, dtype=int)

    l_hs = validate(l_hs,"Left_HS")
    r_hs = validate(r_hs,"Right_HS")
    l_to = validate(l_to,"Left_TO")
    r_to = validate(r_to,"Right_TO")

    return {
        "Left_HS": l_hs,
        "Left_TO": l_to,
        "Right_HS": r_hs,
        "Right_TO": r_to,
        "Toe_Max": toe_max_peaks,
        "raw_l_heel_dist": l_ankle_dist,
        "raw_r_heel_dist": r_ankle_dist,
    }