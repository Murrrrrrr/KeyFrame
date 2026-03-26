import math
from os import name

import numpy as np
import torch
from scipy.signal import find_peaks

# H36M 拓扑关键点
PELVIS_IDX = 0
R_HIP_IDX = 1
R_KNEE_IDX = 2
R_ANKLE_IDX = 3
L_HIP_IDX = 4
L_KNEE_IDX = 5
L_ANKLE_IDX = 6


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

    events_dict = {
        "Left_HS": l_hs,
        "Left_TO": l_to,
        "Right_HS": r_hs,
        "Right_TO": r_to,
        "Toe_Max": toe_max_peaks,
    }

    soft_labels = generate_gaussian_soft_labels(frames, events_dict, sigma=3.6)

    return {
        "Left_HS": l_hs,
        "Left_TO": l_to,
        "Right_HS": r_hs,
        "Right_TO": r_to,
        "Toe_Max": toe_max_peaks,
        "raw_l_heel_dist": l_ankle_dist,
        "raw_r_heel_dist": r_ankle_dist,
    }

def generate_gaussian_soft_labels(num_frame, event_dict, sigma=1.0):
    """
    将稀疏的帧索引转换为 (Num_Frames, 5) 的高斯软标签矩阵，用于 BCE Loss 多标签分类
    :param num_frame: 视频序列的总帧数
    :param event_dict: extract_m_zeni 返回的包含步态事件帧索引的字典
    :param sigma: 高斯核标准差，sigma越大，软标签影响的相邻帧越宽。
                  若 fps=120, sigma=1.0 大约影响前后 2-3 帧
    :return: (num_frames, 5) 的 numpy 数组，值域在 [0, 1] 之间
    """
    # 初始化全零矩阵，形状为 (帧数，5个类别通道)
    # 通道定义：0: Left_HS, 1:Left_TO, 2:Right_HS, 3:Right_TO, 4: Dist_Max (Toe_Max)
    soft_labels = np.zeros((num_frame, 5), dtype=np.float32)
    channel_map = {
        "Left_HS": 0,
        "Left_TO": 1,
        "Right_HS": 2,
        "Right_TO": 3,
        "Toe_Max": 4,
    }

    for event_name, channel_idx in channel_map.items():
        event_frames = event_dict.get(event_name, [])

        for frame in event_frames:
            if not (0 <= int(frame) < num_frame):
                continue

            # 在目标及其前后 3*sigma 的窗口内涂抹高斯分布
            # 这样避免了遍历所有帧
            window = int(np.ceil(3 * sigma))
            start_idx = max(0, frame - window)
            end_idx = min(num_frame, frame + window + 1)

            for i in range(start_idx, end_idx):
                # 计算高斯衰减值: exp(- (x - mu)^2 / (2 * sigma^2))
                decay = math.exp(-((i - frame)**2) / (2 * (sigma ** 2)))
                # 叠加概率 （如果两个同类事件靠的很近，取最大值，避免超过 1.0）
                soft_labels[i, channel_idx] = max(soft_labels[i, channel_idx], decay)
    return soft_labels

def physics_aware_normalize(spatial_centered, velocity, mzeni, p=98):
    """
    物理感知的鲁棒最大绝对值归一化
    针对跑姿数据的多模态特征设计，保留了物理动态和空间长宽比
    :param sptial_centered: [Frames, Joints, 3] 去中心化后的相对 3D 坐标
    :param velocity: [Frames, Feature_Dim] 瞬时速度特征
    :param mzeni: [Frames, 1] M-Zeni 连续物理信号
    :param p: 鲁棒分位数（默认为 98%），用于过滤单目相机的 3D 跟踪飞点噪声
    :return: 归一化后并展平的 (norm_spatial, norm_vel, norm_mzeni)
    """
    import numpy as np
    num_frames = spatial_centered.shape[0]

    # 【空间特征】 等比例缩放
    # 计算所有关节到骨盆的 3D 欧氏距离，找到分位数为 98% 的最大物理半径
    distances = np.linalg.norm(spatial_centered, axis=-1)  # Shape: [Frames, Joints]
    spatial_scale = np.percentile(distances, p)
    spatial_scale = np.maximum(spatial_scale, 1e-6)  # 硬件除零保护

    # 保持长宽比缩放并展平为 [Frames, Joints * 3]
    spatial_flat = spatial_centered.reshape(num_frames, -1)
    norm_spatial = spatial_flat / spatial_scale

    # 【速度特征】 独立的鲁棒最大绝对值缩放
    vel_scale = np.percentile(np.abs(velocity), p)
    vel_scale = np.maximum(vel_scale, 1e-6)
    norm_vel = velocity / vel_scale

    # 【M-Zeni 信号】 独立的鲁棒最大绝对值缩放
    mzeni_scale = np.percentile(np.abs(mzeni), p)
    mzeni_scale = np.maximum(mzeni_scale, 1e-6)
    norm_mzeni = mzeni / mzeni_scale

    return norm_spatial, norm_vel, norm_mzeni