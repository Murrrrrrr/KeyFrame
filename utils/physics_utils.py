import numpy as np

# H36M 拓扑关键点
PELVIS_IDX = 0
R_ANKLE_IDX, R_HEEL_IDX, R_TOE_IDX = 3, 4, 5
L_ANKLE_IDX, L_HEEL_IDX, L_TOE_IDX = 8, 9, 10

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


def extract_m_zeni(keypoints_3d, ankle_weight=0.5):
    """
    提取 M-Zeni 多尺度一致性相位 (局部躯干参考 + 全局跨肢体关系)
    """
    pelvis = keypoints_3d[:, PELVIS_IDX, :]
    r_ankle, r_heel, r_toe = keypoints_3d[:, R_ANKLE_IDX, :], keypoints_3d[:, R_HEEL_IDX, :], keypoints_3d[
        :, R_TOE_IDX, :]
    l_ankle, l_heel, l_toe = keypoints_3d[:, L_ANKLE_IDX, :], keypoints_3d[:, L_HEEL_IDX, :], keypoints_3d[
        :, L_TOE_IDX, :]

    # 局部躯干参考相位
    zeni_left = np.linalg.norm(l_heel - pelvis, axis=1) + np.linalg.norm(l_toe - pelvis,
                                                                         axis=1) + ankle_weight * np.linalg.norm(
        l_ankle - pelvis, axis=1)
    zeni_right = np.linalg.norm(r_heel - pelvis, axis=1) + np.linalg.norm(r_toe - pelvis,
                                                                          axis=1) + ankle_weight * np.linalg.norm(
        r_ankle - pelvis, axis=1)
    zeni_phase_clock = zeni_left - zeni_right

    # 全局跨肢体协同规律 (ToeL - ToeR 的差分向量长度，本身就具有周期性)
    m_zeni_distance = np.linalg.norm(l_toe - r_toe, axis=1) + \
                      np.linalg.norm(l_heel - r_heel, axis=1) + \
                      ankle_weight * np.linalg.norm(l_ankle - r_ankle, axis=1)

    # 为了保持方向性（相位），我们让跨肢体距离带有正负号（跟随 zeni_phase_clock 的符号）
    m_zeni_phase_clock = m_zeni_distance * np.sign(zeni_phase_clock)

    # 物理一致性融合：放弃会导致梯度消失和波形尖锐的乘法，改用加权线性叠加
    # 这样既能融合“单腿”和“双腿”的运动学规律，又能完美保持极限环流形的平滑性
    fused_phase_clock = 0.5 * zeni_phase_clock + 0.5 * m_zeni_phase_clock

    norm_fused_phase = robust_normalize_phase(fused_phase_clock)
    return norm_fused_phase.reshape(-1, 1)

def extract_classic_zeni(keypoints_3d, ankle_weight=0.5):
    """
    提取经典 Zeni 物理相位 (构建连续的差分时钟)
    """
    pelvis = keypoints_3d[:, PELVIS_IDX, :]
    r_ankle = keypoints_3d[:, R_ANKLE_IDX, :]
    r_heel = keypoints_3d[:, R_HEEL_IDX, :]
    r_toe = keypoints_3d[:, R_TOE_IDX, :]

    l_ankle = keypoints_3d[:, L_ANKLE_IDX, :]
    l_heel = keypoints_3d[:, L_HEEL_IDX, :]
    l_toe = keypoints_3d[:, L_TOE_IDX, :]

    # 分别计算左腿和右腿的伸展程度
    zeni_left = np.linalg.norm(l_heel - pelvis, axis=1) + \
                np.linalg.norm(l_toe - pelvis, axis=1) + \
                ankle_weight * np.linalg.norm(l_ankle - pelvis, axis=1)

    zeni_right = np.linalg.norm(r_heel - pelvis, axis=1) + \
                 np.linalg.norm(r_toe - pelvis, axis=1) + \
                 ankle_weight * np.linalg.norm(r_ankle - pelvis, axis=1)

    # 生成一个在正负之间振荡的波形，正数代表左腿在前，负数代表右腿在前
    zeni_phase_clock = zeni_left - zeni_right

    norm_zeni_phase = robust_normalize_phase(zeni_phase_clock)
    return norm_zeni_phase.reshape(-1, 1)