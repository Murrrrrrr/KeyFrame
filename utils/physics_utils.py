import numpy as np

def normalize_signal(signal):
    """ 将一维信号归一化到 0~1 之间 """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return (signal - min_val) / (max_val - min_val)

def extract_physics_signal(keypoints_3d, mode='m_zeni', ankle_weight=0.5):
    """
    统一的 M-Zeni 物理先验特征提取器
    :param keypoints_3d: 形状为 (Frames, Joints, 3)
    :param ankle_weight: 踝关节约束权重
    :return: 形状为（Frames, 1）的融合物理判据信号
    """
    # H36M 拓扑的关键点硬编码
    PELVIS_IDX = 0
    R_ANKLE_IDX, R_HEEL_IDX, R_TOE_IDX = 3, 4, 5
    L_ANKLE_IDX, L_HEEL_IDX, L_TOE_IDX = 8, 9, 10

    pelvis = keypoints_3d[:, PELVIS_IDX, :]
    r_ankle, r_heel, r_toe = keypoints_3d[:, R_ANKLE_IDX, :], keypoints_3d[:, R_HEEL_IDX, :], keypoints_3d[
        :, R_TOE_IDX, :]
    l_ankle, l_heel, l_toe = keypoints_3d[:, L_ANKLE_IDX, :], keypoints_3d[:, L_HEEL_IDX, :], keypoints_3d[
        :, L_TOE_IDX, :]

    # 1. 经典Zeni算法(足部-骨盆相对位移)
    zeni_left = np.linalg.norm(l_heel - pelvis, axis=1) + \
                np.linalg.norm(l_toe - pelvis, axis=1) + \
                ankle_weight * np.linalg.norm(l_ankle - pelvis, axis=1)

    zeni_right = np.linalg.norm(r_heel - pelvis, axis=1) + \
                 np.linalg.norm(r_toe - pelvis, axis=1) + \
                 ankle_weight * np.linalg.norm(r_ankle - pelvis, axis=1)

    zeni_signal = np.maximum(zeni_left, zeni_right)
    norm_zeni = normalize_signal(zeni_signal)

    if mode == 'classic_zeni':
        return norm_zeni.reshape(-1, 1) + 1e-6

    elif mode == 'm_zeni':
        m_zeni_signal = np.linalg.norm(l_toe - r_toe, axis=1) + \
                        np.linalg.norm(l_heel - r_heel, axis=1) + \
                        ankle_weight * np.linalg.norm(l_ankle - r_ankle, axis=1)

        norm_m_zeni = normalize_signal(m_zeni_signal)

        # 联合约束归一化融合 (逻辑与门控)
        combined_constraint_signal = norm_zeni * norm_m_zeni
        return combined_constraint_signal.reshape(-1, 1) + 1e-6
    else:
        raise ValueError(f"未知的物理特征模式: {mode}")