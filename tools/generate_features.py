import os
import numpy as np
import glob
import json
import yaml
import torch
from utils.physics_utils import (estimate_forward_direction, project_forward_distance ,physics_aware_normalize,
                                 PELVIS_IDX, L_ANKLE_IDX, R_ANKLE_IDX,
                                 compute_kinematics_derivative, ema_lowpass_filter_tensor)

def normalize_signal(signal):
    """ 将一维信号归一化到 0~1 之间 """
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val == 0:
        return signal
    return (signal - min_val) / (max_val - min_val)

def extract_43d_features(h36m_file_path,fps,ema_alpha,selected_indices, use_normalization=False, extract_m_zeni=True):
    """
    填入软硬件协同特征提取逻辑：单目空间坐标 + M-Zeni规则 + 速度特征
    """
    # 读取 H36M 3D 骨架数据（Frame, Joints, 3）
    keypoints_3d = np.load(h36m_file_path)
    num_frames = keypoints_3d.shape[0]

    # 硬件时间基准
    dt = 1.0 / fps # 视频帧率

    # 选取 7 个关键点构建 21 维空间特征 （必须包含 labels 中用到的 3 个关键点）
    pose_7_joints_abs = keypoints_3d[:, selected_indices, :] # (Frames, 11, 3)

    # 展开为 33 维空间坐标
    spatial_features_abs = pose_7_joints_abs.reshape(num_frames, len(selected_indices) * 3)

    # 临时将 Numpy 转化为 [Batch, SeqLen, Dim] 的张量
    spatial_tensor = torch.tensor(spatial_features_abs, dtype=torch.float32).unsqueeze(0)
    dt_tensor = torch.full((1, num_frames, 1), dt, dtype=torch.float32)
    # 计算运动学速度
    vel_tensor = compute_kinematics_derivative(spatial_tensor, dt_tensor)
    # EMA 抗混叠滤波
    smoothed_vel_tensor = ema_lowpass_filter_tensor(vel_tensor, alpha=ema_alpha)

    # 重新转回 NumPy 矩阵 [SeqLen, Dim]
    velocity_features = smoothed_vel_tensor.squeeze(0).numpy()

    # 绝对的空间去中心化（保证平移不变性）
    pelvis_center = keypoints_3d[:, PELVIS_IDX:PELVIS_IDX+1, :].copy()
    keypoints_3d_centered = keypoints_3d - pelvis_center

    # 提取连续的 1D M-Zeni 信号
    if extract_m_zeni:
        pelvis_pos = keypoints_3d[:, PELVIS_IDX, :]
        l_ankle_pos = keypoints_3d[:, L_ANKLE_IDX, :]
        r_ankle_pos = keypoints_3d[:, R_ANKLE_IDX, :]

        # 估计全局前进方向（3，）
        forward_dir = estimate_forward_direction(pelvis_pos)

        # 计算左右脚踝在前进方向上的投影距离（Frames,）
        dist_ankle_l = project_forward_distance(l_ankle_pos, pelvis_pos, forward_dir)
        dist_ankle_r = project_forward_distance(r_ankle_pos, pelvis_pos, forward_dir)

        mzeni_1d = dist_ankle_l - dist_ankle_r
        mzeni_signal = np.expand_dims(mzeni_1d, axis=-1)
    else:
        mzeni_signal = np.zeros((num_frames, 1), dtype=np.float32)

    # 提取我们真正需要的关键点节点集合 [Frames, Selected_Joints, 3]
    selected_spatial = keypoints_3d_centered[:, selected_indices, :]

    if use_normalization:
        spatial_features, velocity_features, mzeni_signal = physics_aware_normalize(
            selected_spatial, velocity_features, mzeni_signal, p=98
        )
    else:
        spatial_features = selected_spatial.reshape(num_frames, -1)

    fused_features = np.concatenate([spatial_features, velocity_features, mzeni_signal], axis=-1)
    fused_features = fused_features.astype(np.float32)

    if np.isnan(fused_features).all() or np.isinf(fused_features).any():
        print(f"[硬件报警] 提取的特征中包含 NaN 或者 Inf (文件：{h36m_file_path})，已执行截断清洗。")
        fused_features = np.nan_to_num(fused_features, nan=0.0, posinf=5.0, neginf=-5.0)

    return fused_features

def main():
    print("-"*50)
    print(" 开始执行 43维 特征提取")
    print("-"*50)

    # 基础路径配置
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "struct_lnn_mzeni.yaml")
    if not os.path.exists(COFIG_PATH):
        print(f"[错误] 找不到配置文件：{COFIG_PATH}，请确保路径正确")
        return
    with open(COFIG_PATH, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 从data节点中解析配置字典
    FPS = config['data'].get("fps", 120)
    EMA_ALPHA = config['data'].get("ema_alpha", 0.7)
    SELECTED_INDICES = config['data'].get("selected_indices", [0,1,2,3,4,5,6])
    USE_NORM = config['data'].get("use_normalization", True)
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "AthletePose3D", "data")
    OUTPUT_FEATURES_DIT = os.path.join(PROJECT_ROOT, "data", "processed_features")
    SPLIT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "athlete_pose_splits.json")

    if not os.path.exists(SPLIT_JSON_PATH):
        print(f" 找不到数据划分文件：{SPLIT_JSON_PATH}")
        return

    with open(SPLIT_JSON_PATH, "r", encoding='utf-8') as f:
        splits_info = json.load(f)

    allowed_subdirs = list(set([subdir for subdirs in splits_info.get("splits", {}).values() for subdir in subdirs]))

    h36m_files = []
    for subdir in allowed_subdirs:
        target_dir = os.path.join(RAW_DATA_DIR, subdir)
        if os.path.exists(target_dir):
            h36m_files.extend(glob.glob(os.path.join(target_dir, "**", "*_h36m.npy"), recursive=True))

    if not h36m_files:
        print(f"没有找到任何 *_h36m.npy 文件， 请检查原始数据目录")
        return

    processed_count = 0
    for h36m_file in h36m_files:
        rel_path = os.path.relpath(os.path.dirname(h36m_file), RAW_DATA_DIR)
        target_save_dir = os.path.join(OUTPUT_FEATURES_DIT, rel_path)
        os.makedirs(target_save_dir, exist_ok=True)

        # 将解析出来的 YAML 参数动态传入函数
        feature_matrix = extract_43d_features(
            h36m_file_path=h36m_file,
            fps=FPS,
            ema_alpha=EMA_ALPHA,
            selected_indices=SELECTED_INDICES,
            use_normalization=USE_NORM,
        )
        base_filename = os.path.basename(h36m_file)
        feature_name = base_filename.replace("_h36m.npy", "_feature.npy")
        out_path = os.path.join(target_save_dir, feature_name)
        np.save(out_path, feature_matrix)

        print(f" 已生成特征：{out_path} | Shape: {feature_matrix.shape}")
        processed_count += 1

    print("-" * 50)
    print(f" 特征工程执行完毕！共生成 {processed_count} 个 feature.npy 文件。")


if __name__ == "__main__":
    main()