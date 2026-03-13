import os
import numpy as np
import glob
import json

def extract_61d_features(h36m_file_path):
    """
    填入软硬件协同特征提取逻辑：单目空间坐标 + M-Zeni规则 + 速度特征
    """
    # 读取 H36M 3D 骨架数据（Frame, Joints, 3）
    keypoints_3d = np.load(h36m_file_path)
    num_frames = keypoints_3d.shape[0]

    # 硬件时间基准
    dt = 1.0 / 120.0 # 视频帧率

    # 选取 11 个关键点构建 33 维空间特征 （必须包含 labels 中用到的 7 个关键点）
    selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    pose_11_joints = keypoints_3d[:, selected_indices, :] # (Frames, 10, 30)

    # 展开为 33 维空间坐标
    spatial_features = pose_11_joints.reshape(num_frames, 33)

    # 计算 33 维运动学速度特征 (一阶差分 + EMA 滤波， 对应 physics_priors.py)
    vel = np.zeros_like(spatial_features)
    vel[1:] = (spatial_features[1:] - spatial_features[:-1]) / dt
    vel[0] = vel[1] # 零阶保持器填补第一帧

    # EMA(指数移动平均) 抗混叠滤波
    alpha = 0.7
    smoothed_vel = np.zeros_like(vel)
    smoothed_vel[0] = vel[0]
    for t in range(1, num_frames):
        smoothed_vel[t] = alpha * vel[t] + (1 - alpha) * smoothed_vel[t - 1]

    velocity_features = smoothed_vel

    # 计算1 维 M-Zeni 物理先验约束信号
    pelvis = keypoints_3d[:, 0, :]
    r_ankle, r_heel, r_toe = keypoints_3d[:, 3, :], keypoints_3d[:, 4, :], keypoints_3d[:, 5, :]
    l_ankle, l_heel, l_toe = keypoints_3d[:, 8, :], keypoints_3d[:, 9, :], keypoints_3d[:, 10, :]

    # 计算各节点相对于骨盆的动态欧式距离
    dist_heel_l = np.linalg.norm(l_heel - pelvis, axis=1, keepdims=True)
    dist_heel_r = np.linalg.norm(r_heel - pelvis, axis=1, keepdims=True)
    dist_toe_l = np.linalg.norm(l_toe - pelvis, axis=1, keepdims=True)
    dist_toe_r = np.linalg.norm(r_toe - pelvis, axis=1, keepdims=True)
    dist_ankle_l = np.linalg.norm(l_ankle - pelvis, axis=1, keepdims=True)
    dist_ankle_r = np.linalg.norm(r_ankle - pelvis, axis=1, keepdims=True)

    toe_weight = 0.5
    ankle_weight = 0.8

    # 构建步态相位流行信号，即确认何时使用M-Zeni规则
    mzeni_signal = (dist_heel_l - dist_heel_r) + \
        toe_weight * (dist_toe_l - dist_toe_r) + \
        ankle_weight * (dist_ankle_l - dist_ankle_r)
    mzeni_signal = mzeni_signal + 1e-6  # 添加极小扰动防止数值下溢

    # 多模态总线拼接： 33 + 33 + 1 = 67 维
    fused_features = np.concatenate([spatial_features, velocity_features, mzeni_signal], axis=1)
    fused_features = fused_features.astype(np.float32)

    if np.isnan(fused_features).any():
        print(f"[硬件报警] 提取的特征中包含 NaN (文件：{h36m_file_path})，已自动补0。")
        fused_features = np.nan_to_num(fused_features)

    return fused_features

def main():
    print("-"*50)
    print(" 开始执行 67维 特征提取管道")
    print("-"*50)

    # 基础路径配置
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "AthletePose3D", "data")
    OUTPUT_FEATURES_DIT = os.path.join(PROJECT_ROOT, "data", "processed_features")

    SPLIT_JSON_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "athlete_pose_splits.json")

    if not os.path.exists(SPLIT_JSON_PATH):
        print(f" 找不到数据划分文件：{SPLIT_JSON_PATH}")
        return

    with open(SPLIT_JSON_PATH, "r", encoding='utf-8') as f:
        splits_info = json.load(f)

    # 收集 JSON 中声明的所有受试者子目录（合并 train, valid, test）
    allowed_subdirs = []
    for mode, subdirs in splits_info.get("splits", {}).items():
        allowed_subdirs.extend(subdirs)

    # 去重 （防止 JSON 中声明的所有受试者子目录）
    allowed_subdirs = list(set(allowed_subdirs))
    print(f" 从 JSON 读取到 {len(allowed_subdirs)} 个合法子目录，准备扫描... ")

    # 精确扫描
    h36m_files = []
    for subdir in allowed_subdirs:
        # 拼接出精准的搜索根目录
        target_dir = os.path.join(RAW_DATA_DIR, subdir)

        if not os.path.exists(target_dir):
            print(f" [IO 警告] JSON中声明的目录不存在，跳过：{target_dir}")
            continue

        # 在指定目录下搜索
        search_pattern = os.path.join(target_dir, "**", "*_h36m.npy")
        matched = glob.glob(search_pattern, recursive=True)
        h36m_files.extend(matched)

    if not h36m_files:
        print(f"没有找到任何 *_h36m.npy 文件， 请检查原始数据目录")
        return

    # 特征的提取与保存
    processed_count = 0
    for h36m_file in h36m_files:
        # 保持与原始数据相同的目录树结构
        rel_path = os.path.relpath(os.path.dirname(h36m_file), RAW_DATA_DIR)
        target_save_dir = os.path.join(OUTPUT_FEATURES_DIT, rel_path)
        os.makedirs(target_save_dir, exist_ok=True)

        # 提取特征
        feature_matrix = extract_61d_features(h36m_file)

        out_path = os.path.join(target_save_dir, "feature.npy")
        np.save(out_path, feature_matrix)

        print(f" 已生成特征：{out_path} | Shape: {feature_matrix.shape}")
        processed_count += 1

    print("-"*50)
    print(f" 特征工程执行完毕！共生成 {processed_count} 个 feature.npy 文件。")

if __name__ == "__main__":
    main()