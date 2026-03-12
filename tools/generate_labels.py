import os
import numpy as np
import glob
from scipy.signal import find_peaks

def normalize_signal(signal):
    """将一维信号归一化到 0~1 之间，便于不同量纲的特征融合"""
    min_val = np.min(signal)
    max_val = np.max(signal)
    if max_val - min_val ==  0:
        return signal
    return (signal - min_val) / (max_val - min_val)

def generate_labels_for_athlete_pose(dataset_root, output_root, splits):
    """
    根据数据集自带的 3D骨架数据(h36m.npy)自动生成跑姿关键帧标签
    物理依据：M-Zeni规则
    """
    print("开始自动生成跑姿关键帧标签...")

    # H36M 拓扑中，右脚尖标签点是5，左脚尖标签点是10，骨盆标签点是0
    PELVIS_IDX = 0
    LEFT_FOOT_IDX = 10
    RIGHT_FOOT_IDX = 5

    for split_name, floders in splits.items():
        for floder in floders:
            folder_path = os.path.join(dataset_root, floder)
            search_pattern = os.path.join(folder_path, "*_h36m.npy")
            matched_files = glob.glob(search_pattern)

            if not matched_files:
                print(f" 警告：找不到{search_pattern}，跳过该目录。")
                continue

            h36m_path = matched_files[0]

            # shape 为（Frames, Joints, 3）
            keypoints_3d = np.load(h36m_path)
            num_frames = keypoints_3d.shape[0]

            # 剥离左右脚的3D空间坐标
            pelvis_coords = keypoints_3d[:, PELVIS_IDX, :]
            left_foot_coords =  keypoints_3d[:, LEFT_FOOT_IDX, :]
            right_foot_coords = keypoints_3d[:, RIGHT_FOOT_IDX, :]

            #经典Zeni算法(足部-骨盆相对位移)
            zeni_left = np.linalg.norm(left_foot_coords - pelvis_coords, axis=1)
            zeni_right = np.linalg.norm(right_foot_coords - pelvis_coords, axis=1)
            zeni_signal = np.maximum(zeni_left, zeni_right)

            # 计算 M-Zeni 物理特征（双足空间欧式距离）
            m_zeni_signal= np.linalg.norm(left_foot_coords - right_foot_coords, axis=1)

            # 联合约束：将两个信号归一化后进行融合
            # 本实验采用相乘来放大极值瞬间的“尖锐度”，消除单一算法的误检
            norm_zeni = normalize_signal(zeni_signal)
            norm_m_zeni = normalize_signal(m_zeni_signal)
            combined_constraint_signal = norm_zeni * norm_m_zeni # 构建最终的联合物理判据信号

            # 寻峰与生成标签
            peaks, properties = find_peaks(combined_constraint_signal, distance=15, prominence=0.05)

            # 生成对应的独热编码标签
            labels = np.zeros((num_frames, 1), dtype=np.float32)
            labels[peaks] = 1.0
            save_dir = os.path.join(output_root, floder)
            os.makedirs(save_dir, exist_ok=True)
            labels_save_path = os.path.join(save_dir, "labels.npy")
            np.save(labels_save_path, labels)

            print(f" 处理完毕：{floder} | 提取到 M-Zeni 联合关键帧：{len(peaks)}个")

if __name__ == "__main__":
    ROOT_DIR = "D:/Python项目/KeyFrame/AthletePose3D/data"
    OUTPUT_DIR = r"D:\Python项目\KeyFrame\data\processed_labels"
    TARGET_SPLITS = {
        "train": ["train_set/S3", "train_set/S4"],
        "valid": ["valid_set/S2"],
        "test": ["test_set/S2"]
    }
    generate_labels_for_athlete_pose(ROOT_DIR, OUTPUT_DIR, TARGET_SPLITS)