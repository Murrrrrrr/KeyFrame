import os
import numpy as np
import glob
from scipy.signal import find_peaks
from utils.physics_utils import calculate_mzeni_prior

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
    LEFT_FOOT_HEEL_IDX = 9
    RIGHT_FOOT_HEEL_IDX = 4
    LEFT_FOOT_ANKLE_IDX = 8
    RIGHT_FOOT_ANKLE_IDX = 3
    LEFT_FOOT_TOE_IDX = 10
    RIGHT_FOOT_TOE_IDX = 5

    for split_name, floders in splits.items():
        for floder in floders:
            folder_path = os.path.join(dataset_root, floder)
            search_pattern = os.path.join(folder_path, "*_h36m.npy")
            matched_files = glob.glob(search_pattern)

            if not matched_files:
                print(f" 警告：找不到{search_pattern}，跳过该目录。")
                continue

            for h36m_path in matched_files:
                base_filename = os.path.basename(h36m_path)
                label_filename = base_filename.replace("_h36m.npy", "_label.npy")

                # shape 为（Frames, Joints, 3）
                keypoints_3d = np.load(h36m_path)
                num_frames = keypoints_3d.shape[0]

                combined_constraint_signal = calculate_mzeni_prior(keypoints_3d, ankle_weight=0.5) # 构建最终的联合物理判据信号
                signal_1d = combined_constraint_signal.flatten()

                # 寻峰与生成标签
                peaks, properties = find_peaks(signal_1d, distance=15, prominence=0.05)

                # 生成对应的独热编码标签
                labels = np.zeros((num_frames, 1), dtype=np.float32)
                labels[peaks] = 1.0
                save_dir = os.path.join(output_root, floder)
                os.makedirs(save_dir, exist_ok=True)
                labels_save_path = os.path.join(save_dir, label_filename)
                np.save(labels_save_path, labels)

                print(f" 处理完毕：{floder} / {label_filename} | 提取到 M-Zeni 联合关键帧：{len(peaks)}个")

if __name__ == "__main__":
    ROOT_DIR = "D:/Python项目/KeyFrame/AthletePose3D/data"
    OUTPUT_DIR = r"D:\Python项目\KeyFrame\data\processed_labels"
    TARGET_SPLITS = {
        "train": ["train_set/S3", "train_set/S4"],
        "valid": ["valid_set/S2"],
        "test": ["test_set/S2"]
    }
    generate_labels_for_athlete_pose(ROOT_DIR, OUTPUT_DIR, TARGET_SPLITS)