import os
import numpy as np
import glob
from scipy.signal import find_peaks
from utils.physics_utils import extract_m_zeni

def generate_labels_for_athlete_pose(dataset_root, output_root, splits):
    """
    根据数据集自带的 3D骨架数据(h36m.npy)自动生成跑姿关键帧标签
    物理依据：M-Zeni规则
    """
    print("开始自动生成跑姿关键帧标签...")

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

                keyframes_dict = extract_m_zeni(
                    keypoints_3d,
                    min_frames_between_steps=15,
                    prominence=0.05,
                    cross_limb_weight=0.5,
                    tolerance = 5
                )

                all_peaks = np.concatenate([
                    keyframes_dict["Left_Extreme"],
                    keyframes_dict["Right_Extreme"],
                ]).astype(int)

                # 防止同一帧同时被判定为触地和离地
                all_peaks = np.sort(np.unique(all_peaks))
                all_peaks = all_peaks[all_peaks < num_frames]

                labels = np.zeros((num_frames, 1), dtype=np.float32)
                if len(all_peaks) > 0:
                    labels[all_peaks] = 1.0

                save_dir = os.path.join(output_root, floder)
                os.makedirs(save_dir, exist_ok=True)
                labels_save_path = os.path.join(save_dir, label_filename)
                np.save(labels_save_path, labels)

                print(f" 处理完毕：{floder} / {label_filename} | 提取到 M-Zeni 联合关键帧：{len(all_peaks)}个")

if __name__ == "__main__":
    ROOT_DIR = "D:/Python项目/KeyFrame/AthletePose3D/data"
    OUTPUT_DIR = r"D:\Python项目\KeyFrame\data\processed_labels"
    TARGET_SPLITS = {
        "train": ["train_set/S3", "train_set/S4"],
        "valid": ["valid_set/S2"],
        "test": ["test_set/S2"]
    }
    generate_labels_for_athlete_pose(ROOT_DIR, OUTPUT_DIR, TARGET_SPLITS)