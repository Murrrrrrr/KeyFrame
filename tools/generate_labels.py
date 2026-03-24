import os
import numpy as np
import glob
from scipy.signal import find_peaks
from utils.physics_utils import extract_m_zeni
import matplotlib.pyplot as plt

def debug_plot_zeni_signal(l_heel_dist, l_hs, r_heel_dist, r_hs, video_name):
    """
    终极波形可视化工具：看看算法眼里的人是怎么跑的
    """
    plt.figure(figsize=(14, 6))
    plt.plot(l_heel_dist, label='Left Heel Forward Dist', color='blue', alpha=0.7)
    plt.plot(r_heel_dist, label='Right Heel Forward Dist', color='red', alpha=0.7)

    # 标记最终确定的触地点
    plt.plot(l_hs, l_heel_dist[l_hs], "x", color='cyan', markersize=12, markeredgewidth=3, label='Left HS (Detected)')
    plt.plot(r_hs, r_heel_dist[r_hs], "x", color='magenta', markersize=12, markeredgewidth=3,
             label='Right HS (Detected)')

    plt.title(f"Zeni Projection Signal - {video_name}")
    plt.xlabel("Frames")
    plt.ylabel("Distance (Front/Back relative to Pelvis)")
    plt.legend()
    plt.grid(True)

    print(">>> 正在显示 Debug 波形图。请查看弹出的窗口。关闭窗口后程序将继续运行...")
    plt.show()

def generate_labels_for_athlete_pose(dataset_root, output_root, splits):
    """
    根据数据集自带的 3D骨架数据(h36m.npy)自动生成跑姿关键帧标签
    物理依据：M-Zeni规则
    """
    print("开始自动生成跑姿关键帧标签...")
    has_plotted = False

    for split_name, folders in splits.items():
        for folder in folders:
            folder_path = os.path.join(dataset_root, folder)
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
                    min_frames_between_steps=35,
                    prominence=0.05,
                    cross_limb_weight=0.5,
                    tolerance = 18
                )

                # 画出特征波形图
                if not has_plotted:
                    debug_plot_zeni_signal(
                        l_heel_dist=keyframes_dict["raw_l_heel_dist"],
                        l_hs=keyframes_dict["Left_HS"],
                        r_heel_dist=keyframes_dict["raw_r_heel_dist"],
                        r_hs=keyframes_dict["Right_HS"],
                        video_name=base_filename
                    )
                    has_plotted = True  # 画过一次就锁定，不再画了

                # 初始化四通道矩阵
                # 对应关系：0（Left_HS）,1（Left_TO）,2（Right_HS）,3（Right_TO）
                labels = np.zeros((num_frames, 4), dtype=np.float32)

                # 安全提取索引并过滤掉超出帧数范围的无效索引
                def get_valid_frames(frames_array):
                    return frames_array[frames_array < num_frames]

                valid_l_hs = get_valid_frames(keyframes_dict["Left_HS"])
                valid_l_to = get_valid_frames(keyframes_dict["Left_TO"])
                valid_r_hs = get_valid_frames(keyframes_dict["Right_HS"])
                valid_r_to = get_valid_frames(keyframes_dict["Right_TO"])

                # 赋值 One-hot 标签
                if len(valid_l_hs) > 0: labels[valid_l_hs, 0] = 1.0
                if len(valid_l_to) > 0: labels[valid_l_to, 1] = 1.0
                if len(valid_r_hs) > 0: labels[valid_r_hs, 2] = 1.0
                if len(valid_r_to) > 0: labels[valid_r_to, 3] = 1.0

                total_events = len(valid_l_hs) + len(valid_l_to) + len(valid_r_hs) + len(valid_r_to)

                save_dir = os.path.join(output_root, folder)
                os.makedirs(save_dir, exist_ok=True)
                labels_save_path = os.path.join(save_dir, label_filename)
                np.save(labels_save_path, labels)

                print(f" 处理完毕：{folder} / {label_filename} | 提取到 M-Zeni 联合关键帧：{total_events}个")
                # 打印一下分布，看看左右脚是否平衡
                print(f" -> L_HS:{len(valid_l_hs)}, L_TO:{len(valid_l_to)}, R_HS:{len(valid_r_hs)}, R_TO:{len(valid_r_to)}")

if __name__ == "__main__":
    ROOT_DIR = "D:/Python项目/KeyFrame/AthletePose3D/data"
    OUTPUT_DIR = r"D:\Python项目\KeyFrame\data\processed_labels"
    TARGET_SPLITS = {
        "train": ["train_set/S3", "train_set/S4"],
        "valid": ["valid_set/S2"],
        "test": ["test_set/S2"]
    }
    generate_labels_for_athlete_pose(ROOT_DIR, OUTPUT_DIR, TARGET_SPLITS)