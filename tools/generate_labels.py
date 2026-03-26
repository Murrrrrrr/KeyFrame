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
    plt.plot(r_hs, r_heel_dist[r_hs], "x", color='magenta', markersize=12, markeredgewidth=3,label='Right HS (Detected)')

    plt.title(f"Zeni Projection Signal - {video_name}")
    plt.xlabel("Frames")
    plt.ylabel("Distance (Front/Back relative to Pelvis)")
    plt.legend()
    plt.grid(True)

    print(">>> 正在显示 Debug 波形图。请查看弹出的窗口。关闭窗口后程序将继续运行...")
    plt.show()

def apply_gaussian_label(labels, channel_idx, event_frames, num_frames, sigma=2.0):
    """
    在指定的通道上，以各个事件帧 (t0) 为中心，生成 1D 高斯概率分布
    :param labels:  shape [T, 4] 的标签矩阵
    :param channel_idx: 当前写入的通道 (0~3)
    :param event_frames: 检测到的峰值帧索引数组
    :param num_frames: 序列总帧数 T
    :param sigma: 高斯分布的标准差，控制标签的“宽度”
    """
    for t0 in event_frames:
        # 高斯分布在 3 * sigma 之外的值非常小，所以只截取这个窗口
        radius = int(np.ceil(3 * sigma))
        start = max(0,t0 - radius)
        end = min(num_frames, t0 + radius + 1)

        # 计算窗口内的时间步
        t = np.arange(start, end)

        # 核心公式：y_t = exp(-(t - t0)^2 / (2 * sigma^2))
        gaussian = np.exp(-((t-t0) ** 2) / (2 * sigma ** 2))

        # 使用了 np.maximum 而不是简单的加减法
        # 这样即使两个步态事件离得异常近，最高的概率也依然限制在 1.0 以内
        labels[start:end, channel_idx] = np.maximum(labels[start:end, channel_idx], gaussian  )

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
                # 对应关系：0（Left_HS）,1（Left_TO）,2（Right_HS）,3（Right_TO） ,4(Toe_Max)
                labels = np.zeros((num_frames, 5), dtype=np.float32)

                # 安全提取索引并过滤掉超出帧数范围的无效索引
                def get_valid_frames(frames_array):
                    return frames_array[frames_array < num_frames]

                valid_l_hs = get_valid_frames(keyframes_dict["Left_HS"])
                valid_l_to = get_valid_frames(keyframes_dict["Left_TO"])
                valid_r_hs = get_valid_frames(keyframes_dict["Right_HS"])
                valid_r_to = get_valid_frames(keyframes_dict["Right_TO"])
                valid_toe_max = get_valid_frames(keyframes_dict["Toe_Max"])

                # sigma 值设定建议：
                # 如果视频是 120 FPS，建议 sigma=3.0 ~ 4.0 之间 （覆盖前后大概10帧）
                # 如果你的视频是 50~60 FPS，建议 sigma=1.5 ~ 2.0 (覆盖大概前后3~6帧)
                # 如果视频是 30 FPS，建议 sigma=0.5 ~ 1.5 (覆盖前后大概1~3帧)
                SIGMA_VAL = 3.5

                apply_gaussian_label(labels, 0, valid_l_hs, num_frames, sigma=SIGMA_VAL)
                apply_gaussian_label(labels, 1, valid_l_to, num_frames, sigma=SIGMA_VAL)
                apply_gaussian_label(labels, 2, valid_r_hs, num_frames, sigma=SIGMA_VAL)
                apply_gaussian_label(labels, 3, valid_r_to, num_frames, sigma=SIGMA_VAL)
                apply_gaussian_label(labels, 4, valid_toe_max, num_frames, sigma=SIGMA_VAL)

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