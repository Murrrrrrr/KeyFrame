import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import find_peaks

def check_label_file(label_path, output_dir, show_plot=False):
    if not os.path.exists(label_path):
        print(f"找不到文件：{label_path}")
        return

    labels = np.load(label_path)
    num_frames = labels.shape[0]
    num_channels = labels.shape[1] if len(labels.shape) > 1 else 1
    if num_channels != 5:
        print(f"警告：期望的标签通道数为 5，但实际为 {num_channels} ! 请检查数据生成脚本！")

    # 提取4个通道
    l_hs_idx, _ = find_peaks(labels[:, 0], height=0.5)
    l_to_idx, _ = find_peaks(labels[:, 1], height=0.5)
    r_hs_idx, _ = find_peaks(labels[:, 2], height=0.5)
    r_to_idx, _ = find_peaks(labels[:, 3], height=0.5)
    toe_max_idx, _ = find_peaks(labels[:, 4], height=0.5)

    # 获取上一级目录名，方便在报告中区分
    folder_name = os.path.basename(os.path.dirname(label_path))
    split_name = os.path.basename(os.path.dirname(os.path.dirname(label_path)))
    file_basename = os.path.basename(label_path).replace(".npy", "")

    print(f"=== 标签文件检查报告: [{folder_name}] ===")
    print(f"文件路径: {label_path}")
    print(f"总帧数: {num_frames}")
    print(f"提取出事件总数: {len(l_hs_idx) + len(l_to_idx) + len(r_hs_idx) + len(r_to_idx)}")
    print(f"  - L_HS (左脚触地): {len(l_hs_idx)} 次 | 前5次帧号: {l_hs_idx[:5]}")
    print(f"  - L_TO (左脚离地): {len(l_to_idx)} 次 | 前5次帧号: {l_to_idx[:5]}")
    print(f"  - R_HS (右脚触地): {len(r_hs_idx)} 次 | 前5次帧号: {r_hs_idx[:5]}")
    print(f"  - R_TO (右脚离地): {len(r_to_idx)} 次 | 前5次帧号: {r_to_idx[:5]}")
    print(f"  - Toe_Max (双足最大距离): {len(toe_max_idx)} 次 | 前5次帧号: {toe_max_idx[:5]}")
    if len(l_hs_idx) > 1 and len(r_hs_idx) > 1:
        l_stride = np.mean(np.diff(l_hs_idx))
        r_stride = np.mean(np.diff(r_hs_idx))
        print(f"步态周期 (Stride): 左脚平均 {l_stride:.1f} 帧，右脚平均 {r_stride:.1f} 帧")
    print("=========================================\n")

    display_frames = min(500, num_frames)
    plt.figure(figsize=(12, 5))
    def filter_display(indices):
        """只保留需要展示的帧范围内的索引"""
        return indices[indices < display_frames]

    frames_x = np.arange(display_frames)

    # 画 4 条参考虚线
    for y in [1, 2, 3, 4, 5]:
        plt.axhline(y=y, color='gray', linestyle='--', alpha=0.3)

    # 绘制高斯分布连续曲线
    plt.plot(frames_x, labels[:display_frames, 0] + 5, color='blue', alpha=0.7)
    plt.plot(frames_x, labels[:display_frames, 1] + 4, color='cyan', alpha=0.7)
    plt.plot(frames_x, labels[:display_frames, 2] + 3, color='red', alpha=0.7)
    plt.plot(frames_x, labels[:display_frames, 3] + 2, color='magenta', alpha=0.7)
    plt.plot(frames_x, labels[:display_frames, 4] + 1, color='green', alpha=0.7)  # <- 新增：第 5 条曲线

     # 在峰顶绘制散点
    plt.scatter(filter_display(l_hs_idx), labels[filter_display(l_hs_idx), 0] + 5, color='blue', marker='v', s=60,
                    label='L_HS')
    plt.scatter(filter_display(l_to_idx), labels[filter_display(l_to_idx), 1] + 4, color='cyan', marker='^', s=60,
                    label='L_TO')
    plt.scatter(filter_display(r_hs_idx), labels[filter_display(r_hs_idx), 2] + 3, color='red', marker='v', s=60,
                    label='R_HS')
    plt.scatter(filter_display(r_to_idx), labels[filter_display(r_to_idx), 3] + 2, color='magenta', marker='^',
                    s=60, label='R_TO')
    plt.scatter(filter_display(toe_max_idx), labels[filter_display(toe_max_idx), 4] + 1, color='green', marker='*',
                    s=80, label='Toe_Max')

    plt.title(f"5-Channel Gaussian Soft Labels (First {display_frames} frames) - Dataset: {folder_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Event Probability (Stacked)")
    plt.yticks([1, 2, 3, 4, 5], ['Toe Max','Right TO', 'Right HS', 'Left TO', 'Left HS'])
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, alpha=0.2, axis='x')
    plt.tight_layout()
    save_filename = f"{split_name}_{folder_name}_{file_basename}.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"走势图已保存至：: {save_path}")
    if show_plot:
        plt.show(block=True)
    else:
        plt.close()

if __name__ == "__main__":
    PROCESSED_DIR = "D:/Python项目/KeyFrame/data/processed_labels"
    RESULT_DIR = r"D:\Python项目\KeyFrame\data\check_labels_result"

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 深度递归搜索
    search_pattern = os.path.join(PROCESSED_DIR, "**", "*_label.npy")
    all_label_files = glob.glob(search_pattern, recursive=True)

    if not all_label_files:
        print(f"警告: 在 {PROCESSED_DIR} 及其子目录下没有找到任何 label.npy 文件！")
    else:
        print(f"共找到 {len(all_label_files)} 个标签文件，开始批量体检...\n")

        for idx, label_file in enumerate(all_label_files):
            print(f"正在分析第 {idx + 1}/{len(all_label_files)} 个文件...")
            check_label_file(label_file, output_dir=RESULT_DIR, show_plot=False)

        print("所有标签文件检查完毕！数据准备就绪！")