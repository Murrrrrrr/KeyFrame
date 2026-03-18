import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def check_label_file(label_path, output_dir, show_plot=False):
    if not os.path.exists(label_path):
        print(f"找不到文件：{label_path}")
        return

    labels = np.load(label_path)
    num_frames = labels.shape[0]

    # 找到所有标签为 1（关键帧）的帧索引
    keyframe_indices = np.where(labels == 1.0)[0]

    # 获取上一级目录名，方便在报告中区分
    folder_name = os.path.basename(os.path.dirname(label_path))
    split_name = os.path.basename(os.path.dirname(os.path.dirname(label_path)))
    file_basename = os.path.basename(label_path).replace(".npy", "")

    print(f"=== 标签文件检查报告: [{folder_name}] ===")
    print(f"文件路径: {label_path}")
    print(f"总帧数: {num_frames}")
    print(f"提取出的关键帧总数: {len(keyframe_indices)}")
    if len(keyframe_indices) > 0:
        print(f"前 20 个关键帧的帧号: {keyframe_indices[:20]}")

    # 计算关键帧之间的平均间隔
    if len(keyframe_indices) > 1:
        intervals = np.diff(keyframe_indices)
        print(f"关键帧平均间隔: {np.mean(intervals):.2f} 帧 (在 120fps 下约 {np.mean(intervals) / 120:.2f} 秒/步)")
    print("=========================================\n")

    display_frames = min(500, num_frames)
    plt.figure(figsize=(12, 4))
    plt.plot(labels[:display_frames], color='green', label='Ground Truth (1=Keyframe)')
    plt.title(f"Label Visualization (First {display_frames} frames) - Dataset: {folder_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Label Value")
    plt.yticks([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3)
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