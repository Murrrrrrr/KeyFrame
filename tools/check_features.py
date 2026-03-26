import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def check_feature_file(feature_path, output_dir, show_plot=False):
    if not os.path.exists(feature_path):
        print(f"找不到文件：{feature_path}")
        return False

    # 1. 加载数据
    features = np.load(feature_path)

    # 获取基础信息
    num_frames = features.shape[0]
    num_channels = features.shape[1] if len(features.shape) > 1 else 1

    folder_name = os.path.basename(os.path.dirname(feature_path))
    file_basename = os.path.basename(feature_path).replace(".npy", "")

    print(f"\n=== 特征文件检查报告: [{folder_name} / {file_basename}] ===")
    print(f"  - 序列总帧数 (T): {num_frames}")
    print(f"  - 特征维度数 (D): {num_channels}")

    # 2. 维度断言测试
    if num_channels != 43:
        print(f"  ❌ [致命错误] 期望维度是 43，但实际提取到了 {num_channels} 维！请检查 generate_features.py")
        return False
    else:
        print("  ✅ 维度检查通过 (21空间 + 21速度 + 1 M-Zeni = 67维)")

    # 3. 脏数据断言测试
    if np.isnan(features).any() or np.isinf(features).any():
        print("  ❌ [致命错误] 特征中发现 NaN 或 Inf 脏数据！模型训练会直接崩溃！")
        return False
    else:
        print("  ✅ 纯净度检查通过 (无 NaN / Inf)")

    # 4. 统计学分布测试 (验证物理归一化是否成功)
    # 因为使用 98% 分位数最大绝对值归一化，正常数值应主要分布在 [-1.5, 1.5] 之间
    global_max = np.max(features)
    global_min = np.min(features)
    global_mean = np.mean(features)

    print(f"  - 全局最大值: {global_max:.4f} (期望在 1.0 ~ 2.0 之间)")
    print(f"  - 全局最小值: {global_min:.4f} (期望在 -2.0 ~ -1.0 之间)")
    print(f"  - 全局平均值: {global_mean:.4f} (期望在 0 附近)")

    if global_max > 10.0 or global_min < -10.0:
        print(" ⚠️ [警告] 发现极端的游离值！可能是 3D 骨架估计出了严重的飞点，请看波形图确认。")

    # 5. 波形可视化 (抽查 3 个典型的物理通道)
    display_frames = min(300, num_frames)
    frames_x = np.arange(display_frames)

    plt.figure(figsize=(14, 8))

    # 5.1 画 M-Zeni 物理时钟 (最核心的信号)
    plt.subplot(3, 1, 1)
    plt.plot(frames_x, features[:display_frames, 42], color='green', linewidth=2, label='Ch:42 (M-Zeni Signal)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title(f"Feature Waveform Inspection - {file_basename}")
    plt.ylabel("Norm Dist")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 5.2 画 空间坐标特征
    plt.subplot(3, 1, 2)
    plt.plot(frames_x, features[:display_frames, 0], color='blue', alpha=0.8, label='Ch:0 (Spatial Pos)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel("Norm Pos")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 5.3 画 速度特征
    plt.subplot(3, 1, 3)
    plt.plot(frames_x, features[:display_frames, 21], color='red', alpha=0.8, label='Ch:21 (Velocity)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Frame Index")
    plt.ylabel("Norm Vel")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    save_filename = f"FeatureCheck_{folder_name}_{file_basename}.png"
    save_path = os.path.join(output_dir, save_filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')

    if show_plot:
        plt.show(block=True)
    else:
        plt.close()

    return True


if __name__ == "__main__":
    # 配置你的路径
    PROCESSED_DIR = "D:/Python项目/KeyFrame/data/processed_features"
    RESULT_DIR = r"D:\Python项目\KeyFrame\data\check_features_result"

    os.makedirs(RESULT_DIR, exist_ok=True)

    # 深度递归搜索
    search_pattern = os.path.join(PROCESSED_DIR, "**", "*_feature.npy")
    all_feature_files = glob.glob(search_pattern, recursive=True)

    if not all_feature_files:
        print(f"警告: 在 {PROCESSED_DIR} 及其子目录下没有找到任何 feature.npy 文件！请先运行 generate_features.py")
    else:
        print(f"共找到 {len(all_feature_files)} 个特征文件，开始硬核体检...\n")

        success_count = 0
        for idx, feature_file in enumerate(all_feature_files):
            # 为了防止弹窗太多，这里默认 show_plot=False，只保存图片
            # 你可以去 check_features_result 文件夹里查看生成的波形图
            is_ok = check_feature_file(feature_file, output_dir=RESULT_DIR, show_plot=False)
            if is_ok:
                success_count += 1

        print("\n" + "=" * 50)
        print(f" 体检完成！ {success_count} / {len(all_feature_files)} 个文件完全合格！")
        print(f" 详细的特征波形图已保存至: {RESULT_DIR}")
        print("=" * 50)