import os
import numpy as np
import glob

def extract_61d_features(raw_file_path):
    """
    填入软硬件协同特征提取逻辑：单目空间坐标 + M-Zeni规则 + 速度特征
    """
    # 数据提取逻辑
    # 1.读取原始数据（*.csv，*.json，或是通过 OpenCV 读取 .mp4 喂给 MediaPipe）
    # 2.获取视频的总帧数 total_frames

    # ---------------- 模拟数据提取 (请替换为你自己的真实逻辑) ----------------
    # 假设我们从某个 raw_file 中解析出这组动作有 1200 帧
    # 这里用随机数模拟，真实情况你应该基于物理规则计算
    total_frames = 1200

    # 2. 构建 61 维特征:
    # 例如：
    # dim 0-32: 空间坐标 (x, y, z 等)
    # dim 33-45: 速度、加速度 (差分求导)
    # dim 46-60: M-Zeni 物理约束特征 (例如关节角度、重心偏移等)

    # 生成 [T, 61] 维度的 float32 矩阵
    # 硬件提示：边缘设备通常使用 FP32 甚至 FP16 推理，这里务必强制设为 np.float32 避免双精度浪费内存
    features = np.random.randn(total_frames, 61).astype(np.float32)

    # -------------------------------------------------------------------------

    # 【工程安全校验】特征绝对不能包含 NaN（计算除以0时极易产生）
    if np.isnan(features).any():
        print(f"⚠️ [硬件报警] 提取的特征中包含 NaN (文件: {raw_file_path})，已自动补0。")
        features = np.nan_to_num(features)

    return features


def main():
    print("=" * 50)
    print("⚙️  开始执行 61维 特征提取管道 (Feature Extraction)")
    print("=" * 50)

    # 1. 配置路径
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 假设你的原始数据按受试者存放在 raw_data 里
    # 比如: data/raw_data/train_set/S3/run_01.csv
    RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw_data")

    # 目标输出路径 (必须与之前 test_dataset.py 里的 EXPECTED 路径一致！)
    OUTPUT_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "processed_features")

    # 如果原始数据目录不存在，给出工程提示
    if not os.path.exists(RAW_DATA_DIR):
        print(f"⚠️ 找不到原始数据目录: {RAW_DATA_DIR}")
        print("请创建一个假目录进行测试，或者修改为你的真实数据路径。")
        # 我们可以先建几个假目录来跑通管道
        os.makedirs(os.path.join(RAW_DATA_DIR, "train_set", "S3"), exist_ok=True)
        os.makedirs(os.path.join(RAW_DATA_DIR, "train_set", "S4"), exist_ok=True)
        # 生成一个假文件触发流程
        with open(os.path.join(RAW_DATA_DIR, "train_set", "S3", "dummy_run.csv"), 'w') as f:
            f.write("dummy")

    # 2. 遍历并处理所有原始数据
    # 这里我们遍历 raw_data 下所有的受试者文件夹
    search_pattern = os.path.join(RAW_DATA_DIR, "**", "*.*")  # 适配 .mp4, .csv 等
    raw_files = glob.glob(search_pattern, recursive=True)

    valid_files = [f for f in raw_files if os.path.isfile(f) and not f.endswith('.npy')]

    if not valid_files:
        print("没有找到需要处理的原始数据文件。")
        return

    processed_count = 0
    for raw_file in valid_files:
        # 计算相对路径，以便在 processed_features 下建立完全一样的镜像目录结构
        rel_path = os.path.relpath(os.path.dirname(raw_file), RAW_DATA_DIR)
        target_dir = os.path.join(OUTPUT_FEATURE_DIR, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        # 提取 61 维特征矩阵
        feature_matrix = extract_61d_features(raw_file)

        # 保存为统一的 features.npy
        # 注意：如果你每个受试者有多个动作序列，最好用原始文件名命名，如 run_01_features.npy
        # 简单起见，如果按你之前的报错信息，这里直接存为 features.npy (覆盖式，需确保每个目录下只有一个序列)
        out_filename = "features.npy"
        out_path = os.path.join(target_dir, out_filename)

        # 将矩阵以二进制高密度存入磁盘
        np.save(out_path, feature_matrix)
        print(f"✅ 已生成特征: {out_path} | Shape: {feature_matrix.shape}")
        processed_count += 1

    print("=" * 50)
    print(f"🎉 特征工程执行完毕！共生成 {processed_count} 个 features.npy 文件。")
    print("现在你可以再次运行 datasets/test_dataset.py 测试数据喂养管道了！")


if __name__ == "__main__":
    main()