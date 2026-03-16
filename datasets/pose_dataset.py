import os
import numpy as np
import torch
import json
from torch.utils.data import Dataset

class PoseSequenceDataset(Dataset):
    """
    设计专门对于 LNN 的时序姿态数据集
    支持长视频滑窗切割、多模态特征融合以及时间间隔特征的提取
    """
    def __init__(self,data_dir, label_dir, split_file, split_mode='train', seq_len=64, stride=16, base_fps=50, simulate_jitter=False):
        """
        :param data_dir: 处理后的67维特征数据 （.npy）
        :param label_dir: 处理后的 one-hot 标签目录
        :param seq_len: 滑动窗口的时序长度（默认为64帧）
        :param stride:  滑动步长，计算 Overlap (seq_len - stride)
        :param base_fps: 原始采集帧率，用于计算基准 dt
        :param simulate_jitter: 硬件仿真标志位。设为 True 时模拟边缘设备的丢帧和时钟抖动
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.split_file = split_file
        self.split_mode = split_mode
        self.seq_len = seq_len
        self.stride = stride
        self.base_dt = 1.0/base_fps
        self.simulate_jitter = simulate_jitter

        self.samples = []
        self._build_index()

    def _build_index(self):
        """解析 JSON 并定向扫描指定目录构建滑窗索引"""
        # 挂载并解析JSON划分配置文件
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"警告： 找不到数据划分文件")

        with open(self.split_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)

        if self.split_mode not in split_info['splits']:
            raise ValueError(f"[配置错误] 未知的划分模式'{self.split_mode}'。请检查 JSON 文件。")

        allowed_subdirs = split_info['splits'][self.split_mode]

        # 仅在允许的受试者/环境子目录中进行检索
        for subdir in allowed_subdirs:
            target_data_dir = os.path.join(self.data_dir, subdir)

            if not os.path.exists(target_data_dir):
                print(f"[IO警告] 配置中声明的数据子目录不存在，跳过：{target_data_dir}")
                continue

            for root, _, files in os.walk(target_data_dir):
                for file in files:
                    if not file.endswith('.npy'):
                        continue

                    data_path = os.path.join(root, file)
                    # 镜像映射到标签路径，保持相对目录树一致
                    rel_dir = os.path.relpath(root, self.data_dir)
                    label_path = os.path.join(self.label_dir, rel_dir, 'labels.npy')

                    if not os.path.exists(label_path):
                        print(f"[数据缺失警告] 找不到对应的标签文件，跳过：{label_path}")
                        continue

                    # 内存映射读取 shape，避免资源爆炸
                    data_shape = np.load(data_path, mmap_mode = 'r').shape
                    total_frames = data_shape[0]

                    if total_frames < self.seq_len:
                        continue

                    for start_idx in range(0, total_frames-self.seq_len + 1, self.stride):
                        self.samples.append({
                            'data_path': data_path,
                            'label_path': label_path,
                            'start_idx': start_idx,
                        })

            print(f"[数据集准备就绪] 模式：{self.split_mode.upper()} | 共构建{len(self.samples)} 个序列窗口。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        start = sample_info['start_idx']
        end = start + self.seq_len

        # 提取多模态特征矩阵
        features = np.load(sample_info['data_path'], mmap_mode='r')[start:end].copy()
        labels = np.load(sample_info['label_path'], mmap_mode='r')[start:end].copy()

        # 硬件时间戳抽象：生成 dt 数组
        dt_array = np.full((self.seq_len, 1), self.base_dt, dtype=np.float32)

        # (软硬协同模块) 开启时磨你边缘端 I2C/MIPI 接口读取传感器时的时钟抖动和总线阻塞丢帧
        if self.simulate_jitter:
            jitter = np.random.normal(0, self.base_dt * 0.2, (self.seq_len, 1))
            dt_array += np.clip(dt_array + jitter, a_min=1e-4, a_max=None)
            drop_mask = np.random.rand(self.seq_len, 1) < 0.05
            dt_array[drop_mask] *= np.random.randint(2, 4)

        # 转换为PyTorch Tensors
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32)
        dt_tensor = torch.tensor(dt_array, dtype=torch.float32)

        return x_tensor, y_tensor, dt_tensor


# ==========================================
# 测试脚本：直接运行 python datasets/pose_dataset.py 即可触发
# ==========================================
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import os

    print("=== 开始测试 PoseSequenceDataset 软硬协同数据管道 ===")

    # 【动态寻址魔法】获取当前脚本的绝对路径，并反推项目根目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))  # datasets 目录
    project_root = os.path.dirname(current_script_dir)  # KeyFrame 根目录

    # 动态拼接出绝对路径，彻底消灭路径找不到的问题！
    TEST_DATA_DIR = os.path.join(project_root, "data","processed_features")
    TEST_LABEL_DIR = os.path.join(project_root, "data", "processed_labels")
    TEST_SPLIT_FILE = os.path.join(project_root, "data","splits", "athlete_pose_splits.json")

    print(f"🔍 正在检查配置文件路径: {TEST_SPLIT_FILE}")
    if not os.path.exists(TEST_SPLIT_FILE):
        print("❌ 致命错误：划分文件确实不存在！请检查 configs 文件夹下是否有 athlete_pose_splits.json")
        exit(1)

    try:
        # 1. 实例化 Dataset (开启硬件抖动仿真)
        dataset = PoseSequenceDataset(
            data_dir=TEST_DATA_DIR,
            label_dir=TEST_LABEL_DIR,
            split_file=TEST_SPLIT_FILE,
            split_mode='train',
            seq_len=64,
            stride=16,
            base_fps=120,
            simulate_jitter=True
        )

        print(f"\n✅ 数据集实例化成功！共提取了 {len(dataset)} 个时序滑窗。")

        # 2. 包装进 DataLoader
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        # 3. 抽取第一个 Batch 进行维度和数据校验
        for batch_idx, (x, y, dt) in enumerate(dataloader):
            print("\n=== 抽取第一个 Batch 数据分析 ===")
            print(f"📦 特征 X shape:  {x.shape}  -> 期望为 [4, 64, 67] (Batch, Seq_len, Features)")
            print(f"🎯 标签 Y shape:  {y.shape}  -> 期望为 [4, 64, 1]  (Batch, Seq_len, Label)")
            print(f"⏱️ 时间 dt shape: {dt.shape} -> 期望为 [4, 64, 1]  (Batch, Seq_len, dt)")

            # 检查软硬协同的 dt 抖动情况
            print("\n=== 检查硬件时钟抖动仿真 (simulate_jitter) ===")
            dt_numpy = dt.numpy()
            print(f"理论基准 dt: {1.0 / 120:.5f} 秒")
            print(f"实际生成的 dt 均值: {dt_numpy.mean():.5f} 秒")
            print(f"实际生成的 dt 最小: {dt_numpy.min():.5f} 秒")
            print(f"实际生成的 dt 最大: {dt_numpy.max():.5f} 秒")

            # 检查 NaN 硬件报警机制
            if torch.isnan(x).any():
                print("⚠️ 警告：特征数据中包含 NaN 硬件报警信号！")
            else:
                print("✅ 特征数据干净，无 NaN。")

            break  # 测完第一个 Batch 就退出

    except Exception as e:
        print(f"\n❌ 测试失败。错误信息：\n{e}")