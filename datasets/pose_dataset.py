""" 将 feature.npy 文件打包成 PyTorch 的张量tensor"""
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
    def __init__(self, config, split='train'):
        self.config = config
        self.split_mode = split

        # 从config 中安全解析参数
        dataset_cfg = config.get('dataset', {})
        if not dataset_cfg:  # 兼容处理
            dataset_cfg = config.get('data', {})
        self.data_dir = dataset_cfg.get('data_dir', 'data/processed_features')
        self.label_dir = dataset_cfg.get('label_dir', 'data/processed_labels')
        self.split_file = dataset_cfg.get('split_file', 'data/splits/athlete_pose_splits.json')

        self.seq_len = dataset_cfg.get('seq_len', 64)
        self.stride = dataset_cfg.get('stride', 16)
        fps = dataset_cfg.get('fps', 120)
        self.base_dt = 1.0 / fps

        self.simulate_jitter = dataset_cfg.get('simulate_jitter', False)

        self.jitter_std = dataset_cfg.get('jitter_std', 0.2)
        self.drop_rate = dataset_cfg.get('drop_rate', 0.05)
        self.extract_m_zeni = dataset_cfg.get('extract_m_zeni', True)
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
                    label_file_name = file.replace('feature.npy', 'label.npy')
                    label_path = os.path.join(self.label_dir, rel_dir, label_file_name)

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

    def _get_mmap(self, file_path):
        if file_path not in self.mmap_cache:
            self.mmap_cache[file_path] = np.load(file_path, mmap_mode='r')
        return self.mmap_cache[file_path]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        start = sample_info['start_idx']
        end = start + self.seq_len

        # 提取多模态特征矩阵
        feature_mmap = np.load(sample_info['data_path'], mmap_mode='r')
        features = feature_mmap[start:end].copy()
        if not self.extract_m_zeni:
            features = features[:, :-1]

        # 真实物理事件（标签）在客观世界中不受传感器卡顿影响，依然保持正常时序
        label_mmap = np.load(sample_info['label_path'], mmap_mode='r')
        labels = label_mmap[start:end].copy()

        # 硬件时间戳抽象：生成 dt 数组
        dt_array = np.full((self.seq_len, 1), self.base_dt, dtype=np.float32)

        # 开启时模拟边缘端 I2C/MIPI 接口读取传感器时的时钟抖动和总线阻塞丢帧
        if self.simulate_jitter:
            # 模拟底层晶振微秒级时钟抖动
            jitter = np.random.normal(0, self.base_dt * self.jitter_std, (self.seq_len, 1))
            dt_array = np.clip(dt_array + jitter, 0, 1)
            # ⭐ 2. 模拟真实的突发性总线阻塞 (Burst Packet Loss)
            drop_mask = np.zeros(self.seq_len, dtype=bool)

            i = 1  # 保证第 0 帧永远有效，作为初始参考
            while i < self.seq_len:
                # 掷骰子，判断当前时刻总线是否发生故障
                if np.random.rand() < self.drop_rate:
                    # 一旦故障，随机连续卡死 1 到 10 帧 (可根据硬件恶劣程度在参数里配置)
                    burst_length = np.random.randint(1, 11)

                    # 计算故障结束的帧索引，注意不要越界
                    end_idx = min(i + burst_length, self.seq_len)

                    # 将这一段连续区间全部标记为丢帧
                    drop_mask[i:end_idx] = True

                    # 故障期间不再重复掷骰子，直接跳过这段黑暗期
                    i = end_idx
                else:
                    i += 1  # 设备正常，看下一帧

            # 3. 执行物理层面的“零阶保持 (Zero-Order Hold)”
            for i in range(1, self.seq_len):
                if drop_mask[i]:
                    # 硬件卡顿期间，内存缓冲区只能吐出上一帧的陈旧残留数据
                    features[i] = features[i - 1]

        # 转换为PyTorch Tensors
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32)
        dt_tensor = torch.tensor(dt_array, dtype=torch.float32)

        return (x_tensor, dt_tensor), y_tensor