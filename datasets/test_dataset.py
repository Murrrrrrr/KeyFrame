import os
import time
import torch
from torch.utils.data import DataLoader
import sys

# 确保能正确导入上层目录的 datasets 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.pose_dataset import PoseSequenceDataset


def test_data_pipeline():
    print("=" * 50)
    print("🚀 开始执行数据加载管道压力测试")
    print("=" * 50)

    # 1. 模拟你的项目根目录配置
    # 请根据实际情况修改这些路径
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_features")
    LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")
    SPLIT_FILE = os.path.join(PROJECT_ROOT, "data","splits","athlete_pose_splits.json")

    # 硬件与 DataLoader 优化参数
    BATCH_SIZE = 32
    # 建议设置为 CPU 核心数的 1/2 到 1/4。用于绕过 Python GIL，实现多进程异步读取
    NUM_WORKERS = 4
    # 将内存页锁定（Page-locked memory），允许 DMA 控制器直接将数据通过 PCIe 拷贝到 GPU，极大提升传输速率
    PIN_MEMORY = torch.cuda.is_available()

    splits_to_test = ['train', 'valid', 'test']

    for mode in splits_to_test:
        print(f"\n[{mode.upper()} 模式验证]")
        try:
            # 实例化 Dataset
            dataset = PoseSequenceDataset(
                data_dir=DATA_DIR,
                label_dir=LABEL_DIR,
                split_file=SPLIT_FILE,
                split_mode=mode,
                seq_len=64,
                stride=16,
                base_fps=50,
                simulate_jitter=(mode == 'train')  # 仅在训练时模拟边缘端时钟抖动以增加鲁棒性
            )

            if len(dataset) == 0:
                print(f"⚠️ 警告: {mode} 数据集为空，请检查目录结构或 JSON 配置。")
                continue

            # 实例化 DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=(mode == 'train'),  # 仅训练集打乱，打乱会增加内存随机寻址开销
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=(mode == 'train')  # 训练时丢弃不足 batch 的尾部数据，保证矩阵维度一致
            )

            # 抽取一个 Batch 进行维度验证
            start_time = time.time()
            for batch_idx, (x, dt, y) in enumerate(dataloader):
                load_time = time.time() - start_time

                print(f"✅ 成功加载首个 Batch! 耗时: {load_time:.4f} 秒")
                print(f"   -> 特征张量 X shape:  {x.shape} (预期: [Batch, SeqLen, Features])")
                print(f"   -> 时间张量 dt shape: {dt.shape} (预期: [Batch, SeqLen, 1])")
                print(f"   -> 标签张量 Y shape:  {y.shape} (预期: [Batch, SeqLen, Classes])")

                # 数学维度严格校验
                assert x.dim() == 3, "特征矩阵必须是 3 阶张量"
                assert dt.shape[2] == 1, "时间间隔必须是标量特征"

                # 打印单次 DMA 传输预估内存大小 (MB)
                # 计算公式: (元素总数 * 4字节(float32)) / (1024 * 1024)
                bytes_per_batch = (x.numel() + dt.numel() + y.numel()) * 4
                print(f"   -> 单个 Batch 内存占用: {bytes_per_batch / (1024 * 1024):.2f} MB")
                break  # 仅测试第一个 batch

        except Exception as e:
            print(f"❌ 初始化 {mode} 数据集失败: {str(e)}")


if __name__ == "__main__":
    test_data_pipeline()