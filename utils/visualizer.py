import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from models.struct_lnn import StructLNN
from datasets.pose_dataset import PoseSequenceDataset

# 配置 Matplotlib 字体，防止中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description='跑姿关键帧可视化工具')
    parser.add_argument("--config", type=str, default="configs/struct_lnn_mzeni.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Running_Gait_Analsy_mzeni_false_best.pth", help="训练好的模型权重路径")
    parser.add_argument("--sample_idx", type=int, default=15, help="在验证集中抽取第几个样本进行可视化")
    parser.add_argument("--save_path", type=str, default="result_visualization.png", help="图表保存路径")
    return parser.parse_args()

def main():
    args = parse_args()

    # 加载配置和环境
    print(f" $ 正在加载配置文件：{args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载最好权重
    print(f" $ 正在挂载 LNN 模型权重: {args.checkpoint}")
    model = StructLNN(config).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件 {args.checkpoint}，请确认路径是否确认。")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载验证数据集
    print(f" $ 正在加载验证集数据...")
    val_dataset = PoseSequenceDataset(config, split="valid")

    if args.sample_idx >= len(val_dataset):
        print(f"[警告] sample_idx ({args.sample_idx}) 超出数据集长度，自动重置为 0 ")
        args.sample_idx = 0

    # 获取单个样本：x_data=(features, dt), y_data=labels
    (x_tensor, dt_tensor), y_tensor = val_dataset[args.sample_idx]

    # 增加 Batch 维度并送入设备 [1, SeqLen, Features]
    x_input = x_tensor.unsqueeze(0).to(device)
    dt_input = dt_tensor.unsqueeze(0).to(device)
    y_target = y_tensor.numpy().flatten()

    # 模型推理
    print(" $ 正在进行连续时间推演...")
    with torch.no_grad():
        logits = model((x_input, dt_input))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy() # [SeqLen] 转换为 0~1 的概率

    # 提取特征进行可视化
    # 在67维特征中 ，索引66是 M-Zeni 物理特征
    mzeni_feature = x_tensor[:, 66].numpy()

    # 归一化 M-Zeni 以便在同一张图上展示（将其缩放到了 0~1 之间）
    mzeni_norm = (mzeni_feature - np.min(mzeni_feature)) / (np.max(mzeni_feature) - np.min(mzeni_feature) + 1e-6)

    # 使用寻峰算法
    peaks, _ = find_peaks(probs, height = 0.3, distance = 5)

    print(f" $ 正在生成可视化图标...")
    frames = np.arange(len(probs))

    fig, ax = plt.subplots(figsize=(14,6), dpi=300)

    # 绘制底色网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 曲线1：M-Zeni 物理先验信号 (绿色虚线)
    ax.plot(frames, mzeni_norm, label="M-Zeni 物理约束信号 (Normalized)", color="#2ca02c", linestyle="-.", linewidth=2,
            alpha=0.7)

    # 曲线2：真实标签 Ground Truth (红色突起)
    ax.plot(frames, y_target, label="真实触地标签 (Ground Truth)", color="#d62728", linestyle="--", linewidth=2.5,
            alpha=0.8)

    # 曲线3：模型预测的连续概率 (蓝色主曲线)
    ax.plot(frames, probs, label="Struct-LNN 预测概率", color="#1f77b4", linewidth=3)

    # 散点：模型预测出的关键帧波峰 (黑色 X 标记)
    if len(peaks) > 0:
        ax.plot(frames[peaks], probs[peaks], "x", color="black", markersize=12, markeredgewidth=3,
                label="预测触地点 (Detected Peaks)")

    # 装饰图表
    ax.set_title("跑姿触地关键帧提取结果可视化 (基于 M-Zeni 与 LNN 连续时间建模)", fontsize=16, fontweight='bold')
    ax.set_xlabel("视频帧序号 (Frames)", fontsize=14)
    ax.set_ylabel("激活概率 / 归一化特征幅值", fontsize=14)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(0, len(probs))

    # 优化图例位置
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    # 紧凑布局并保存
    plt.tight_layout()
    save_dir = "result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    final_save_path = os.path.join(save_dir, f"sample_{args.sample_idx}_visualization.png")
    plt.savefig(final_save_path)
    print(f"[成功] 可视化图表已保存至: {args.save_path}")

    # 如果你在带有 GUI 的环境下运行，可以直接显示：
    # plt.show()

if __name__ == "__main__":
    main()