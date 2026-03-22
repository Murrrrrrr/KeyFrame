# 文件路径: utils/visualizer_mlp.py
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 【关键修改 1】
from models.mlp_baseline import MLPBaseline
from datasets.pose_dataset import PoseSequenceDataset

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    parser = argparse.ArgumentParser(description='跑姿关键帧可视化工具 (MLP 版)')
    parser.add_argument("--config", type=str, default="configs/mlp_baseline.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/MLP_Baseline_Ablation_best.pth", help="训练好的模型权重路径")
    parser.add_argument("--sample_idx", type=int, default=15, help="在验证集中抽取第几个样本进行可视化")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 【关键修改 2】：实例化 MLP 模型
    print(f" $ 正在挂载 MLP 模型权重: {args.checkpoint}")
    model = MLPBaseline(config=config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f" $ 正在加载验证集数据...")
    val_dataset = PoseSequenceDataset(config, split="valid")

    # 获取数据与推理
    (x_tensor, dt_tensor), y_tensor = val_dataset[args.sample_idx]
    x_input = x_tensor.unsqueeze(0).to(device)
    dt_input = dt_tensor.unsqueeze(0).to(device)
    y_target = y_tensor.numpy().flatten()

    with torch.no_grad():
        logits = model((x_input, dt_input))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    mzeni_feature = x_tensor[:, 66].numpy()
    mzeni_norm = (mzeni_feature - np.min(mzeni_feature)) / (np.max(mzeni_feature) - np.min(mzeni_feature) + 1e-6)
    peaks, _ = find_peaks(probs, height=0.3, distance=5)

    # 画图
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.plot(np.arange(len(probs)), mzeni_norm, label="M-Zeni 物理约束信号 (Normalized)", color="#2ca02c",
            linestyle="-.", linewidth=2, alpha=0.7)
    ax.plot(np.arange(len(probs)), y_target, label="真实触地标签 (Ground Truth)", color="#d62728", linestyle="--",
            linewidth=2.5, alpha=0.8)

    # 【关键修改 3】：图例和标题改名
    ax.plot(np.arange(len(probs)), probs, label="MLP 基线模型预测概率", color="#ff7f0e", linewidth=3)  # 换成橙色对比

    if len(peaks) > 0:
        ax.plot(np.arange(len(probs))[peaks], probs[peaks], "x", color="black", markersize=12, markeredgewidth=3,
                label="预测触地点 (Detected Peaks)")

    ax.set_title("跑姿触地关键帧提取结果可视化 (无时序记忆的 MLP 基线)", fontsize=16, fontweight='bold')
    ax.set_xlabel("视频帧序号 (Frames)", fontsize=14)
    ax.set_ylabel("激活概率 / 归一化特征幅值", fontsize=14)
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(0, len(probs))
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    plt.tight_layout()
    save_dir = "result/MLP_result"
    os.makedirs(save_dir, exist_ok=True)
    final_save_path = os.path.join(save_dir, f"mlp_sample_{args.sample_idx}_visualization.png")
    plt.savefig(final_save_path)
    print(f"[成功] 可视化图表已保存至: {final_save_path}")


if __name__ == "__main__":
    main()