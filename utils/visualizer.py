import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from models.struct_lnn import StructLNN
from datasets.pose_dataset import PoseSequenceDataset

# 配置 Matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def parse_args():
    parser = argparse.ArgumentParser(description='跑姿关键帧多分类可视化工具')
    parser.add_argument("--config", type=str, default="configs/struct_lnn_mzeni.yaml", help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/Running_Gait_Analysis_MZeni_best.pth", help="训练好的模型权重路径")
    parser.add_argument("--sample_idx", type=int, default=15, help="在验证集中抽取第几个样本进行可视化")
    parser.add_argument("--save_path", type=str, default="result", help="图表保存目录")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"[*] 正在加载配置文件：{args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 正在挂载 LNN 模型权重: {args.checkpoint}")
    model = StructLNN(config, num_classes=5).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到权重文件 {args.checkpoint}，请确认路径。")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[*] 正在加载验证集数据...")
    val_dataset = PoseSequenceDataset(config, split="valid")

    if args.sample_idx >= len(val_dataset):
        print(f"[警告] sample_idx ({args.sample_idx}) 超出数据集长度，自动重置为 0 ")
        args.sample_idx = 0

    (x_tensor, dt_tensor), y_tensor = val_dataset[args.sample_idx]

    x_input = x_tensor.unsqueeze(0).to(device)
    dt_input = dt_tensor.unsqueeze(0).to(device)

    # y_target 现在的形状是 [SeqLen, 5]
    y_target = y_tensor.numpy()

    print("[*] 正在进行连续时间推演...")
    with torch.no_grad():
        logits = model((x_input, dt_input))
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # 形状: [SeqLen, 5]

    # 无论前面有多少维空间或速度特征，M-Zeni 总是在最后一维
    mzeni_feature = x_tensor[:, -1].numpy()
    mzeni_norm = (mzeni_feature - np.min(mzeni_feature)) / (np.max(mzeni_feature) - np.min(mzeni_feature) + 1e-6)

    print(f"[*] 正在生成多类别级联可视化图表...")
    seq_len = probs.shape[0]
    frames = np.arange(seq_len)

    classes = ["Left_HS", "Left_TO", "Right_HS", "Right_TO", "Toe_Max"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), dpi=300, sharex=True,
                             gridspec_kw={'height_ratios': [1.5, 2.5, 2]})

    # Subplot 1: 物理先验层
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].plot(frames, mzeni_norm, label="M-Zeni 物理先验信号 (Normalized)",
                 color="gray", linestyle="-", linewidth=2.5, alpha=0.8)
    axes[0].set_title("LNN 跑姿关键帧推理可视化", fontsize=18, fontweight='bold', pad=15)
    axes[0].set_ylabel("物理幅度", fontsize=12, fontweight='bold')
    axes[0].legend(loc="upper right", fontsize=11)

    # Subplot 2: 网络概率层
    axes[1].grid(True, linestyle='--', alpha=0.5)
    for c in range(5):
        axes[1].plot(frames, probs[:, c], label=f"{classes[c]} 激活概率",
                     color=colors[c], linewidth=2.5, alpha=0.9)
    axes[1].set_ylabel("网络输出概率 (Probability)", fontsize=12, fontweight='bold')
    axes[1].set_ylim(-0.05, 1.1)
    axes[1].legend(loc="upper right", fontsize=10, ncol=5)

    # Subplot 3: 离散事件对齐层
    axes[2].grid(True, axis='x', linestyle='--', alpha=0.5)

    # 为 5 个动作分配不同的 Y 轴高度轨道 (0, 1, 2, 3, 4)，防止标记重叠
    for c in range(5):
        track_y = 4 - c  # 倒序排列轨道

        # 1. 寻找真实标签 (Ground Truth) 的波峰
        gt_peaks, _ = find_peaks(y_target[:, c], height=0.5)
        # 2. 寻找模型预测 (Prediction) 的波峰
        pred_peaks, _ = find_peaks(probs[:, c], height=0.3, distance=5)

        # 绘制 GT 辅助虚线与空心圆
        for p in gt_peaks:
            axes[2].axvline(x=p, color=colors[c], linestyle='--', alpha=0.4, linewidth=1.5)
            axes[2].plot(p, track_y, marker='o', markersize=10, markeredgewidth=2,
                         markerfacecolor='none', markeredgecolor=colors[c])

        # 绘制 Pred 实心倒三角
        for p in pred_peaks:
            axes[2].plot(p, track_y, marker='v', markersize=10, color=colors[c])

    # 绘制图例 (仅需借用占位符)
    axes[2].plot([], [], marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', linestyle='None',
                 label='真实帧 (Ground Truth)')
    axes[2].plot([], [], marker='v', markersize=8, color='black', linestyle='None', label='预测帧 (Prediction)')
    axes[2].legend(loc="upper right", fontsize=11)

    axes[2].set_yticks(np.arange(5))
    axes[2].set_yticklabels(classes[::-1], fontweight='bold', fontsize=11)
    axes[2].set_xlabel("时间步 / 视频帧序号 (Frames)", fontsize=14, fontweight='bold')
    axes[2].set_ylim(-0.5, 4.5)
    axes[2].set_xlim(0, seq_len - 1)

    plt.tight_layout()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    final_save_path = os.path.join(args.save_path, f"sample_{args.sample_idx}_timeline_vis.png")
    plt.savefig(final_save_path, bbox_inches='tight')
    print(f"[*] 成功！定性分析图表已保存至: {final_save_path}")

if __name__ == "__main__":
    main()