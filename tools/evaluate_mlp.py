# 文件路径: tools/evaluate_mlp.py
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pose_dataset import PoseSequenceDataset
from utils.metrics import SpareseKeyframeMetrics

# 【关键修改 1】：改为导入 MLP 基线模型
from models.mlp_baseline import MLPBaseline

# 设置 plt 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def parse_args():
    parser = argparse.ArgumentParser(description="Struct-LNN 测试机评估工具")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型权重路径（.pth）")
    parser.add_argument("--save_dir", type=str, default="result", help="图表保存目录")
    return parser.parse_args()

def plot_confusion_matrix(result, total_frames, save_dir):
    """
    绘制混淆矩阵
    """
    tp = int(result['TP'])
    fp = int(result['FP'])
    fn = int(result['FN'])
    tn = int(total_frames) - (tp + fp + fn)
    fig, ax = plt.subplots(figsize=(7,6), dpi=300)
    cm = np.array([[tn, fp], [fn, tp]])

    # 绘制热力图底色
    cax = ax.matshow(np.log1p(cm), cmap='Blues', alpha=0.8)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if i==j else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测：腾空', '预测：触地'])
    ax.set_yticklabels(['真实：腾空', '真实：触地'])
    ax.set_title("关键帧混淆矩阵", pad=20, fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f" 图表 1 已保存：{save_path}")

def plot_metrics_bar(result, save_dir):
    """
    性能指标柱状图
    """
    fig, ax = plt.subplots(figsize=(7,6), dpi=300)
    metrics_names = ['精确率（Precision）', '召回率（Recall）', 'F1 分数']
    metrics_values = [result['Precision'], result['Recall'], result['F1_Score']]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']

    bars = ax.bar(metrics_names, metrics_values, color=colors, width=0.5, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_title("Test集核心性能表现", pad=20, fontsize=16, fontweight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0,5),
                    textcoords='offset points',
                    ha='center',va='bottom',fontsize=14,fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "metrics_bar.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f" 图表 2 已保存：{save_path}")

def plot_event_outcomes_pie(result, save_dir):
    """
    事件诊断双饼图
    """
    tp, fp, fn = int(result['TP']), int(result['FP']), int(result['FN'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # 左侧饼图：真实事件去向（召回视角）
    labels_gt = [f'成功检测 (TP)\n{tp}次', f'不幸漏报 (FN)\n{fn}次']
    sizes_gt = [tp, fn]
    colors_gt = ['#2ca02c', '#9370DB']
    explode_gt = (0.05, 0) # 凸显出TP

    ax1.pie(sizes_gt, explode=explode_gt, labels=labels_gt, colors=colors_gt,
            autopct='%1.1f%%', shadow=False, startangle=90,
            labeldistance=1.15, pctdistance=0.75,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title("真实触地事件被检测占比\n(Ground Truth Coverage)", fontsize = 14, fontweight='bold')

    # 右侧饼图：模型报警纯度 (精确视角)
    labels_pred = [f'正确报警 (TP)\n{tp}次', f'错误报警 (FP)\n{fp}次']
    sizes_pred = [tp, fp]
    colors_pred = ['#1f77b4', '#d62728']
    explode_pred = (0.05, 0)  # 凸显 TP

    ax2.pie(sizes_pred, explode=explode_pred, labels=labels_pred, colors=colors_pred,
            autopct='%1.1f%%', shadow=False, startangle=90,
            labeldistance=1.15,pctdistance=0.75,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title("模型输出报警准确率占比\n(Prediction Purity)", fontsize=14, fontweight='bold')

    plt.subplots_adjust(top=0.85,
                        bottom=0.15,
                        left=0.08,
                        right=0.92,
                        wspace=0.3)
    save_path = os.path.join(save_dir, "event_outcomes_pie.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f" 图表 3 已保存: {save_path}")

def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] MLP 评估挂载设备: {device}")

    test_dataset = PoseSequenceDataset(config, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 【关键修改 2】：实例化 MLP 模型
    model = MLPBaseline(config=config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[*] 成功加载 MLP 权重，该权重来自 Epoch: {checkpoint.get('epoch', 'N/A')}")
    model.eval()

    tolerance = config['evaluation'].get('tolerance_windows', 3)
    metrics = SpareseKeyframeMetrics(tolerance=tolerance, from_logits=True, threshold=0.3)
    metrics.reset()

    print(f"\n[Eval] 正在执行容差匹配评估 (Tolerance: ±{tolerance} frames)...")

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = model(batch_data)
            metrics.update(logits, batch_labels)

    result = metrics.compute()
    total_frames = len(test_dataset) * config['training'].get('seq_len', 64)

    # 【关键修改 3】：修改报告标题
    print("\n" + "=" * 50)
    print(f"         MLP 基线模型 最终评估报告")
    print("=" * 50)
    print(f" 匹配容差 (Tolerance) : ±{tolerance} 帧")
    print(f" 测试集样本数         : {len(test_dataset)} (总计 {total_frames} 帧)")
    print("-" * 50)
    print(f" 真正例 (True Positives)  : {result['TP']}")
    print(f" 假正例 (False Positives) : {result['FP']} (误报)")
    print(f" 假阴性 (False Negatives) : {result['FN']} (漏报)")
    print("-" * 50)
    print(f" 精确率 (Precision)       : {result['Precision']:.4f}")
    print(f" 召回率 (Recall)          : {result['Recall']:.4f}")
    print(f" F1-Score                 : {result['F1_Score']:.4f}")
    print("=" * 50)

    # 保存图表到专属的 MLP_result 目录，防止覆盖原来的图
    save_dir = os.path.join(args.save_dir, "MLP_result")
    os.makedirs(save_dir, exist_ok=True)

    print("\n[绘图模块] 正在生成独立的学术图表...")
    plot_confusion_matrix(result, total_frames, save_dir)
    plot_metrics_bar(result, save_dir)
    plot_event_outcomes_pie(result, save_dir)
    print(f"[绘图模块] 全部图表已生成完毕！保存在 {save_dir}")


if __name__ == "__main__":
    main()