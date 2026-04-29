import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.signal import find_peaks

from datasets.pose_dataset import PoseSequenceDataset
from models.struct_lnn import StructLNN
from LSTM_models.baseline_lstm import BaselineLSTM
from transformer_models.baseline_transformer import BaselineTransformer
from utils.metrics import SparseKeyframeMetrics

# 设置 plt 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Struct-LNN 测试机评估工具")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型权重路径（.pth）")
    parser.add_argument("--save_dir", type=str, default="result", help="图表保存目录")
    return parser.parse_args()

def plot_multiclass_metrics_bar(result, classes, save_dir):
    """图表1：分组柱状图"""
    metrics = ['Precision', 'Recall', 'F1_Score']
    data = {m: [result[c][m] for c in classes] for m in metrics}
    x = np.arange(len(classes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width, data['Precision'], width, label='精确率 (Precision)', color='#1f77b4', alpha=0.9)
    rects2 = ax.bar(x, data['Recall'], width, label='召回率 (Recall)', color='#ff7f0e', alpha=0.9)
    rects3 = ax.bar(x + width, data['F1_Score'], width, label='F1 分数', color='#2ca02c', alpha=0.9)
    ax.set_ylabel('得分 (Score)', fontweight='bold')
    ax.set_title('性能评估分组柱状图', pad=20, fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontweight='bold')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "multiclass_metrics_bar.png"), bbox_inches='tight')
    plt.close(fig)

def plot_f1_radar_chart(result, classes, save_dir):
    """图表2：F1-Score雷达图"""
    f1_scores = [result[c]['F1_Score'] for c in classes] + [result[classes[0]]['F1_Score']]
    angles = np.linspace(0, 2 * np.pi, len(classes), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300, subplot_kw=dict(polar=True))
    ax.plot(angles, f1_scores, color='#8A2BE2', linewidth=2, linestyle='solid')
    ax.fill(angles, f1_scores, color='#8A2BE2', alpha=0.25)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
    ax.set_title("各关键帧类别 F1-Score 雷达分布图", pad=20, fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(save_dir, "f1_radar_chart.png"), bbox_inches='tight')
    plt.close(fig)

def plot_macro_event_outcomes_pie(macro_result, save_dir):
    """图表3：宏观事件输出饼图"""
    tp, fp, fn = int(macro_result['TP']), int(macro_result['FP']), int(macro_result['FN'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    ax1.pie([tp, fn] if (tp + fn) > 0 else [1, 0], explode=(0.05, 0), labels=[f'TP\n{tp}', f'FN\n{fn}'],
            colors=['#2ca02c', '#9370DB'], autopct='%1.1f%%', startangle=90)
    ax1.set_title("总体检出率 (Recall View)", fontweight='bold')
    ax2.pie([tp, fp] if (tp + fp) > 0 else [1, 0], explode=(0.05, 0), labels=[f'TP\n{tp}', f'FP\n{fp}'],
            colors=['#1f77b4', '#d62728'], autopct='%1.1f%%', startangle=90)
    ax2.set_title("总体准确率 (Precision View)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "macro_pie.png"), bbox_inches='tight')
    plt.close(fig)

def plot_event_confusion_matrix(all_probs_np, all_targets_np, classes, tolerance, save_dir):
    """ 图表4：5x5混淆矩阵 """
    gt_peaks = []
    pred_peaks = []

    for c in range(len(classes)):
        g_p, _ = find_peaks(all_targets_np[:, c], height=0.5)
        for p in g_p:
            gt_peaks.append((p, c))

        p_p, _ = find_peaks(all_probs_np[:, c], height=0.3)
        for p in p_p:
            pred_peaks.append((p, c))

    gt_peaks.sort(key=lambda x: x[0])
    pred_peaks.sort(key=lambda x: x[0])

    cm = np.zeros((6, 6), dtype=int)
    matched_pred = set()

    for gt_t, gt_c in gt_peaks:
        best_dist = tolerance + 1
        best_pred_idx, best_pred_c = -1, -1

        for i, (pr_t, pr_c) in enumerate(pred_peaks):
            if i in matched_pred: continue
            dist = abs(gt_t - pr_t)
            if dist <= tolerance:
                if dist < best_dist or (dist == best_dist and pr_c == gt_c):
                    best_dist = dist
                    best_pred_idx, best_pred_c = i, pr_c

        if best_pred_idx != -1:
            cm[gt_c, best_pred_c] += 1
            matched_pred.add(best_pred_idx)
        else:
            cm[gt_c, 5] += 1  # 漏报 (FN)

    for i, (pr_t, pr_c) in enumerate(pred_peaks):
        if i not in matched_pred:
            cm[5, pr_c] += 1  # 误报 (FP)

    display_classes = ['LHS', 'LTO', 'RHS', 'RTO', 'TMAX', 'Background\n(None)']
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    cax = ax.matshow(cm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    for i in range(6):
        for j in range(6):
            if i == 5 and j == 5:
                ax.text(j, i, "-", ha="center", va="center", fontsize=16, fontweight="bold", color="black")
                continue
            val = int(cm[i, j])
            text_color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=14, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(display_classes, fontweight='bold', fontsize=11)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(display_classes, fontweight='bold', fontsize=11)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('模型预测类别 (Pred ->)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_ylabel('真实物理事件 (GT <—)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title("5x5 时序事件混淆矩阵", pad=20, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "event_confusion_matrix.png"), bbox_inches='tight')
    plt.close(fig)

def plot_roc_curve_and_auc(all_probs, all_targets, classes, save_dir):
    """图表5：ROC曲线"""
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(all_targets[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, color in zip(range(len(classes)), colors):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{classes[i]} (AUC = {roc_auc[i]:0.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve', pad=20, fontsize=16, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve_auc.png"), bbox_inches='tight')
    plt.close(fig)

def plot_pr_curve(all_probs, all_targets, classes, save_dir):
    """图表6：Precision-Recall curve（PR曲线）"""
    precision, recall, average_precision = {}, {}, {}
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(all_targets[:, i], all_probs[:, i])
        average_precision[i] = average_precision_score(all_targets[:, i], all_probs[:, i])
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, color in zip(range(len(classes)), colors):
        ax.plot(recall[i], precision[i], color=color, lw=2, label=f'{classes[i]} (AP = {average_precision[i]:0.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率 (Recall)', fontweight='bold')
    ax.set_ylabel('精确率 (Precision)', fontweight='bold')
    ax.set_title('Precision-Recall (PR) 曲线', pad=20, fontsize=16, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), bbox_inches='tight')
    plt.close(fig)

def plot_temporal_error_distribution(temporal_errors, classes, save_dir, tolerance=3):
    """ 图表7：时间偏移误差分布小提琴图 (加入了抗离散塌陷优化) """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    data = []
    for c in range(len(classes)):
        errors = temporal_errors.get(c, [])
        if len(errors) > 0:
            # 为离散的整数误差加上 [-0.4, 0.4] 的微小均匀噪声
            # 这使得大量堆叠在 0 的数据散开，KDE 就能画出漂亮的“肚子”
            jittered_errors = np.array(errors) + np.random.uniform(-0.4, 0.4, len(errors))
            data.append(jittered_errors)
        else:
            data.append([0])

    # 绘制小提琴图
    parts = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#8A2BE2')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    ax.set_xticks(np.arange(1, len(classes) + 1))
    ax.set_xticklabels(classes, fontweight='bold', fontsize=12)
    ax.set_ylabel('时间误差帧数 $\Delta t$ (Pred - GT)', fontweight='bold')
    ax.set_title('模型关键帧时间定位误差分布 (Temporal Error Distribution)', pad=20, fontsize=16, fontweight='bold')

    ax.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='完美对齐 (0 Error)')

    # 因为误差严格限制在 [-tolerance, tolerance] 之间，强制 Y 轴以此为界
    # 这样就算某类动作全是 0 误差，它的 Y 轴也不会被自动放大，看起来就不会是一条线
    ax.set_yticks(np.arange(-tolerance, tolerance + 1))
    ax.set_ylim(-tolerance - 1, tolerance + 1)

    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_error_distribution.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"[*] 图表7已保存。")

def plot_timeline_visualization(sample_probs, sample_targets, classes, save_dir):
    """图表8：单序列时序推理可视化"""
    seq_len = sample_probs.shape[0]
    time_axis = np.arange(seq_len)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    for i in range(5):
        ax.plot(time_axis, sample_probs[:, i], color=colors[i], lw=2, label=f'{classes[i]} Prob')
        gt_peaks, _ = find_peaks(sample_targets[:, i], height=0.5)
        for p in gt_peaks:
            ax.axvline(x=p, ymin=0, ymax=1, color=colors[i], linestyle='--', alpha=0.5)
            ax.scatter(p, 1.05, marker='v', color=colors[i], s=50)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, seq_len - 1)
    ax.set_xlabel('时间步 (Frames)', fontweight='bold')
    ax.set_ylabel('网络激活概率', fontweight='bold')
    ax.set_title('单序列多通道时序推理可视化', pad=20, fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', ncol=5, fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "timeline_visualization.png"), bbox_inches='tight')
    plt.close(fig)

def plot_event_interval_consistency(all_probs_np, all_targets_np, save_dir):
    """图表9：步态时间间隔一致性直方图"""
    l_hs_gt, _ = find_peaks(all_targets_np[:, 0], height=0.5)
    r_hs_gt, _ = find_peaks(all_targets_np[:, 2], height=0.5)
    l_hs_pred, _ = find_peaks(all_probs_np[:, 0], height=0.3)
    r_hs_pred, _ = find_peaks(all_probs_np[:, 2], height=0.3)

    def calc_intervals(peaks):
        return np.diff(peaks) if len(peaks) > 1 else []

    gt_intervals = [x for x in calc_intervals(np.sort(np.concatenate([l_hs_gt, r_hs_gt]))) if 10 < x < 100]
    pred_intervals = [x for x in calc_intervals(np.sort(np.concatenate([l_hs_pred, r_hs_pred]))) if 10 < x < 100]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    if len(gt_intervals) > 0: ax.hist(gt_intervals, bins=20, density=True, alpha=0.5, color='#2ca02c',
                                      label='Ground Truth 步频物理间隔')
    if len(pred_intervals) > 0: ax.hist(pred_intervals, bins=20, density=True, alpha=0.5, color='#1f77b4',
                                        label='Struct-LNN 预测步频间隔')

    ax.set_xlabel('事件间隔帧数 ($\Delta t$)', fontweight='bold')
    ax.set_ylabel('密度 (Density)', fontweight='bold')
    ax.set_title('步态时间间隔一致性 (Event Interval Consistency)', pad=20, fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "event_interval_consistency.png"), bbox_inches='tight')
    plt.close(fig)

def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

    test_dataset = PoseSequenceDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    backbone_type = config.get('model', {}).get('backbone', 'CfC')
    input_dim = config.get('model', {}).get('input_dim', 66)
    d_model = config.get('model', {}).get('hidden_size', 64)

    if backbone_type == "LSTM":
        print(f"[*] 评估模式：正在加载 Baseline LSTM ...")
        model = BaselineLSTM(input_dim=input_dim, hidden_size=d_model)
    elif backbone_type == "Transformer":
        print(f"[*] 评估模式：正在加载 Baseline Transformer ...")
        model = BaselineTransformer(input_dim=input_dim, hidden_dim=d_model)
    else:
        print(f"[*] 评估模式：正在加载 Struct-LNN (CfC) ...")
        model = StructLNN(config=config)
    model.to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tolerance = config['evaluation'].get('tolerance_windows', 3)
    metrics = SparseKeyframeMetrics(tolerance=tolerance, from_logits=True, threshold=0.3, num_classes=5)
    metrics.reset()

    all_probs_list, all_targets_list = [], []
    sample_probs, sample_targets = None, None

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_data = tuple(item.to(device, non_blocking=True) for item in batch_data)
            batch_labels = batch_labels.to(device, non_blocking=True)

            logits = model(*batch_data)
            metrics.update(logits, batch_labels)

            probs = torch.sigmoid(logits)

            if sample_probs is None:
                sample_probs = probs[0].cpu().numpy()
                sample_targets = batch_labels[0].cpu().numpy()

            batch_size = probs.size(0)
            seq_len = probs.size(1) if len(probs.shape) == 3 else 1
            all_probs_list.append(probs.view(batch_size * seq_len, -1).cpu().numpy())
            all_targets_list.append(batch_labels.view(batch_size * seq_len, -1).cpu().numpy())

    result = metrics.compute()
    macro = result.get('Macro', result)
    classes = ["Left_HS", "Left_TO", "Right_HS", "Right_TO", "Toe_Max"]

    total_tp = sum([result[c]['TP'] for c in classes])
    total_fp = sum([result[c]['FP'] for c in classes])
    total_fn = sum([result[c]['FN'] for c in classes])

    macro_p, macro_r, macro_f1 = macro['Precision'], macro['Recall'], macro['F1_Score']
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * (micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    support_dict = {c: (result[c]['TP'] + result[c]['FN']) for c in classes}
    total_support = sum(support_dict.values())
    if total_support > 0:
        weighted_p = sum(result[c]['Precision'] * support_dict[c] for c in classes) / total_support
        weighted_r = sum(result[c]['Recall'] * support_dict[c] for c in classes) / total_support
        weighted_f1 = sum(result[c]['F1_Score'] * support_dict[c] for c in classes) / total_support
    else:
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    all_probs_np = np.concatenate(all_probs_list, axis=0)
    all_targets_np = np.concatenate(all_targets_list, axis=0)
    all_errors = []
    # 遍历 5 个关键帧类别
    for c in range(5):
        # 分别找出真实标签和模型预测的极值点（峰值）
        g_p, _ = find_peaks(all_targets_np[:, c], height=0.5)
        p_p, _ = find_peaks(all_probs_np[:, c], height=0.3)

        matched_pred = set()
        # 对于每一个真实的物理关键帧
        for gt_t in g_p:
            best_dist = tolerance + 1
            best_pred_idx = -1

            # 在预测的关键帧中寻找距离最近的（且在容差窗口 W=3 内的）
            for i, pr_t in enumerate(p_p):
                if i in matched_pred: continue
                dist = abs(gt_t - pr_t)
                if dist <= tolerance:
                    if dist < best_dist:
                        best_dist = dist
                        best_pred_idx = i

            # 如果成功匹配到了预测帧，记录它们之间的绝对时间误差
            if best_pred_idx != -1:
                matched_pred.add(best_pred_idx)
                all_errors.append(best_dist)

    # 计算平均绝对误差
    overall_mae = np.mean(all_errors) if len(all_errors) > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"                 {backbone_type} 测试集评估报告")
    print("=" * 80)

    for c in classes:
        c_res = result.get(c, {'Precision': 0, 'Recall': 0, 'F1_Score': 0, 'TP': 0, 'FP': 0, 'FN': 0})
        print(
            f" {c:<12} | {c_res['Precision']:<10.4f} | {c_res['Recall']:<10.4f} | {c_res['F1_Score']:<10.4f} | {c_res['TP']:<5} | {c_res['FP']:<5} | {c_res['FN']:<5}")
    print("-" * 80)
    print(
        f" {'MACRO AVG':<12} | {macro_p:<10.4f} | {macro_r:<10.4f} | {macro_f1:<10.4f} | {'-':<5} | {'-':<5} | {'-':<5}")
    print(
        f" {'MICRO AVG':<12} | {micro_p:<10.4f} | {micro_r:<10.4f} | {micro_f1:<10.4f} | {total_tp:<5} | {total_fp:<5} | {total_fn:<5}")
    print(
        f" {'WEIGHTED AVG':<12} | {weighted_p:<10.4f} | {weighted_r:<10.4f} | {weighted_f1:<10.4f} | {'-':<5} | {'-':<5} | {'-':<5}")
    print("=" * 80)
    print(f" [时间定位精度] 平均绝对误差 (MAE): {overall_mae:.3f} 帧")
    print(f" (注：在 120fps 下，1帧 ≈ 8.33ms。MAE={overall_mae:.3f} 意味着误差约 {overall_mae * 8.33:.1f} 毫秒)")
    print("=" * 80)

    os.makedirs(args.save_dir, exist_ok=True)
    print("\n[绘图模块] 正在生成 8 组顶级学术图表...")
    try:
        # [基础制图]
        plot_multiclass_metrics_bar(result, classes, args.save_dir)
        plot_f1_radar_chart(result, classes, args.save_dir)
        plot_macro_event_outcomes_pie(macro, args.save_dir)

        # [处理全量数据]
        all_probs_np = np.concatenate(all_probs_list, axis=0)
        all_targets_np = np.concatenate(all_targets_list, axis=0)
        binary_targets_np = (all_targets_np >= 0.5).astype(int)

        # [调用修复好的最新版混淆矩阵 (传入 5 个参数)]
        plot_event_confusion_matrix(all_probs_np, all_targets_np, classes, tolerance, args.save_dir)

        # [高级制图]
        plot_roc_curve_and_auc(all_probs_np, binary_targets_np, classes, args.save_dir)
        plot_pr_curve(all_probs_np, binary_targets_np, classes, args.save_dir)

        temporal_errors = result.get('Temporal_Errors', {c: [] for c in range(5)})
        plot_temporal_error_distribution(temporal_errors, classes, args.save_dir)

        plot_timeline_visualization(sample_probs, sample_targets, classes, args.save_dir)
        plot_event_interval_consistency(all_probs_np, all_targets_np, args.save_dir)

        print("[绘图模块] 全部图表已生成完毕，请前往 result 目录查看！")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[错误] 绘图发生异常: {e}")

if __name__ == "__main__":
    main()