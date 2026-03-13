import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.pose_dataset import PoseDataset
from models.struct_lnn import StructLNN
from utils.metrics import SparseKeyframeMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Struct-LNN 测试集评估工具")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型权重路径 (.pth)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] 评估挂载设备: {device}")

    # 1. 实例化测试集
    test_dataset = PoseDataset(config, mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # 2. 挂载模型并加载参数
    model = StructLNN(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"[Model] 成功加载权重，该权重来自 Epoch: {checkpoint.get('epoch', 'N/A')}")
    model.eval()

    # 3. 初始化容差评估器
    tolerance = config['evaluation'].get('tolerance_windows', 3)
    metrics = SparseKeyframeMetrics(tolerance=tolerance)
    metrics.reset()

    # 4. 执行测试推理
    print(f"[Eval] 正在执行容差匹配评估 (Tolerance: ±{tolerance} frames)...")

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(test_loader, desc="Testing"):
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            # 边缘设备通常采用 FP32 或针对性量化推理，这里保持 FP32 精度验证
            logits = model(batch_data)
            metrics.update(logits, batch_labels)

    # 5. 输出综合报告
    result = metrics.compute()

    print("\n" + "=" * 40)
    print(f"        STRUCT-LNN 最终评估报告")
    print("=" * 40)
    print(f" 匹配容差 (Tolerance) : ±{tolerance} 帧")
    print(f" 测试集样本数         : {len(test_dataset)}")
    print("-" * 40)
    print(f" True Positives (TP) : {result['TP']}")
    print(f" False Positives(FP) : {result['FP']}")
    print(f" False Negatives(FN) : {result['FN']}")
    print("-" * 40)
    print(f" Precision           : {result['Precision']:.4f}")
    print(f" Recall              : {result['Recall']:.4f}")
    print(f" F1-Score            : {result['F1_Score']:.4f}")
    print("=" * 40)

if __name__ == "__main__":
    main()