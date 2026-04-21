import os
import argparse
from math import gamma

import yaml
import multiprocessing
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.pose_dataset import PoseSequenceDataset
from models.baselines.baseline_lstm import BaselineLSTM
from models.struct_lnn import StructLNN
from models.baselines.baseline_transformer import BaselineTransformer

# 从项目核心模块中导入
from engine.loss import StructLNNLoss
from engine.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Entrance')
    parser.add_argument('--config', type=str, default='configs/model_struct_lnn.yaml', help='path to config file')
    parser.add_argument("--resume", type=str, default=None, help='path to latest checkpoint (default: none)')
    return parser.parse_args()

def main():
    args = parse_args()

    print(f" [初始化] 正在加载配置文件：{args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    print(f" [初始化] 训练过程即将挂载至设备：{device}")

    # 开启硬件底层加速
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f" [数据] 正在构建多线程数据加载器...")
    train_dataset = PoseSequenceDataset(config, split='train')
    val_dataset = PoseSequenceDataset(config, split='valid')

    # 从config中载入参数
    train_cfg = config.get('training',{})
    batch_size = train_cfg.get('batch_size', 32)
    num_workers = train_cfg.get('num_workers', min(8, multiprocessing.cpu_count() // 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True, #锁页内存
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    backbone_type = config.get('model', {}).get('backbone', 'CfC')
    print(f" [模型] 正在装载架构：{backbone_type}")

    # 架构分类
    if backbone_type == 'LSTM':
        model = BaselineLSTM(config=config)
    elif backbone_type == 'Transformer':
        model = BaselineTransformer(config=config)
    else:
        model = StructLNN(config=config)

    # 损失函数实例化
    loss_cfg = train_cfg.get('loss', {})
    criterion = StructLNNLoss(
        physics_weight=loss_cfg.get('physics_penalty_weight', 1.0),
        pos_weight=loss_cfg.get('pos_weight', 60.0),
        gamma = loss_cfg.get('gamma', 2.0)
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr = train_cfg.get('learning_rate', 0.001),
        weight_decay = 1e-2
    )

    # 中断恢复逻辑
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        print(f" [恢复] 正在从 {args.resume} 恢复训练上下文...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f" [恢复] 恢复成功！将从第{start_epoch + 1} 个 Epoch 继续。")

    print(f"数据模型装配完毕...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config
    )
    trainer.start_epoch = start_epoch
    trainer.run()

if __name__ == '__main__':
    main()