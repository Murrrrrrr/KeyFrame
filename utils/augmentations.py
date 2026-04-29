import numpy as np

class Compose:
    """多种数据增强的方法"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RandomSpatialJitter:
    """
    空间坐标高斯抖动
    模拟硬件摄像头在光线暗或者遮挡时产生的关键点坐标微小漂移
    """
    def __init__(self, sigma=0.01, apply_prob=0.5):
        self.sigma = sigma
        self.apply_prob = apply_prob

    def __call__(self, x):
        if np.random.rand() < self.apply_prob:
            noise = np.random.normal(loc=0.0, scale=self.sigma, size=x.shape)
            x = x + noise
        return x


class TemporalDropout:
    """
    时序丢帧模拟
    将序列中随机某些帧的特征置为 0，模拟硬件传感器丢包或视觉遮挡
    """

    def __init__(self, drop_ratio=0.1, apply_prob=0.5):
        self.drop_ratio = drop_ratio
        self.apply_prob = apply_prob

    def __call__(self, x):
        if np.random.rand() < self.apply_prob:
            seq_len = x.shape[0]
            num_drops = int(seq_len * self.drop_ratio)
            drop_indices = np.random.choice(seq_len, num_drops, replace=False)

            x_aug = x.copy()
            x_aug[drop_indices, :] = 0.0
            return x_aug
        return x


class RandomScale:
    """全局尺度微调，模拟不同身高的人群或摄像头焦距微小差异"""

    def __init__(self, scale_range=(0.9, 1.1), apply_prob=0.5):
        self.scale_range = scale_range
        self.apply_prob = apply_prob

    def __call__(self, x):
        if np.random.rand() < self.apply_prob:
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            x = x * scale_factor
        return x


def get_train_transforms():
    """返回训练集默认的数据增强管道"""
    return Compose([
        RandomScale(scale_range=(0.95, 1.05), apply_prob=0.7),
        RandomSpatialJitter(sigma=0.015, apply_prob=0.6),
        TemporalDropout(drop_ratio=0.05, apply_prob=0.5)
    ])