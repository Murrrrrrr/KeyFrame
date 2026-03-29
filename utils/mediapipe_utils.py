import cv2
import numpy as np
import mediapipe as mp

class MediaPipePoseExtractor:
    """
    封装好的 MediaPipe 姿态估计
    负责从 BGR 视频帧中提取三维骨架，并将其映射为下游模型所需的 H36M 关节拓扑
    """