"""
detector.py
===========
多尺度检测器模块

位置：detect/detector.py
职责：
    1. 接收彩色/灰度图像。
    2. 计算整张图的积分图。
    3. 按照 scale_factor 遍历多个尺度。
    4. 在每个尺度下按步长滑动窗口，计算方差并调用 CascadeClassifier 判决。
    5. 返回候选的 bounding boxes，供 NMS 过滤。
"""

import cv2
import numpy as np
from typing import List, Tuple

from train.integral_image import build, window_variance  
from train.haar_features import iter_scales, BASE_WIN_SIZE
from detect.cascade_classifier import CascadeClassifier

class Detector:
    def __init__(self, cascade: CascadeClassifier, scale_factor: float = 1.25, step_delta: float = 1.0):
        self.cascade = cascade
        self.scale_factor = scale_factor
        self.step_delta = step_delta
        self.min_face_size = 24
        self.max_face_size = 500

    def detect(self, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if len(img_bgr.shape) == 3:
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_bgr

        H, W = img_gray.shape
        iimg = build(img_gray)
        candidates = []

        scales = iter_scales(H, W, 
                             scale_factor=self.scale_factor, 
                             min_face_px=self.min_face_size, 
                             max_face_px=self.max_face_size)

        for scale in scales:
            win_size_px = int(round(BASE_WIN_SIZE * scale))
            
            # 【优化 2：动态步长】
            # 步长随尺度放大。比如 base 步长为 2，随着 scale 变大，步长变成 3, 4, 5...
            # step_delta = 1.0 时偏精度，step_delta = 1.5 时偏速度
            step = max(1, int(round(scale * self.step_delta * 2))) 
            
            # 这里如果不使用外部的 iter_window_positions，可以直接写循环，更直观
            for r in range(0, H - win_size_px + 1, step):
                for c in range(0, W - win_size_px + 1, step):
                    
                    # 4.1 O(1) 计算子窗口方差
                    var = window_variance(iimg, r, c, win_size_px)
                    
                    # 【优化 1：低方差快速拒识】
                    # 如果方差小于某个极小值(如 10.0，代表图像太平滑)，绝对不是人脸，直接跳过
                    # 这样可以免除大量的 predict_window 函数调用开销
                    if var < 10.0:
                        continue
                    
                    # 4.2 丢给级联分类器判断
                    is_face = self.cascade.predict_window(
                        ii=iimg.ii,
                        scale=scale,
                        win_r=r,
                        win_c=c,
                        variance=var
                    )
                    
                    # 4.3 记录候选框 (x, y, w, h) -> (c, r, w, h)
                    if is_face:
                        candidates.append((c, r, win_size_px, win_size_px))

        return candidates