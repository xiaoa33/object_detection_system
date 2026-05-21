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
    
# detect/detector.py
import cv2
import numpy as np
from typing import List, Tuple

from train.integral_image import build, window_variance  
from train.haar_features import iter_scales, BASE_WIN_SIZE
from detect.cascade_classifier import CascadeClassifier

# 【新增】导入 nms 模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nms import nms

class Detector:
    def __init__(self, cascade: CascadeClassifier, scale_factor: float = 1.25, step_delta: float = 1.0):
        self.cascade = cascade
        self.scale_factor = scale_factor
        self.step_delta = step_delta
        self.min_face_size = 24
        self.max_face_size = 500

    # 【修改】为 detect 函数增加 iou_threshold 和 min_votes 参数
    def detect(self, img_bgr: np.ndarray, iou_threshold=0.3, min_votes=3) -> List[Tuple[int, int, int, int]]:
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
            step = max(1, int(round(scale * self.step_delta * 2))) 
            
            for r in range(0, H - win_size_px + 1, step):
                for c in range(0, W - win_size_px + 1, step):
                    # 获取最大可能的特征偏移量，防止特征框越界
                    # 基础窗口是 24x24，特征最大坐标不会超过 24
                    max_feat_offset = 24 
                    # 增加边界安全检查：
                    # 如果当前窗口位置 + 缩放后的特征最大范围 > 图像尺寸，跳过
                    if (r + int(round(max_feat_offset * scale)) >= H or 
                        c + int(round(max_feat_offset * scale)) >= W):
                        continue
                    
                    var = window_variance(iimg, r, c, win_size_px)
                    if var < 10.0:
                        continue
                    
                    is_face = self.cascade.predict_window(
                        ii=iimg.ii,
                        scale=scale,
                        win_r=r,
                        win_c=c,
                        variance=var
                    )
                    
                    if is_face:
                        candidates.append((c, r, win_size_px, win_size_px))

        # 【修改】返回前调用 nms 处理候选框
        # min_votes=3 表示同一位置至少被3个不同尺度/位移的窗口命中，才输出最终结果
        final_faces = nms(candidates, iou_threshold=iou_threshold, min_votes=min_votes)
        
        return final_faces
    
    
