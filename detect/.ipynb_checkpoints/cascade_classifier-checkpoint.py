"""
cascade_classifier.py
=====================
级联分类器推理模块（推理流水线核心）

位置：detect/cascade_classifier.py
职责：
    1. 从磁盘加载序列化的级联模型。
    2. 提供单窗口的级联推理接口 predict_window()。
    3. 严格实现论文中的早期拒绝（Early Rejection）机制。

原理（论文 Section 4, Attentional Cascade）：
    检测窗口依次通过每一层强分类器。
    如果任意一层的结果为 0（拒绝），则立即返回，停止计算后续特征。
    只有通过所有层，才判定为候选人脸。
    
注意：在推理阶段，特征值需要除以该窗口的标准差（sigma）以完成方差归一化，
这与训练时 data_loader 对图像整体做方差归一化的逻辑等效。
"""

import pickle
from typing import List
import numpy as np
import sys
sys.path.append("D:\cv\大作业\object_detection_system\train")  # 确保能导入训练模块
# 导入成员 A 编写的数据结构和函数
from train.adaboost import StrongClassifier
from train.haar_features import compute_feature_at_scale

class CascadeClassifier:
    """
    级联分类器推理引擎。
    """
    def __init__(self, model_path: str):
        # 你的逻辑：加载 pickle 模型
        import pickle
        with open(model_path, "rb") as f:
            self.stages = pickle.load(f)
        # ----------------
    def load(self, model_path: str):
        """
        从磁盘加载序列化的级联模型。
        """
        print(f"[Cascade] 正在加载级联模型: {model_path} ...")
        with open(model_path, "rb") as f:
            self.stages = pickle.load(f)
        print(f"[Cascade] 模型加载成功！共包含 {len(self.stages)} 层级联。")
        for i, stage in enumerate(self.stages):
            print(f"  - 第 {i+1} 层: {len(stage.weak_classifiers)} 个弱分类器，阈值 {stage.threshold:.4f}")

    def predict_window(self, 
                       ii: np.ndarray, 
                       scale: float, 
                       win_r: int, 
                       win_c: int, 
                       variance: float) -> int:
        """
        对单个滑动窗口进行级联分类（早期拒绝机制）。
        
        参数：
            ii       : padded 积分图 (H+1, W+1)，即 IntegralImage.ii
            scale    : 当前窗口的缩放比例（基础窗口是 24x24）
            win_r    : 窗口左上角在原图中的行坐标
            win_c    : 窗口左上角在原图中的列坐标
            variance : 该窗口的像素方差（用于动态光照归一化）
            
        返回：
            1 表示接受（人脸），0 表示拒绝（非人脸）
        """
        # 方差过小的区域通常是纯色（如纯白墙壁、纯黑背景），直接拒绝
        if variance < 1e-4:
            return 0
            
        # 计算标准差，用于对特征值进行归一化
        sigma = np.sqrt(variance)
        
        # 依次通过级联的每一层
        for stage_idx, stage in enumerate(self.stages):
            stage_score = 0.0
            
            # 计算当前层内所有弱分类器的加权得分
            for wc in stage.weak_classifiers:
                # 1. 在积分图上按 scale 动态计算该 Haar 特征的绝对值
                raw_feat_val = compute_feature_at_scale(
                    desc=wc.feature_idx, # 注意：训练时序列化保存的需是 FeatureDesc 对象，或在外部保有映射表
                    ii=ii, 
                    scale=scale, 
                    win_r=win_r, 
                    win_c=win_c
                )
                
                # 【关键修正】除以 scale 的平方，将特征值还原到 24x24 的标准尺度量级
                normalized_scale_feat = raw_feat_val / (scale * scale)
                norm_feat_val = normalized_scale_feat / sigma
                
                # 3. 弱分类器判定：h(x) = 1 if p*f(x) < p*theta else 0
                if wc.polarity * norm_feat_val < wc.polarity * wc.threshold:
                    stage_score += wc.alpha
            
            # 4. 强分类器判定：如果当前层得分低于阈值，触发早期拒绝 (Early Rejection)
            if stage_score < stage.threshold:
                return 0  # 立即判定为非人脸
                
        # 只有通过了所有的级联层，才被认为是一张人脸
        return 1