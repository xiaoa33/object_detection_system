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
import os

# ─── 动态添加项目根目录到 Python 路径 ───
# 这样无论从哪个目录运行，都能正确导入 train 模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 导入成员 A 编写的数据结构和函数
from train.adaboost import StrongClassifier, WeakClassifier
from train.haar_features import compute_feature_at_scale, enumerate_features, FeatureDesc


# ═══ numpy 版本兼容性补丁 ═══
# 模型的 pickle 可能是用 numpy 2.x 训练的（路径：numpy._core），
# 但当前环境只支持 numpy 1.x（路径：numpy.core）。
# 通过自定义 Unpickler 重映射 _core → core 来解决。
_OLD_NUMPY_CORE = "numpy._core"
_NEW_NUMPY_CORE = "numpy.core"


class _CompatUnpickler(pickle.Unpickler):
    """兼容 numpy 2.x → 1.x 的 pickle 加载器"""
    def find_class(self, module, name):
        if module.startswith(_OLD_NUMPY_CORE):
            module = module.replace(_OLD_NUMPY_CORE, _NEW_NUMPY_CORE, 1)
        return super().find_class(module, name)


# ═══ pickle 兼容性补丁 ═══
# 旧版训练代码将 StrongClassifier/WeakClassifier 以
# "adaboost.StrongClassifier"/"adaboost.WeakClassifier" 路径序列化，
# 而当前 module 路径为 "train.adaboost"。pickle.load() 会按旧路径查找，
# 找不到则报 ModuleNotFoundError / AttributeError。
# 解决方案：在 sys.modules 中注册一个兼容别名。
import sys as _sys
_adaboost_holder = type(_sys)('adaboost')
_adaboost_holder.StrongClassifier = StrongClassifier
_adaboost_holder.WeakClassifier = WeakClassifier
_sys.modules['adaboost'] = _adaboost_holder


class CascadeClassifier:
    """
    级联分类器推理引擎。
    """
    def __init__(self, model_path: str, verbose: bool = False):
        self._verbose = verbose
        # 使用兼容性 Unpickler 加载 pickle 模型
        # 兼容 numpy 2.x → 1.x 的 _core 路径映射
        with open(model_path, "rb") as f:
            state = _CompatUnpickler(f).load()
        # 训练检查点格式：dict with keys 'stages','layer_idx','overall_tpr','X_neg'
        if isinstance(state, dict):
            self.stages = state['stages']
        else:
            self.stages = state

        # ─── 特征索引 → 特征描述符 转换 ───
        # 训练时 cascade_model.pkl 保存的是 checkpoint（特征索引为整数），
        # 推理前需要将整数索引替换为 FeatureDesc 对象。
        self._features_desc = enumerate_features(win_size=24)
        for stage in self.stages:
            for wc in stage.weak_classifiers:
                if isinstance(wc.feature_idx, (int, np.integer)):
                    wc.feature_idx = self._features_desc[wc.feature_idx]
        # ----------------

    def load(self, model_path: str):
        """
        从磁盘加载序列化的级联模型。
        """
        print(f"[Cascade] 正在加载级联模型: {model_path} ...")
        with open(model_path, "rb") as f:
            state = _CompatUnpickler(f).load()
        # 训练检查点格式：dict with keys 'stages','layer_idx','overall_tpr','X_neg'
        if isinstance(state, dict):
            self.stages = state['stages']
        else:
            self.stages = state
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

        # 预计算归一化因子，避免在弱分类器循环中重复做除法
        inv_norm = 1.0 / (scale * scale * sigma)

        # 缓存局部引用，减少属性访问开销
        stages = self.stages
        _compute = compute_feature_at_scale

        # 调试日志计数器（仅 verbose 模式）
        if self._verbose:
            self._debug_count = 0

        # 依次通过级联的每一层（早期拒绝机制）
        for stage_idx, stage in enumerate(stages):
            stage_score = 0.0
            wc_list = stage.weak_classifiers
            th_stage = stage.threshold

            # 计算当前层内所有弱分类器的加权得分
            for wc in wc_list:
                # 在积分图上按 scale 动态计算该 Haar 特征值
                raw_feat_val = _compute(
                    desc=wc.feature_idx,
                    ii=ii,
                    scale=scale,
                    win_r=win_r,
                    win_c=win_c
                )

                # 尺度归一化 + 方差归一化（用乘法替代除法）
                normalized_feat = raw_feat_val * inv_norm

                # 调试日志（仅 verbose 模式，每个窗口最多打印 8 条）
                if self._verbose and stage_idx <= 1:
                    self._debug_count += 1
                    if self._debug_count <= 8:
                        print(f"[DEBUG] Stage{stage_idx+1} WC: raw_feat={raw_feat_val:.1f}, "
                              f"sigma={sigma:.2f}, scale={scale:.3f}, "
                              f"norm_feat={normalized_feat:.2f}, "
                              f"threshold={wc.threshold:.2f}, polarity={wc.polarity}, "
                              f"vote={'Y' if wc.polarity * normalized_feat < wc.polarity * wc.threshold else 'N'}")

                # 弱分类器判定：h(x) = 1 if p*f(x) < p*theta else 0
                if wc.polarity * normalized_feat < wc.polarity * wc.threshold:
                    stage_score += wc.alpha

            # 强分类器判定：如果当前层得分低于阈值，触发早期拒绝
            if stage_score < th_stage:
                return 0  # 立即判定为非人脸

        # 只有通过了所有的级联层，才被认为是一张人脸
        return 1