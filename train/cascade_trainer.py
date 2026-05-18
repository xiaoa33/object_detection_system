"""
cascade_trainer.py
==================
级联分类器训练与 Hard Negative Mining (HNM) 模块
"""

import os
import time
import pickle
import numpy as np
from typing import List

from adaboost import train_adaboost, adjust_threshold_for_detection_rate, StrongClassifier
from haar_features import compute_all_features

class CascadeTrainer:
    def __init__(self, 
                 X_pos_train: np.ndarray, y_pos_train: np.ndarray,
                 X_neg_train: np.ndarray, y_neg_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 target_fpr=1e-5,
                 layer_max_fpr=0.50,
                 layer_min_dr=0.995,
                 checkpoint_path="../models/cascade_checkpoint.pkl", # [新增] 检查点保存路径
                 resume_state=None                                   # [新增] 恢复的断点状态字典
    ):
        # 数据集引用
        self.X_pos = X_pos_train
        self.y_pos = y_pos_train
        self.X_val = X_val
        self.y_val = y_val

        # 超参数
        self.F_target = target_fpr
        self.f_target = layer_max_fpr
        self.d_target = layer_min_dr
        self.checkpoint_path = checkpoint_path
        
        # [新增] 断点续训初始化逻辑
        if resume_state is not None:
            print(f"[Checkpoint] 检测到断点状态，正在恢复...")
            self.stages = resume_state['stages']
            self.start_layer_idx = resume_state['layer_idx']
            self.overall_fpr = resume_state['overall_fpr']
            # 恢复经过 HNM 筛选后的负样本
            self.X_neg = resume_state['X_neg']
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"  -> 恢复完成！将从第 {self.start_layer_idx + 1} 层继续训练。当前整体 FPR: {self.overall_fpr:.2e}")
        else:
            self.stages: List[StrongClassifier] = []
            self.start_layer_idx = 0
            self.overall_fpr = 1.0
            self.X_neg = X_neg_train
            self.y_neg = y_neg_train

    def evaluate_stage_on_val(self, stage: StrongClassifier) -> tuple:
        """评估某一层在验证集上的 TPR (检测率) 和 FPR (假正率)"""
        preds = stage.classify(self.X_val)
        val_pos_mask = (self.y_val == 1)
        val_neg_mask = (self.y_val == 0)
        
        tp = np.sum((preds == 1) & val_pos_mask)
        fp = np.sum((preds == 1) & val_neg_mask)
        
        tpr = tp / np.sum(val_pos_mask) if np.sum(val_pos_mask) > 0 else 0
        fpr = fp / np.sum(val_neg_mask) if np.sum(val_neg_mask) > 0 else 0
        return tpr, fpr

    # [新增] 状态保存函数
    def _save_checkpoint(self, layer_idx: int, overall_fpr: float):
        """将当前训练进度序列化到磁盘"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        state = {
            'stages': self.stages,
            'layer_idx': layer_idx,
            'overall_fpr': overall_fpr,
            'X_neg': self.X_neg  # 必须保存当前的负样本，避免重新 HNM
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Checkpoint] 第 {layer_idx} 层训练进度已保存至: {self.checkpoint_path}")

    def train(self) -> List[StrongClassifier]:
        """执行论文 Table 2 的层级级联训练循环。"""
        # [修改] 使用断点中的初始状态
        overall_fpr = self.overall_fpr  
        layer_idx = self.start_layer_idx
        
        print(f"\n{'='*60}")
        print(f"[Cascade] 开始/继续训练级联分类器")
        print(f"  目标整体FPR : {self.F_target:.2e}")
        print(f"  层目标DR  : {self.d_target:.4f}, 单层最大FPR : {self.f_target:.4f}")
        print(f"{'='*60}")

        while overall_fpr > self.F_target:
            layer_idx += 1
            print(f"\n[Cascade] >>> 开始训练第 {layer_idx} 层 <<<")
    
            # 前几层追求速度，特征少；后几层追求精度，特征多
            if layer_idx <= 2:
                n_features = 2 + (layer_idx - 1) * 3  # 第1层:2, 第2层:5
            else:
                n_features = 10 + (layer_idx - 3) * 10 # 第3层以后:10, 20, 30...
            current_stage_fpr = 1.0
            best_stage_model = None
            
            X_train = np.vstack([self.X_pos, self.X_neg])
            y_train = np.hstack([self.y_pos, self.y_neg])

            while current_stage_fpr > self.f_target:
                print(f"  [Layer {layer_idx}] 尝试使用 {n_features} 个弱分类器...")
                
                stage_model = train_adaboost(X_train, y_train, n_features_to_select=n_features, verbose=False)
                
                adjust_threshold_for_detection_rate(
                    stage_model, self.X_pos, self.y_pos, target_detection_rate=self.d_target
                )
                
                current_stage_tpr, current_stage_fpr = self.evaluate_stage_on_val(stage_model)
                print(f"  [Layer {layer_idx}] {n_features} 个特征 -> 验证集 TPR={current_stage_tpr*100:.2f}%, FPR={current_stage_fpr*100:.2f}%")
                
                best_stage_model = stage_model
                max_features_per_stage = 200 

                if current_stage_fpr > self.f_target:
                    n_features += max(2, int(n_features * 0.5))
                    if n_features > max_features_per_stage:
                        print("警告: 单层特征数达到上限，强行终止本层训练以防止过拟合！")
                        break
                    
            self.stages.append(best_stage_model)
            overall_fpr = overall_fpr * current_stage_fpr
            
            print(f"[Cascade] 第 {layer_idx} 层训练成功！本层包含 {len(best_stage_model.weak_classifiers)} 个特征。")
            print(f"[Cascade] 当前系统整体假正率: {overall_fpr:.2e} (目标: {self.F_target:.2e})")
            
            if overall_fpr <= self.F_target:
                print("\n[Cascade] 达到系统目标 FPR，停止级联训练。")
                self._save_checkpoint(layer_idx, overall_fpr) # [新增] 结束时也保存一次
                break
                
            # 4. Hard Negative Mining
            print(f"\n[Cascade] 执行 Hard Negative Mining (HNM)...")
            self.X_neg = self._mine_hard_negatives()
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32) # [新增] 更新负样本标签对齐新长度

            # [新增] 每一层训练并挖掘完难例后，自动保存检查点
            self._save_checkpoint(layer_idx, overall_fpr)

        return self.stages

    def _mine_hard_negatives(self) -> np.ndarray:
            print("  [HNM] 使用当前级联模型过滤负样本池...")
            start_time = time.time()
            
            # 初始时，所有当前负样本都被认为是活跃的 (True)
            active_mask = np.ones(len(self.X_neg), dtype=bool)
            
            # 逐层过滤
            for stage_idx, stage in enumerate(self.stages):
                if not np.any(active_mask):
                    break # 如果全被过滤掉了，提前结束
                    
                # 取出当前仍然被误判为正例的负样本
                current_hard_neg = self.X_neg[active_mask]
                
                # 【核心优化】一次性批量预测，而不是一个一个循环
                preds = stage.classify(current_hard_neg)
                
                # 更新掩码：只有预测为 1 (仍被误判为正例) 的样本才保留到下一层
                # 注意这里巧妙地更新原始的 active_mask
                active_mask[active_mask] = (preds == 1)
                
            hard_neg_arr = self.X_neg[active_mask]
            
            print(f"  [HNM] 耗时: {time.time() - start_time:.2f}s | 挖掘出 {len(hard_neg_arr)} 个难负样本供下一层学习。")
            
            if len(hard_neg_arr) < 50:
                print("  [HNM] 警告：难负样本太少，级联可能已收敛或过拟合。")
                
            return hard_neg_arr