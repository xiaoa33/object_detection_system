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
import cv2
from train.integral_image import build
from train.adaboost import train_adaboost, adjust_threshold_for_detection_rate, StrongClassifier
from train.haar_features import compute_all_features, compute_feature_at_scale

class CascadeTrainer:
    def __init__(self, 
                 X_pos_train: np.ndarray, y_pos_train: np.ndarray,
                 X_neg_train: np.ndarray, y_neg_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray,
                 target_fpr=1e-5,
                 layer_max_fpr=0.50,
                 layer_min_dr=0.995,
                 checkpoint_path="../models/cascade_checkpoint.pkl",
                 resume_state=None,
                 features_desc=None,              
                 background_dir="../data/train/negative"  
    ):
        # 数据集引用
        self.X_pos = X_pos_train
        self.y_pos = y_pos_train
        self.X_val = X_val
        self.y_val = y_val
        
        self.features_desc = features_desc
        self.background_dir = background_dir

        # 超参数
        self.F_target = target_fpr
        self.f_target = layer_max_fpr
        self.d_target = layer_min_dr
        self.checkpoint_path = checkpoint_path
        
        if resume_state is not None:
            print(f"[Checkpoint] 检测到断点状态，正在恢复...")
            self.stages = resume_state['stages']
            self.start_layer_idx = resume_state['layer_idx']
            self.overall_fpr = resume_state['overall_fpr']
            self.overall_tpr = resume_state.get('overall_tpr', 1.0)
            self.X_neg = resume_state['X_neg']
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"  -> 恢复完成！将从第 {self.start_layer_idx + 1} 层继续训练。当前整体 FPR: {self.overall_fpr:.2e}")
        else:
            self.stages: List[StrongClassifier] = []
            self.start_layer_idx = 0
            self.overall_fpr = 1.0
            self.overall_tpr = 1.0  # 初始累积检测率为 1.0
            self.X_neg = X_neg_train
            self.y_neg = y_neg_train

    # 【修复问题 1】引入级联整体评估方法
    def evaluate_cascade_on_val(self, current_stage: StrongClassifier) -> tuple:
        """评估包含当前层的完整级联（第 1 层 ~ 第 i 层）在验证集上的累积 TPR 和 FPR。"""
        all_stages = self.stages + [current_stage]
        
        # 初始所有样本均预测为正（1）
        preds = np.ones(len(self.X_val), dtype=np.int32)
        
        # 依次运行每一层级联，若某一层预测为 0 (拒绝)，则将其标记为 0，后续不再进入下层计算
        for stage in all_stages:
            still_positive_idx = np.where(preds == 1)[0]
            if len(still_positive_idx) == 0:
                break
            stage_preds = stage.classify(self.X_val[still_positive_idx])
            
            # 记录被当前层拒绝的样本
            preds[still_positive_idx[stage_preds == 0]] = 0
            
        val_pos_mask = (self.y_val == 1)
        val_neg_mask = (self.y_val == 0)
        
        tp = np.sum((preds == 1) & val_pos_mask)
        fp = np.sum((preds == 1) & val_neg_mask)
        
        tpr = tp / np.sum(val_pos_mask) if np.sum(val_pos_mask) > 0 else 0.0
        fpr = fp / np.sum(val_neg_mask) if np.sum(val_neg_mask) > 0 else 0.0
        return tpr, fpr

    def _save_checkpoint(self, layer_idx: int, overall_fpr: float):
        """将当前训练进度序列化到磁盘"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        state = {
            'stages': self.stages,
            'layer_idx': layer_idx,
            'overall_fpr': overall_fpr,
            'overall_tpr': self.overall_tpr,
            'X_neg': self.X_neg
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Checkpoint] 第 {layer_idx} 层训练进度已保存至: {self.checkpoint_path}")
        
        
    def train(self) -> List[StrongClassifier]:
        """执行级联训练循环。"""
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
    
            # 【优化：自适应特征继承】
            # 如果已有训练好的前序层，则从前一层特征数的 80% 开始尝试，避免从 2 开始导致的无效重复训练
            if self.stages:
                prev_features = len(self.stages[-1].weak_classifiers)
                n_features = max(2, int(prev_features * 0.8))
                print(f"  [Layer {layer_idx}] 自适应继承：前一层包含 {prev_features} 个特征，本层搜索起点设为 {n_features} 个特征。")
            else:
                n_features = 2
            
            current_stage_fpr = 1.0
            best_stage_model = None
            
            X_train = np.vstack([self.X_pos, self.X_neg])
            y_train = np.hstack([self.y_pos, self.y_neg])

            # 计算当前层需要达到的级联累积 TPR 目标: Di >= d * Di-1
            target_cascade_tpr = self.d_target * self.overall_tpr

            while current_stage_fpr > self.f_target:
                print(f"  [Layer {layer_idx}] 尝试使用 {n_features} 个弱分类器...")
                
                # 训练强分类器
                stage_model = train_adaboost(X_train, y_train, n_features_to_select=n_features, verbose=False)
                
                # 粗调阈值
                adjust_threshold_for_detection_rate(
                    stage_model, self.X_pos, self.y_pos, target_detection_rate=self.d_target
                )
                
                # 在验证集上微调阈值
                cascade_tpr, cascade_fpr = self.evaluate_cascade_on_val(stage_model)
                while cascade_tpr < target_cascade_tpr and stage_model.threshold > -50.0:
                    stage_model.threshold -= 0.05
                    cascade_tpr, cascade_fpr = self.evaluate_cascade_on_val(stage_model)
                    print(f"    [Threshold Adjust] 验证集检测率不足，微调阈值降至 {stage_model.threshold:.3f} | Cascade TPR={cascade_tpr*100:.2f}% (目标: {target_cascade_tpr*100:.2f}%)")

                # 计算单层等效假正率
                current_stage_fpr = cascade_fpr / overall_fpr if overall_fpr > 0 else 0.0
                
                print(f"  [Layer {layer_idx}] {n_features} 个特征 -> 级联验证集累积 TPR={cascade_tpr*100:.2f}%, 单层等效 FPR={current_stage_fpr*100:.2f}% (目标 <= {self.f_target*100:.2f}%)")
                
                best_stage_model = stage_model
                max_features_per_stage = 200 

                if current_stage_fpr > self.f_target:
                    # 如果未达标，则按一定比例增加特征数
                    n_features += max(2, int(n_features * 0.3))
                    if n_features > max_features_per_stage:
                        print("    [Warning] 单层特征数达到上限，强行终止本层训练以防止过拟合！")
                        break
                    
            self.stages.append(best_stage_model)
            
            # 直接使用真实的级联 FPR 更新
            overall_fpr = cascade_fpr
            self.overall_tpr = cascade_tpr

            print(f"[Cascade] 第 {layer_idx} 层训练成功！本层包含 {len(best_stage_model.weak_classifiers)} 个特征。")
            print(f"[Cascade] 当前系统整体假正率: {overall_fpr:.2e} (目标: {self.F_target:.2e})")
            
            if overall_fpr <= self.F_target:
                print("\n[Cascade] 达到系统目标 FPR，停止级联训练。")
                self._save_checkpoint(layer_idx, overall_fpr)
                break
                
            # 执行 Hard Negative Mining
            print(f"\n[Cascade] 执行 Hard Negative Mining (HNM)...")
            self.X_neg = self._mine_hard_negatives()
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32)

            self._save_checkpoint(layer_idx, overall_fpr)

        return self.stages

    def _predict_patch(self, ii: np.ndarray, variance: float) -> bool:
        """在线预测单个 24x24 局部积分图是否能通过当前所有的级联阶段"""
        if variance < 1e-4:
            return False
        sigma = np.sqrt(variance)
        
        # 依次运行目前已训练出的所有 Stage
        for stage in self.stages:
            stage_score = 0.0
            for wc in stage.weak_classifiers:
                feat_desc = self.features_desc[wc.feature_idx] if isinstance(wc.feature_idx, (int, np.integer)) else wc.feature_idx
                raw_feat_val = compute_feature_at_scale(
                    desc=feat_desc,
                    ii=ii,
                    scale=1.0,
                    win_r=0,
                    win_c=0
                )
                norm_feat_val = raw_feat_val / sigma
                if wc.polarity * norm_feat_val < wc.polarity * wc.threshold:
                    stage_score += wc.alpha
            if stage_score < stage.threshold:
                return False
        return True

    def _mine_hard_negatives(self) -> np.ndarray:
        """
        真正的 Hard Negative Mining：
        扫描 background_dir 下的所有图像，利用滑动窗口判定，收集新的 False Positives。
        若不具备背景大图像源（例如文件夹仅含小尺寸切片），则自动降级为对现有负样本库进行特征过滤。
        """
        if not self.background_dir or not os.path.exists(self.background_dir):
            print(f"  [HNM] 警告：背景图文件夹不存在，自动降级为旧版负特征过滤模式...")
            return self._fallback_filter_negatives()

        bg_files = [os.path.join(self.background_dir, f) for f in os.listdir(self.background_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not bg_files:
            print(f"  [HNM] 警告：在背景文件夹中未检测到图片，自动降级为旧版负特征过滤模式...")
            return self._fallback_filter_negatives()

        print(f"  [HNM] 正在扫描背景图像。目标搜集数量: 1000 个...")
        start_time = time.time()
        
        target_hnm_count = 1000
        mined_patches = []
        
        np.random.shuffle(bg_files)
        found_count = 0
        
        for file_path in bg_files:
            if found_count >= target_hnm_count:
                break
                
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 多尺度下采样扫描
            h_orig, w_orig = img.shape
            scales = [1.0, 0.75, 0.5, 0.3]
            
            for sc in scales:
                if found_count >= target_hnm_count:
                    break
                
                nh, nw = int(h_orig * sc), int(w_orig * sc)
                if nh < 24 or nw < 24:
                    continue
                    
                resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
                
                step = 12
                for r in range(0, nh - 24 + 1, step):
                    for c in range(0, nw - 24 + 1, step):
                        if found_count >= target_hnm_count:
                            break
                            
                        patch = resized_img[r:r+24, c:c+24]
                        
                        var = np.var(patch)
                        if var < 10.0:
                            continue
                            
                        patch_iimg = build(patch)
                        if self._predict_patch(patch_iimg.ii, var):
                            mined_patches.append(patch)
                            found_count += 1

        print(f"  [HNM] 扫描完成！耗时: {time.time() - start_time:.2f}s | 共计挖出: {len(mined_patches)} / {target_hnm_count}")
        
        if len(mined_patches) > 0:
            normalized_patches = []
            for p in mined_patches:
                mean = np.mean(p)
                std = np.std(p)
                if std < 1e-4:
                    std = 1.0
                normalized_patches.append((p - mean) / std)
            
            from train.integral_image import build_batch
            iimgs_mined = build_batch(normalized_patches)
            
            print("  [HNM] 正在对挖出的难例进行特征提取...")
            X_hnm = compute_all_features(iimgs_mined, self.features_desc, scale=1.0).astype(np.float32)
            return X_hnm
        else:
            return self._fallback_filter_negatives()

    def _fallback_filter_negatives(self) -> np.ndarray:
        """退化/备用策略：过滤当前的负特征空间"""
        current_hard_neg = self.X_neg
        for stage in self.stages:
            if len(current_hard_neg) == 0:
                break
            preds = stage.classify(current_hard_neg)
            current_hard_neg = current_hard_neg[preds == 1]
        
        if len(current_hard_neg) < 200:
            fallback_indices = np.random.choice(len(self.X_neg), min(500 - len(current_hard_neg), len(self.X_neg)), replace=False)
            if len(current_hard_neg) > 0:
                current_hard_neg = np.vstack([current_hard_neg, self.X_neg[fallback_indices]])
            else:
                current_hard_neg = self.X_neg[fallback_indices]
        return current_hard_neg