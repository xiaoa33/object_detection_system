"""
cascade_trainer.py
==================
级联分类器训练与 Hard Negative Mining (HNM) 模块（支持多分支交互决策与低冗余输出）
"""

import os
import time
import pickle
import numpy as np
from typing import List, Tuple
import cv2

from train.integral_image import build, build_batch
from train.adaboost import train_adaboost, StrongClassifier
from train.haar_features import compute_all_features, compute_feature_at_scale


# ─────────────────────────────────────────────────────────────
#  模块级辅助函数
# ─────────────────────────────────────────────────────────────

def _variance_normalize_patch(patch: np.ndarray):
    """
    对单个 24×24 灰度 patch 做均值/方差归一化。
    """
    mean = np.mean(patch, dtype=np.float64)
    var  = np.mean(patch.astype(np.float64) ** 2) - mean ** 2
    if var < 1e-4:
        return None        # 方差过小，无判别价值，直接跳过
    sigma = np.sqrt(var)
    return (patch.astype(np.float32) - mean) / sigma


def _collect_patches_from_dir(
    image_dir: str,
    n_samples: int,
    win_size: int = 24,
    step: int = 4,
    mode: str = 'random',
    cascade_classifier=None,
    one_per_image: bool = False,
) -> List[np.ndarray]:
    """
    从指定彩色大图中采集灰度归一化 patch。
    """
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    ]
    if not files:
        print(f"  [采样] [WARN] 目录 {image_dir} 中未找到图片！")
        return []

    np.random.shuffle(files)
    patches = []

    for img_path in files:
        if len(patches) >= n_samples:
            break

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [采样] 无法读取: {img_path}，跳过。")
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        if h < win_size or w < win_size:
            continue

        if mode == 'random':
            if one_per_image:
                max_per_img = 1
            else:
                max_per_img = max(1, n_samples // max(1, len(files) // 2))

            count_this = 0
            rows       = np.arange(0, h - win_size + 1)
            cols       = np.arange(0, w - win_size + 1)
            positions  = [(r, c) for r in rows for c in cols]
            np.random.shuffle(positions)

            for (r, c) in positions:
                if len(patches) >= n_samples or count_this >= max_per_img:
                    break
                patch_norm = _variance_normalize_patch(img_gray[r:r+win_size, c:c+win_size])
                if patch_norm is None:
                    continue
                patches.append(patch_norm)
                count_this += 1

        elif mode == 'scan':
            for r in range(0, h - win_size + 1, step):
                if len(patches) >= n_samples:
                    break
                for c in range(0, w - win_size + 1, step):
                    if len(patches) >= n_samples:
                        break
                    patch_norm = _variance_normalize_patch(img_gray[r:r+win_size, c:c+win_size])
                    if patch_norm is None:
                        continue
                    if cascade_classifier is not None and cascade_classifier(patch_norm):
                        patches.append(patch_norm)

            print(f"  [HNM] 已处理: {os.path.basename(img_path)}，"
                  f"已收集 {len(patches)}/{n_samples}")

    return patches


def _patches_to_feature_matrix(
    patches: List[np.ndarray],
    features_desc: list,
) -> np.ndarray:
    """
    将归一化 patch 列表批量转为 Haar 特征矩阵。
    """
    if len(patches) == 0:
        return np.empty((0, len(features_desc)), dtype=np.float32)
    patches_arr = np.array(patches, dtype=np.float32)
    iimgs = build_batch(patches_arr)
    return compute_all_features(iimgs, features_desc, scale=1.0).astype(np.float32)


def _next_n_features(n: int, max_n: int) -> int:
    """
    分段自适应决定下一轮尝试的特征数量。
    """
    if n < 10:
        nxt = n * 2
    elif n < 30:
        nxt = int(n * 1.5)
    elif n < 100:
        nxt = n + 25
    else:
        nxt = n + 50
    return min(nxt, max_n)


# ─────────────────────────────────────────────────────────────
#  CascadeTrainer 主类
# ─────────────────────────────────────────────────────────────

class CascadeTrainer:
    """
    依据 Viola-Jones 设计训练级联分类器的核心管理器。
    """

    def __init__(
        self,
        X_pos_train: np.ndarray,        # 训练正样本特征
        y_pos_train: np.ndarray,
        X_neg_train: np.ndarray,        # 第一层训练负样本特征
        y_neg_train: np.ndarray,
        X_val_pos: np.ndarray,          # 验证正样本特征
        y_val_pos: np.ndarray,
        X_val_neg: np.ndarray,          # 验证负样本特征（固定特征传入）
        y_val_neg: np.ndarray,
        train_neg_image_dir: str,       # 训练负样本大图目录
        features_desc: list,
        target_fpr: float = 1e-5,
        layer_max_fpr: float = 0.50,
        layer_min_dr: float = 0.99,
        max_features_per_stage: int = 200,
        hnm_step: int = 4,
        checkpoint_path: str = "models/cascade_checkpoint.pkl",
        resume_state: dict = None,
        non_interactive: bool = True,   # 非交互模式：自动选择"继续增加特征"，跳过 input()
    ):
        self.X_pos = X_pos_train
        self.y_pos = y_pos_train

        self.X_val_pos = X_val_pos
        self.y_val_pos = y_val_pos
        self.X_val_neg = X_val_neg
        self.y_val_neg = y_val_neg

        # 组合构建全程固定的验证集
        self.X_val = np.vstack([X_val_pos, X_val_neg])
        self.y_val = np.hstack([y_val_pos, y_val_neg])
        print(f"[CascadeTrainer] 验证集：正 {len(X_val_pos)} + 负 {len(X_val_neg)} "
              f"= {len(self.X_val)} 个样本（全程固定评估）")

        self.train_neg_image_dir = train_neg_image_dir
        self.features_desc       = features_desc

        self.F_target               = target_fpr
        self.f_target               = layer_max_fpr
        self.d_target               = layer_min_dr
        self.max_features_per_stage = max_features_per_stage
        self.hnm_step               = hnm_step
        self.checkpoint_path        = checkpoint_path
        self.non_interactive        = non_interactive

        # 断点加载
        if resume_state is not None:
            print("[Checkpoint] 检测到断点，正在恢复...")
            self.stages          = resume_state['stages']
            self.start_layer_idx = resume_state['layer_idx']
            self.overall_fpr     = resume_state['overall_fpr']
            self.overall_tpr     = resume_state.get('overall_tpr', 1.0)
            self.X_neg           = resume_state['X_neg']
            self.y_neg           = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"  -> 从第 {self.start_layer_idx + 1} 层继续。累计 FPR={self.overall_fpr:.2e}")
        else:
            self.stages: List[StrongClassifier] = []
            self.start_layer_idx = 0
            self.overall_fpr     = 1.0
            self.overall_tpr     = 1.0
            self.X_neg           = X_neg_train
            self.y_neg           = y_neg_train

    def _evaluate_cascade(self, current_stage: StrongClassifier, verbose: bool = True) -> Tuple[float, float]:
        """
        评估已有级联层级对固定验证集的整体表现。
        """
        all_stages = self.stages + [current_stage]
        preds      = np.ones(len(self.X_val), dtype=np.int32)

        for stage in all_stages:
            still = np.where(preds == 1)[0]
            if len(still) == 0:
                break
            stage_preds = stage.classify(self.X_val[still])
            preds[still[stage_preds == 0]] = 0

        pos_mask = (self.y_val == 1)
        neg_mask = (self.y_val == 0)
        n_pos    = np.sum(pos_mask)
        n_neg    = np.sum(neg_mask)

        tp  = np.sum((preds == 1) & pos_mask)
        fp  = np.sum((preds == 1) & neg_mask)

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        if verbose:
            print(f"    [验证集] 累积 TPR={tpr*100:.2f}% ({tp}/{n_pos})，"
                  f"FPR={fpr*100:.4f}% ({fp}/{n_neg})")
        return tpr, fpr

    def _save_checkpoint(self, layer_idx: int, overall_fpr: float) -> None:
        """保存当前阶段的中间进度（用于断点继续）"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        state = {
            'stages':      self.stages,
            'layer_idx':   layer_idx,
            'overall_fpr': overall_fpr,
            'overall_tpr': self.overall_tpr,
            'X_neg':       self.X_neg,
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Checkpoint] 第 {layer_idx} 层进度已保存: {self.checkpoint_path}")

    def train(self) -> List[StrongClassifier]:
        """
        主循环训练逻辑。
        """
        overall_fpr = self.overall_fpr
        layer_idx   = self.start_layer_idx

        print(f"\n{'='*60}")
        print(f"[Cascade] 开始训练")
        print(f"  目标整体 FPR    : {self.F_target:.2e}")
        print(f"  单层最大 FPR    : {self.f_target:.2f}")
        print(f"  单层最低 DR     : {self.d_target:.4f}")
        print(f"  验证集大小      : 正 {np.sum(self.y_val==1)} + 负 {np.sum(self.y_val==0)}（固定）")
        print(f"  交互模式        : {'关闭（自动继续）' if self.non_interactive else '开启（手动选择）'}")
        print(f"{'='*60}")

        while overall_fpr > self.F_target:
            layer_idx += 1
            print(f"\n[Cascade] ========== 第 {layer_idx} 层 ==========")

            if self.stages:
                prev_n     = len(self.stages[-1].weak_classifiers)
                n_features = max(2, prev_n)
                print(f"  [A] 前一层共 {prev_n} 个特征，本层初始候选特征数: {n_features}")
            else:
                n_features = 2
                print(f"  [A] 级联起始第一层，从 {n_features} 个特征开始...")

            X_train = np.vstack([self.X_pos, self.X_neg])
            y_train = np.hstack([self.y_pos, self.y_neg])
            print(f"  [B] 本层训练集: 正 {len(self.X_pos)} + 负 {len(self.X_neg)} = {len(X_train)} 个样本")

            target_tpr = self.d_target * self.overall_tpr
            target_fpr = self.f_target * overall_fpr
            print(f"  [B] 预期目标: 累积 TPR ≥ {target_tpr*100:.2f}%, 累积 FPR ≤ {target_fpr*100:.4f}%")

            # 用于保存本层每次特征增加时的尝试历史
            stage_history = []  # 元素格式为 dict: {'model': stage_model, 'fpr': fpr, 'tpr': tpr, 'n_features': n_features}
            best_model = None

            while True:
                print(f"\n  [C] 尝试训练含有 {n_features} 个弱分类器的强分类器...")
                t0 = time.time()

                stage_model = train_adaboost(
                    X_train, y_train,
                    n_features_to_select=n_features,
                    verbose=False,
                )
                print(f"    AdaBoost 迭代计算完成，耗时 {time.time()-t0:.1f}s")

                # 初次评估输出详细信息
                cascade_tpr, cascade_fpr = self._evaluate_cascade(stage_model, verbose=True)

                adjust_count = 0
                # 调阈值时将 verbose 设为 False，减少控制台冗余输出
                while cascade_tpr < target_tpr and stage_model.threshold > -100.0:
                    stage_model.threshold -= 0.05
                    cascade_tpr, cascade_fpr = self._evaluate_cascade(stage_model, verbose=False)
                    adjust_count += 1

                if adjust_count > 0:
                    print(f"    [阈值调整] 调整结束，降低阈值共 {adjust_count} 次，修正后 threshold={stage_model.threshold:.3f}")
                    # 阈值调整完毕后，在控制台打印一次当前最终结果
                    self._evaluate_cascade(stage_model, verbose=True)

                print(f"  [C] {n_features} 个特征 → "
                      f"累积 TPR={cascade_tpr*100:.2f}% (目标≥{target_tpr*100:.2f}%)，"
                      f"累积 FPR={cascade_fpr*100:.4f}% (目标≤{target_fpr*100:.4f}%)")

                # 记录本次尝试的历史
                current_try = {
                    'model': stage_model,
                    'fpr': cascade_fpr,
                    'tpr': cascade_tpr,
                    'n_features': n_features
                }
                stage_history.append(current_try)
                try_idx = len(stage_history) - 1  # 当前是第几次尝试 (0代表第一次)

                # ─── 核心判定逻辑 ───

                # 1. 如果当前 FPR 指标已经完美达标（且TPR也达标），则保存并结束本层
                if cascade_fpr <= target_fpr and cascade_tpr >= target_tpr:
                    print(f"      [[OK]] 本层指标已达标！FPR={cascade_fpr*100:.4f}%, TPR={cascade_tpr*100:.2f}%")
                    best_model = stage_model
                    break

                # 2. 如果未达标，根据尝试次数执行特定的中止与回滚逻辑
                if try_idx == 1:  # 第二次训练 (索引 1)
                    fpr_1st = stage_history[0]['fpr']
                    fpr_2nd = current_try['fpr']

                    if fpr_2nd > fpr_1st:
                        print(f"      [!] 警告：第二次训练的 FPR ({fpr_2nd*100:.4f}%) 比第一次 ({fpr_1st*100:.4f}%) 还要差！")
                        print(f"      [[OK]] 触发中止：保留第一次训练的特征数（{stage_history[0]['n_features']} 个），直接开启下一层！")
                        best_model = stage_history[0]['model']
                        cascade_fpr = stage_history[0]['fpr']
                        cascade_tpr = stage_history[0]['tpr']
                        break
                    elif fpr_2nd == fpr_1st:
                        print(f"      [提示] 第二次 FPR 与第一次相同 ({fpr_2nd*100:.4f}%)。")
                    else:
                        print(f"      [提示] 第二次 FPR 相比第一次有所改善 ({fpr_2nd*100:.4f}% < {fpr_1st*100:.4f}%)。")

                elif try_idx == 2:  # 第三次训练 (索引 2)
                    fpr_2nd = stage_history[1]['fpr']
                    fpr_3rd = current_try['fpr']

                    if fpr_3rd >= fpr_2nd:
                        print(f"      [!] 警告：第三次训练的 FPR ({fpr_3rd*100:.4f}%) 未能得到优化或比前两次更差！")
                        print(f"      [[OK]] 触发中止：保留第一次训练的特征数（{stage_history[0]['n_features']} 个），直接开启下一层！")
                        best_model = stage_history[0]['model']
                        cascade_fpr = stage_history[0]['fpr']
                        cascade_tpr = stage_history[0]['tpr']
                        break
                    else:
                        print(f"      [提示] 第三次 FPR 取得改善 ({fpr_3rd*100:.4f}%)，继续向下迭代。")

                # 如果已经超过三次尝试，后续的正常降温/防退化保障逻辑
                elif try_idx > 2:
                    best_idx = np.argmin([h['fpr'] for h in stage_history])
                    best_model = stage_history[best_idx]['model']
                    cascade_fpr = stage_history[best_idx]['fpr']
                    cascade_tpr = stage_history[best_idx]['tpr']

                # 3. 准备增加特征进行下一次尝试
                n_next = _next_n_features(n_features, self.max_features_per_stage)

                if n_next >= self.max_features_per_stage:
                    print(f"\n  [C] *** 警报：本层特征数已达上限 {self.max_features_per_stage}，"
                          f"但累积 FPR ({cascade_fpr*100:.4f}%) 仍未达到目标 ({target_fpr*100:.4f}%)！")
                    print("      这说明当前的稀疏特征池已无法进一步区分剩余的困难负样本。")
                    print(f"      [[OK]] 级联训练在第 {layer_idx - 1} 层安全收敛，整个训练在此处结束。")
                    return self.stages

                # ─── 三选一交互确认机制 ───
                if self.non_interactive:
                    # 非交互模式：自动选择 [1] 继续增加特征，等价于原始 VJ 行为
                    action = '1'
                    print(f"\n[自动决策] 当前层指标未达标（累计 FPR: {cascade_fpr*100:.4f}%，目标: {target_fpr*100:.4f}%）")
                    print(f"  -> 非交互模式，自动继续：特征数 {n_features} → {n_next}（+{n_next-n_features}）")
                else:
                    print(f"\n[交互决策] 当前层指标未达标（当前累计 FPR: {cascade_fpr*100:.4f}%，单层目标 FPR: {target_fpr*100:.4f}%）")
                    print("请选择下一步操作：")
                    print(f"  [1] 继续增加特征：增加至 {n_next} 个特征并重新训练本层。")
                    print("  [2] 回退并进入下一层：不加特征，使用本阶段历史尝试中效果最好（FPR最低）的版本结束本层，并开启下一层。")
                    print("  [3] 终止训练：保存已取得成果（含本层最优结果），直接结束整个级联。")

                    action = ""
                    while True:
                        action = input("请输入选项数字 [1/2/3]: ").strip()
                        if action in ['1', '2', '3']:
                            break
                        print("  [!] 输入无效，请输入数字 1, 2 或 3")

                if action == '1':
                    if not self.non_interactive:
                        print(f"  [C] 确认继续，特征数 {n_features} → {n_next}（+{n_next-n_features}）")
                    n_features = n_next
                    # 继续当前层的 while 循环

                elif action == '2':
                    print("  [C] 确认回退。选择本层历史最低 FPR 对应的模型作为本层成果，结束本层并准备进入下一层。")
                    best_idx = np.argmin([h['fpr'] for h in stage_history])
                    best_model = stage_history[best_idx]['model']
                    cascade_fpr = stage_history[best_idx]['fpr']
                    cascade_tpr = stage_history[best_idx]['tpr']
                    break  # 跳出本层 while 循环，进入外层逻辑（HNM与下一层级）

                elif action == '3':
                    print("  [C] 确认直接结束整个级联。正在保存当前所有进度...")
                    best_idx = np.argmin([h['fpr'] for h in stage_history])
                    best_model = stage_history[best_idx]['model']
                    self.stages.append(best_model)
                    overall_fpr = stage_history[best_idx]['fpr']
                    self.overall_tpr = stage_history[best_idx]['tpr']
                    self._save_checkpoint(layer_idx, overall_fpr)
                    print(f"\n[Cascade] 级联训练已安全提前结束，当前共有级联层数: {len(self.stages)}")
                    return self.stages

            # 本层循环结束，保存确定的最合适模型
            self.stages.append(best_model)
            overall_fpr      = cascade_fpr
            self.overall_tpr = cascade_tpr

            print(f"\n[Cascade] 第 {layer_idx} 层组装完成！"
                  f"包含 {len(best_model.weak_classifiers)} 个特征，"
                  f"本层最终判定阈值={best_model.threshold:.3f}")
            print(f"[Cascade] 整体累计 FPR={overall_fpr:.4e}  目标={self.F_target:.2e}")
            print(f"[Cascade] 整体累计 TPR={self.overall_tpr*100:.2f}%")

            if overall_fpr <= self.F_target:
                print(f"\n[Cascade] [OK] 整体已完成，FPR {self.F_target:.2e} 目标已达成。")
                self._save_checkpoint(layer_idx, overall_fpr)
                break

            print(f"\n[Cascade] 开始挖掘第 {layer_idx+1} 层的 Hard Negative 困难样本...")
            self.X_neg = self._mine_hard_negatives(max_count=2400)
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"[Cascade] HNM 挖掘完成，收集到下阶段负样本共 {len(self.X_neg)} 个")

            self._save_checkpoint(layer_idx, overall_fpr)

        print(f"\n[Cascade] 级联构建结束，当前共有级联层数: {len(self.stages)}")
        return self.stages

    def _cascade_predict(self, patch_norm: np.ndarray) -> bool:
        """
        利用当前已有层判定单个 patch 是否会被判为人脸。
        """
        var = np.var(patch_norm)
        if var < 1e-4:
            return False

        ii_obj = build(patch_norm)
        sigma  = np.sqrt(var)

        for stage in self.stages:
            score = 0.0
            for wc in stage.weak_classifiers:
                feat_desc = (self.features_desc[wc.feature_idx]
                             if isinstance(wc.feature_idx, (int, np.integer))
                             else wc.feature_idx)
                raw_val  = compute_feature_at_scale(
                    desc=feat_desc, ii=ii_obj.ii,
                    scale=1.0, win_r=0, win_c=0,
                )
                norm_val = raw_val / sigma
                if wc.polarity * norm_val < wc.polarity * wc.threshold:
                    score += wc.alpha
            if score < stage.threshold:
                return False
        return True

    def _mine_hard_negatives(self, max_count: int = 2400) -> np.ndarray:
        """
        通过滑动窗口扫描负样本图像，收集被错误判定为人脸的子窗口。
        """
        if not self.train_neg_image_dir or not os.path.exists(self.train_neg_image_dir):
            print("  [HNM] [WARN] 负样本大图目录不正确，转入降级过滤...")
            return self._fallback_filter_negatives()

        print(f"  [HNM] 扫描中，滑动步长={self.hnm_step}，收集目标上限={max_count}...")
        t0 = time.time()

        patches = _collect_patches_from_dir(
            image_dir=self.train_neg_image_dir,
            n_samples=max_count,
            win_size=24,
            step=self.hnm_step,
            mode='scan',
            cascade_classifier=self._cascade_predict,
        )

        print(f"  [HNM] 扫描进程耗时 {time.time()-t0:.1f}s，"
              f"收集困难样本: {len(patches)}/{max_count} 个")

        if len(patches) == 0:
            print("  [HNM] [WARN] 未找到困难负样本，转入降级过滤模式...")
            return self._fallback_filter_negatives()

        print(f"  [HNM] 批量转换并提取特征（特征规模：{len(self.features_desc)}）...")
        X_hnm = _patches_to_feature_matrix(patches, self.features_desc)
        print(f"  [HNM] 提取完成，特征矩阵形状为: {X_hnm.shape}")
        return X_hnm

    def _fallback_filter_negatives(self) -> np.ndarray:
        """
        备用兜底模式：无法直接挖掘新 Hard Negative 时，在上一阶段的负样本数据中继续寻找困难负样本。
        """
        print("  [Fallback] 开始级联过滤存量样本...")
        hard_neg = self.X_neg.copy()

        for stage in self.stages:
            if len(hard_neg) == 0:
                break
            preds    = stage.classify(hard_neg)
            hard_neg = hard_neg[preds == 1]
            print(f"    过滤后剩余困难负样本数量: {len(hard_neg)}")

        if len(hard_neg) < 200:
            n_sup = min(500 - len(hard_neg), len(self.X_neg))
            idx   = np.random.choice(len(self.X_neg), n_sup, replace=False)
            hard_neg = (np.vstack([hard_neg, self.X_neg[idx]])
                        if len(hard_neg) > 0 else self.X_neg[idx])
            print(f"  [Fallback] 补充 {n_sup} 个常规负样本，当前负样本大小: {len(hard_neg)}")

        return hard_neg
