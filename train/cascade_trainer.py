"""
cascade_trainer.py
==================
级联分类器训练与 Hard Negative Mining (HNM) 模块

核心设计：
  训练集正样本：全程固定不变（特征矩阵由 data_loader 预处理好后传入）
  训练集负样本：
    - 第一层：由 train_cascade.py 从训练大图随机裁取10000个子窗口传入
    - 后续层：本模块执行 HNM，扫描训练大图，收集误检子窗口（最多6000个）
  验证集正样本：全程固定不变（特征矩阵预计算好直接查表）
  验证集负样本：每轮评估前从验证大图中重新随机采样1000个子窗口
                （每轮重采样避免固定采样的偶然偏差）

  阈值调整策略（改动说明）：
    旧版：先在训练集正样本上粗调阈值（adjust_threshold_for_detection_rate），
          再在验证集上微调。
    新版：去掉粗调步骤，AdaBoost 训练完后直接在验证集上评估，
          若累积 TPR 不足则直接降低阈值，所有调整都以验证集为准。
    原因：粗调是在训练集上操作，与验证集评估标准不一致，且多了一次额外的扫描；
          直接在验证集上调整更简洁、更准确。

  所有负样本大图均为彩色图：
    读入后立即整张转灰度 → 裁子窗口 → 方差归一化
    （先转灰度再裁，等价于先裁再转，但整张转一次效率更高）
"""

import os
import time
import pickle
import numpy as np
from typing import List, Tuple
import cv2

from integral_image import build, build_batch
from adaboost import train_adaboost, StrongClassifier
from haar_features import compute_all_features, compute_feature_at_scale


# ─────────────────────────────────────────────────────────────
#  模块级辅助函数
# ─────────────────────────────────────────────────────────────

def _variance_normalize_patch(patch: np.ndarray):
    """
    对单个 24×24 灰度 patch 做均值/方差归一化（论文 Section 5.4）。

    论文公式：σ² = E[x²] − (E[x])²，归一化：x_norm = (x − mean) / σ

    返回：
        归一化后的 float32 数组；若方差 < 1（纯色块）则返回 None 表示跳过。
    """
    mean = np.mean(patch, dtype=np.float64)
    var  = np.mean(patch.astype(np.float64) ** 2) - mean ** 2
    if var < 1.0:
        return None          # 方差过小，纯色块，无判别价值
    sigma = np.sqrt(var)
    return (patch.astype(np.float32) - mean) / sigma


def _collect_patches_from_dir(
    image_dir: str,
    n_samples: int,
    win_size: int = 24,
    step: int = 4,
    mode: str = 'random',
    cascade_classifier=None,
) -> List[np.ndarray]:
    """
    从 image_dir 下的彩色大图中采集归一化后的 24×24 灰度 patch。

    参数：
        image_dir          : 彩色大图所在目录（jpg/png/jpeg/bmp）
        n_samples          : 目标采集数量（mode='scan' 时满足即停，mode='random' 时尽量凑满）
        win_size           : 子窗口大小，固定 24
        step               : 滑动步长，mode='scan' 时生效（推荐 4）
        mode               : 'random' — 随机位置采样（第一层负样本、验证集负样本）
                             'scan'   — 滑动扫描+级联过滤（HNM）
        cascade_classifier : mode='scan' 时必须传入；接受归一化 patch，返回 bool

    处理流程（适用两种 mode）：
        1. 读彩色大图（IMREAD_COLOR）
        2. 整张转灰度（cvtColor BGR2GRAY）—— 只转一次，效率最高
        3. 裁取 24×24 子窗口
        4. 方差归一化（σ² < 1 则跳过）
        5. mode='scan' 时额外送入当前级联判断，通过才收录

    返回：已做灰度转换 + 方差归一化的 float32 数组列表，每个形状 (24, 24)
    """
    exts  = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(exts)
    ]
    if not files:
        print(f"  [采样] ⚠ 目录 {image_dir} 中未找到图片！")
        return []

    np.random.shuffle(files)   # 打乱，避免每次都读相同的图
    patches = []

    for img_path in files:
        if len(patches) >= n_samples:
            break

        # 读彩色图
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [采样] 无法读取: {img_path}，跳过。")
            continue

        # 整张图转灰度（一次转换，效率最高）
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = img_gray.shape
        if h < win_size or w < win_size:
            continue

        if mode == 'random':
            # ── 随机采样（第一层负样本、验证集负样本）──────
            # 每张图最多采 max_per_img 个，防止全部样本来自同一张图
            max_per_img   = max(1, n_samples // max(1, len(files) // 2))
            count_this    = 0
            rows          = np.arange(0, h - win_size + 1)
            cols          = np.arange(0, w - win_size + 1)
            positions     = [(r, c) for r in rows for c in cols]
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
            # ── 滑动窗口扫描（HNM）─────────────────────────
            # 步长=step，通过当前级联（误判为人脸）才收录
            # 收够 n_samples 立即停止，不需要扫完所有图
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
    流程：构建积分图 → 批量计算约 16 万个特征
    返回：形状 (N, D) 的 float32 特征矩阵
    """
    if len(patches) == 0:
        return np.empty((0, len(features_desc)), dtype=np.float32)
    iimgs = build_batch(patches)
    return compute_all_features(iimgs, features_desc, scale=1.0).astype(np.float32)


def _sample_val_neg_features(
    val_neg_image_dir: str,
    n_samples: int,
    features_desc: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    每轮评估前从验证大图目录随机采样 n_samples 个子窗口并计算特征。

    每轮重新采样的原因：避免固定采样的偶然偏差，1000 个样本统计上足以稳定估计 FPR。

    返回：(X_val_neg, y_val_neg)，形状分别为 (N, D) 和 (N,)
    """
    patches = _collect_patches_from_dir(
        image_dir=val_neg_image_dir,
        n_samples=n_samples,
        win_size=24,
        mode='random',
    )
    if len(patches) == 0:
        return np.empty((0, len(features_desc)), dtype=np.float32), np.array([], dtype=np.int32)
    X = _patches_to_feature_matrix(patches, features_desc)
    y = np.zeros(len(X), dtype=np.int32)
    return X, y


def _next_n_features(n: int, max_n: int) -> int:
    """
    分段加速策略：决定下一轮尝试的特征数。
      n < 10  : 翻倍
      10≤n<30 : +50%
      30≤n<100: +25（论文后期策略）
      n≥100   : +50
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
    按照 Viola-Jones 论文 Table 2 训练级联分类器。

    每轮循环：
      1. 从验证大图随机采样本轮验证负样本（1000个）
      2. AdaBoost 训练强分类器
      3. 直接在验证集上评估累积 TPR/FPR，若 TPR 不足则降低阈值
         （去掉旧版的训练集粗调步骤，所有阈值调整均以验证集为准）
      4. 若 FPR 仍未达标，增加特征数重训（分段加速）
      5. 本层达标后执行 HNM，收集训练大图误检作为下一层负样本
    """

    def __init__(
        self,
        X_pos_train: np.ndarray,        # 训练正样本特征矩阵 (N_pos, D)，全程不变
        y_pos_train: np.ndarray,        # 全 1
        X_neg_train: np.ndarray,        # 第一层训练负样本特征矩阵 (N_neg, D)
        y_neg_train: np.ndarray,        # 全 0
        X_val_pos: np.ndarray,          # 验证正样本特征矩阵 (N_val_pos, D)，全程不变
        y_val_pos: np.ndarray,          # 全 1，全程不变
        val_neg_image_dir: str,         # 验证集负样本大图目录（每轮重新采样）
        train_neg_image_dir: str,       # 训练集负样本大图目录（HNM 扫描）
        features_desc: list,            # Haar 特征描述符列表
        target_fpr: float = 1e-5,       # 目标整体 FPR
        layer_max_fpr: float = 0.50,    # 单层最大 FPR
        layer_min_dr: float = 0.99,     # 单层最低 DR
        max_features_per_stage: int = 200,
        val_neg_per_round: int = 1000,  # 每轮验证负样本数
        hnm_step: int = 4,              # HNM 滑动步长
        checkpoint_path: str = "../models/cascade_checkpoint.pkl",
        resume_state: dict = None,
    ):
        # 固定训练正样本
        self.X_pos = X_pos_train
        self.y_pos = y_pos_train

        # 固定验证正样本
        self.X_val_pos = X_val_pos
        self.y_val_pos = y_val_pos

        # 目录路径
        self.val_neg_image_dir   = val_neg_image_dir
        self.train_neg_image_dir = train_neg_image_dir
        self.features_desc       = features_desc

        # 超参数
        self.F_target               = target_fpr
        self.f_target               = layer_max_fpr
        self.d_target               = layer_min_dr
        self.max_features_per_stage = max_features_per_stage
        self.val_neg_per_round      = val_neg_per_round
        self.hnm_step               = hnm_step
        self.checkpoint_path        = checkpoint_path

        # 断点恢复 / 初始化
        if resume_state is not None:
            print("[Checkpoint] 检测到断点，正在恢复...")
            self.stages          = resume_state['stages']
            self.start_layer_idx = resume_state['layer_idx']
            self.overall_fpr     = resume_state['overall_fpr']
            self.overall_tpr     = resume_state.get('overall_tpr', 1.0)
            self.X_neg           = resume_state['X_neg']
            self.y_neg           = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"  -> 从第 {self.start_layer_idx + 1} 层继续。"
                  f"当前整体 FPR={self.overall_fpr:.2e}，TPR={self.overall_tpr:.4f}")
        else:
            self.stages: List[StrongClassifier] = []
            self.start_layer_idx = 0
            self.overall_fpr     = 1.0
            self.overall_tpr     = 1.0
            self.X_neg           = X_neg_train   # 第一层使用外部传入的随机裁取负样本
            self.y_neg           = y_neg_train

    # ─────────────────────────────────────────────────────────
    #  验证集评估
    # ─────────────────────────────────────────────────────────

    def _evaluate_cascade(
        self,
        current_stage: StrongClassifier,
        X_val_neg: np.ndarray,
        y_val_neg: np.ndarray,
    ) -> Tuple[float, float]:
        """
        评估「已有层 + 本层」在验证集上的累积 TPR 和 FPR。

        验证集 = 固定验证正样本 + 本轮随机采样的验证负样本。

        逻辑：
          - 所有样本初始视为通过（pred=1）
          - 依次经过每一层：被拒绝则 pred=0，后续不再处理
          - 最终统计 TP/FP，计算 TPR/FPR
        """
        X_val = np.vstack([self.X_val_pos, X_val_neg])
        y_val = np.hstack([self.y_val_pos, y_val_neg])

        all_stages = self.stages + [current_stage]
        preds      = np.ones(len(X_val), dtype=np.int32)

        for stage in all_stages:
            still = np.where(preds == 1)[0]
            if len(still) == 0:
                break
            stage_preds      = stage.classify(X_val[still])
            preds[still[stage_preds == 0]] = 0

        pos_mask = (y_val == 1)
        neg_mask = (y_val == 0)
        n_pos    = np.sum(pos_mask)
        n_neg    = np.sum(neg_mask)

        tp  = np.sum((preds == 1) & pos_mask)
        fp  = np.sum((preds == 1) & neg_mask)

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0

        print(f"    [验证集] 累积 TPR={tpr*100:.2f}% ({tp}/{n_pos})，"
              f"FPR={fpr*100:.4f}% ({fp}/{n_neg})")
        return tpr, fpr

    # ─────────────────────────────────────────────────────────
    #  断点存档
    # ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, layer_idx: int, overall_fpr: float) -> None:
        """序列化当前训练进度（含 HNM 结果），支持中断后继续。"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        state = {
            'stages':      self.stages,
            'layer_idx':   layer_idx,
            'overall_fpr': overall_fpr,
            'overall_tpr': self.overall_tpr,
            'X_neg':       self.X_neg,   # 本轮 HNM 结果，恢复后直接用于下一层
        }
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Checkpoint] 第 {layer_idx} 层进度已保存: {self.checkpoint_path}")

    # ─────────────────────────────────────────────────────────
    #  主训练循环
    # ─────────────────────────────────────────────────────────

    def train(self) -> List[StrongClassifier]:
        """
        级联训练主循环（论文 Table 2）。

        关键改动：
          去掉旧版 D2 步骤（在训练集正样本上粗调阈值）。
          训练完 AdaBoost 后，直接在验证集上评估，若 TPR 不足则在验证集上调整阈值。
          所有阈值调整以验证集为唯一标准，与训练集解耦。
        """
        overall_fpr = self.overall_fpr
        layer_idx   = self.start_layer_idx

        print(f"\n{'='*60}")
        print(f"[Cascade] 开始训练")
        print(f"  目标整体 FPR    : {self.F_target:.2e}")
        print(f"  单层最大 FPR    : {self.f_target:.2f}")
        print(f"  单层最低 DR     : {self.d_target:.4f}")
        print(f"  单层特征上限    : {self.max_features_per_stage}")
        print(f"  每轮验证负样本  : {self.val_neg_per_round} 个（每轮重采样）")
        print(f"  HNM 步长        : {self.hnm_step}")
        print(f"{'='*60}")

        while overall_fpr > self.F_target:
            layer_idx += 1
            print(f"\n[Cascade] ========== 第 {layer_idx} 层 ==========")

            # ── A：采样本轮验证集负样本 ──────────────────────
            # 每层重新从验证大图随机采样，保证评估的独立性和随机性
            print(f"  [A] 采样本轮验证集负样本（{self.val_neg_per_round} 个）...")
            t_val = time.time()
            X_val_neg, y_val_neg = _sample_val_neg_features(
                val_neg_image_dir=self.val_neg_image_dir,
                n_samples=self.val_neg_per_round,
                features_desc=self.features_desc,
            )
            print(f"  [A] 采样完成，耗时 {time.time()-t_val:.1f}s，"
                  f"得到 {len(X_val_neg)} 个验证负样本特征向量。")

            if len(X_val_neg) == 0:
                print("  [Cascade] ⚠ 验证集负样本采样失败，终止训练！")
                break

            # ── B：确定本层起始特征数 ─────────────────────────
            # 第1层从2开始；后续层从前一层特征数的一半开始（跳过无效小值区间）
            if self.stages:
                prev_n     = len(self.stages[-1].weak_classifiers)
                n_features = max(2, prev_n // 2)
                print(f"  [B] 前一层 {prev_n} 个特征，本层起点 {n_features} 个。")
            else:
                n_features = 2
                print(f"  [B] 第一层，从 {n_features} 个特征开始。")

            # ── C：组合本层训练集 ─────────────────────────────
            # 正样本全程固定；负样本：第1层为随机裁取的10000个，后续层为上轮 HNM 结果
            X_train = np.vstack([self.X_pos, self.X_neg])
            y_train = np.hstack([self.y_pos, self.y_neg])
            print(f"  [C] 训练集：正 {len(self.X_pos)} + 负 {len(self.X_neg)} "
                  f"= {len(X_train)} 个样本")

            # 本层需要满足的累积指标目标
            target_tpr = self.d_target * self.overall_tpr   # 累积 TPR 下限
            target_fpr = self.f_target * overall_fpr         # 累积 FPR 上限
            print(f"  [C] 目标：累积 TPR ≥ {target_tpr*100:.2f}%，"
                  f"累积 FPR ≤ {target_fpr*100:.4f}%")

            best_model  = None
            cascade_tpr = 0.0
            cascade_fpr = 1.0

            # ── D：内层循环，逐步增加特征数直到 FPR 达标 ────
            while True:
                print(f"\n  [D] 训练 {n_features} 个弱分类器...")
                t0 = time.time()

                # D1：AdaBoost 训练强分类器
                stage_model = train_adaboost(
                    X_train, y_train,
                    n_features_to_select=n_features,
                    verbose=False,
                )
                print(f"    AdaBoost 完成，耗时 {time.time()-t0:.1f}s")

                # D2：【改动】直接在验证集上评估，不再先做训练集粗调
                # 旧版：先调用 adjust_threshold_for_detection_rate 在训练集正样本上粗调，
                #       再在验证集微调。两步操作标准不一致，且多一次训练集扫描。
                # 新版：AdaBoost 训练完即用默认阈值，直接在验证集上评估累积 TPR/FPR。
                cascade_tpr, cascade_fpr = self._evaluate_cascade(
                    stage_model, X_val_neg, y_val_neg
                )

                # D3：若验证集累积 TPR 不足，持续降低阈值（以更多假正换取更高召回）
                # 所有阈值调整只看验证集指标，与训练集完全解耦
                adjust_count = 0
                while cascade_tpr < target_tpr and stage_model.threshold > -100.0:
                    stage_model.threshold -= 0.05
                    cascade_tpr, cascade_fpr = self._evaluate_cascade(
                        stage_model, X_val_neg, y_val_neg
                    )
                    adjust_count += 1
                    if adjust_count % 20 == 0:
                        print(f"    [阈值调整 ×{adjust_count}] "
                              f"threshold={stage_model.threshold:.3f}，"
                              f"TPR={cascade_tpr*100:.2f}% (目标≥{target_tpr*100:.2f}%)")

                if adjust_count > 0:
                    print(f"    [阈值调整] 共调整 {adjust_count} 次，"
                          f"最终 threshold={stage_model.threshold:.3f}")

                print(f"  [D] {n_features} 个特征 → "
                      f"累积 TPR={cascade_tpr*100:.2f}% (目标≥{target_tpr*100:.2f}%)，"
                      f"累积 FPR={cascade_fpr*100:.4f}% (目标≤{target_fpr*100:.4f}%)")

                best_model = stage_model

                # D4：判断 FPR 是否达标
                if cascade_fpr <= target_fpr:
                    print(f"  [D] ✓ FPR 达标，本层训练完成。")
                    break

                # D5：FPR 未达标，分段加速增加特征数
                n_next = _next_n_features(n_features, self.max_features_per_stage)
                if n_next >= self.max_features_per_stage:
                    print(f"  [D] ⚠ 特征数达上限 {self.max_features_per_stage}，"
                          f"强制结束本层（FPR={cascade_fpr*100:.4f}%）。")
                    break

                print(f"  [D] FPR 未达标，特征数 {n_features} → {n_next}（+{n_next-n_features}）")
                n_features = n_next

            # ── E：本层完成，更新累积状态 ────────────────────
            self.stages.append(best_model)
            overall_fpr      = cascade_fpr
            self.overall_tpr = cascade_tpr

            print(f"\n[Cascade] 第 {layer_idx} 层完成！"
                  f"{len(best_model.weak_classifiers)} 个特征，"
                  f"阈值={best_model.threshold:.3f}")
            print(f"[Cascade] 整体 FPR={overall_fpr:.4e}  目标={self.F_target:.2e}")
            print(f"[Cascade] 整体 TPR={self.overall_tpr*100:.2f}%")

            # ── F：是否达到目标 ───────────────────────────────
            if overall_fpr <= self.F_target:
                print(f"\n[Cascade] ✓ 达到目标 FPR {self.F_target:.2e}，训练结束。")
                self._save_checkpoint(layer_idx, overall_fpr)
                break

            # ── G：Hard Negative Mining ───────────────────────
            # 用当前级联扫描训练大图，收集误检子窗口作为下一层负样本（最多6000个）
            print(f"\n[Cascade] HNM：为第 {layer_idx+1} 层准备负样本...")
            self.X_neg = self._mine_hard_negatives(max_count=6000)
            self.y_neg = np.zeros(len(self.X_neg), dtype=np.int32)
            print(f"[Cascade] HNM 完成，下一层负样本数: {len(self.X_neg)}")

            self._save_checkpoint(layer_idx, overall_fpr)

        print(f"\n[Cascade] 训练结束，共 {len(self.stages)} 层。")
        return self.stages

    # ─────────────────────────────────────────────────────────
    #  级联推理（用于 HNM 判断单个 patch）
    # ─────────────────────────────────────────────────────────

    def _cascade_predict(self, patch_norm: np.ndarray) -> bool:
        """
        用当前所有已训练层判断归一化后的 24×24 patch 是否被误判为人脸。

        patch_norm 已经是灰度归一化数据，此处只做积分图构建和特征计算。
        通过所有层 → True（Hard Negative）；被任意层拒绝 → False。
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
                return False   # 被本层拒绝
        return True            # 通过所有层，是 Hard Negative

    # ─────────────────────────────────────────────────────────
    #  Hard Negative Mining
    # ─────────────────────────────────────────────────────────

    def _mine_hard_negatives(self, max_count: int = 6000) -> np.ndarray:
        """
        扫描训练负样本大图，收集被当前级联误判为人脸的子窗口（Hard Negative）。

        策略：
          - 打乱图像列表顺序
          - 单尺度固定 24×24 滑动窗口，步长 = self.hnm_step（推荐 4）
          - 大图先整张转灰度再裁（_collect_patches_from_dir 内完成）
          - 方差 < 1 的 patch 跳过
          - 收够 max_count 个立即停止，不需要扫完所有图（论文上限 6000）

        返回：形状 (N, D) 的特征矩阵，N ≤ max_count
        """
        if not self.train_neg_image_dir or not os.path.exists(self.train_neg_image_dir):
            print("  [HNM] ⚠ 训练负样本目录不存在，降级为特征过滤模式...")
            return self._fallback_filter_negatives()

        print(f"  [HNM] 扫描训练大图，步长={self.hnm_step}，目标={max_count} 个...")
        t0 = time.time()

        patches = _collect_patches_from_dir(
            image_dir=self.train_neg_image_dir,
            n_samples=max_count,
            win_size=24,
            step=self.hnm_step,
            mode='scan',
            cascade_classifier=self._cascade_predict,
        )

        print(f"  [HNM] 扫描完成，耗时 {time.time()-t0:.1f}s，"
              f"收集 {len(patches)}/{max_count} 个 Hard Negative。")

        if len(patches) == 0:
            print("  [HNM] ⚠ 未找到 Hard Negative，降级为特征过滤模式...")
            return self._fallback_filter_negatives()

        print(f"  [HNM] 批量计算 {len(patches)} 个 patch 的 Haar 特征...")
        X_hnm = _patches_to_feature_matrix(patches, self.features_desc)
        print(f"  [HNM] 特征矩阵形状: {X_hnm.shape}")
        return X_hnm

    def _fallback_filter_negatives(self) -> np.ndarray:
        """
        备用策略：HNM 无法执行时，从现有负样本特征矩阵中过滤出仍能通过级联的困难样本。
        若过滤后数量 < 200，随机补充原始负样本防止训练崩溃。
        """
        print("  [Fallback] 对现有负样本特征做级联过滤...")
        hard_neg = self.X_neg.copy()

        for stage in self.stages:
            if len(hard_neg) == 0:
                break
            preds    = stage.classify(hard_neg)
            hard_neg = hard_neg[preds == 1]
            print(f"    过滤后剩余: {len(hard_neg)}")

        if len(hard_neg) < 200:
            n_sup = min(500 - len(hard_neg), len(self.X_neg))
            idx   = np.random.choice(len(self.X_neg), n_sup, replace=False)
            hard_neg = (np.vstack([hard_neg, self.X_neg[idx]])
                        if len(hard_neg) > 0 else self.X_neg[idx])
            print(f"  [Fallback] 补充 {n_sup} 个随机负样本，最终: {len(hard_neg)}")

        return hard_neg