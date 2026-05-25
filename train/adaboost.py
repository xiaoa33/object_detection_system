"""
adaboost.py — AdaBoost 弱分类器训练模块
=========================================
严格按照论文 Table 1 实现单层强分类器的 AdaBoost 训练流程。

参考文献：
    Viola P, Jones M J. Robust Real-Time Face Detection[J].
    International Journal of Computer Vision, 2004, 57(2): 137-154.

论文 Table 1 伪代码还原（逐步对应）：
    - 初始化权重：正样本各 1/(2l)，负样本各 1/(2m)
    - 迭代 T 轮：
        1. 归一化权重（使其和为 1）
        2. 遍历所有特征，对每个特征按特征值排序后单次线性扫描，
           维护 T+, T-, S+, S- 四个累积量，计算最优阈值及加权误差
        3. 选误差最小的特征作为本轮弱分类器 h_t
        4. 计算 β_t = ε_t / (1 - ε_t)，α_t = log(1/β_t)
        5. 更新样本权重：分类正确的乘以 β_t，分类错误的不变
    - 输出强分类器：C(x) = 1 若 Σ α_t·h_t(x) ≥ ½·Σ α_t，否则为 0
"""

import numpy as np
import time
import pickle
import multiprocessing  # 🚀 导入多进程模块
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构定义
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WeakClassifier:
    """
    弱分类器（决策桩，Decision Stump）。

    对应论文 Section 3 中的定义：
        h(x, f, p, θ) = 1  若 p·f(x) < p·θ
                       = 0  否则

    属性：
        feature_idx  : int   — 特征在特征矩阵中的列索引（对应约16万特征之一）
        threshold    : float — 判别阈值 θ
        polarity     : int   — 极性 p ∈ {+1, -1}
        alpha        : float — 该弱分类器在强分类器中的权重 α_t = log(1/β_t)
        error        : float — 本轮加权误差 ε_t（训练时记录，便于调试）
    """
    feature_idx: int
    threshold: float
    polarity: int        # +1 或 -1
    alpha: float = 0.0
    error: float = 0.0


@dataclass
class StrongClassifier:
    """
    强分类器，由 T 个弱分类器加权组成。

    判决规则（论文 Table 1 最后一行）：
        C(x) = 1  若 Σ_{t=1}^{T} α_t · h_t(x) ≥ ½ · Σ_{t=1}^{T} α_t
               = 0  否则

    属性：
        weak_classifiers : list[WeakClassifier] — 所有弱分类器
        threshold        : float                — 判决阈值，默认 ½·sum_alpha_t，
                                                  可在级联训练时调低以提升检测率
    """
    weak_classifiers: List[WeakClassifier] = field(default_factory=list)
    threshold: float = 0.0  # 初始化后由 _update_threshold() 设置

    def _update_threshold(self) -> None:
        """将阈值重置为 ½·sum_alpha_t（论文默认值）。"""
        self.threshold = 0.5 * sum(wc.alpha for wc in self.weak_classifiers)

    def classify(self, feature_values: np.ndarray) -> np.ndarray:
        """
        对一批样本做强分类器判决。

        参数：
            feature_values : ndarray, shape (n_samples, n_features)
                             每行是一个样本的全部特征值。
        返回：
            predictions : ndarray, shape (n_samples,), dtype int
                          1 = 正例（人脸），0 = 负例（非人脸）
        """
        # 累积所有弱分类器的加权输出
        score = np.zeros(feature_values.shape[0], dtype=np.float64)
        for wc in self.weak_classifiers:
            f_vals = feature_values[:, wc.feature_idx]
            # h(x) = 1 若 p·f(x) < p·θ，否则为 0
            h = (wc.polarity * f_vals < wc.polarity * wc.threshold).astype(np.float64)
            score += wc.alpha * h
        # C(x) = 1 若得分 ≥ 阈值
        return (score >= self.threshold).astype(np.int32)

    def score(self, feature_values: np.ndarray) -> np.ndarray:
        """
        返回连续得分（未二值化），供级联训练中调整阈值使用。

        参数：
            feature_values : ndarray, shape (n_samples, n_features)
        返回：
            scores : ndarray, shape (n_samples,), dtype float64
        """
        s = np.zeros(feature_values.shape[0], dtype=np.float64)
        for wc in self.weak_classifiers:
            f_vals = feature_values[:, wc.feature_idx]
            h = (wc.polarity * f_vals < wc.polarity * wc.threshold).astype(np.float64)
            s += wc.alpha * h
        return s


# ──────────────────────────────────────────────────────────────────────────────
# 核心函数：单特征最优阈值求解（论文 Section 3.1）
# ──────────────────────────────────────────────────────────────────────────────

def _find_best_threshold_for_feature(
    feature_values: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, int, float]:
    """
    【向量化加速版】对单个特征找到最优阈值和极性。
    数学逻辑与论文完全一致，利用 np.cumsum 消除 9000 次的 Python 慢速 for 循环。
    """
    n = len(feature_values)

    # 1. 按特征值排序
    sorted_indices = np.argsort(feature_values)
    sorted_vals   = feature_values[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # 2. 分离正负样本的权重矩阵
    w_pos = sorted_weights * (sorted_labels == 1)
    w_neg = sorted_weights * (sorted_labels == 0)

    # 3. 核心加速：使用 np.cumsum 一次性求出所有位置的累加和 (替代原先的 for 循环)
    # S_pos/S_neg 数组的第 i 个元素，即为原代码循环到第 i 次时的 S_pos/S_neg 值
    S_pos_arr = np.cumsum(w_pos)
    S_neg_arr = np.cumsum(w_neg)

    # 全局正负权重之和
    T_pos = S_pos_arr[-1]
    T_neg = S_neg_arr[-1]

    # 4. 向量化计算所有可能分割点的两种误差
    # e1: 阈值以下标为负 (p = -1) 的误差
    e1_arr = S_pos_arr + (T_neg - S_neg_arr)
    # e2: 阈值以下标为正 (p = +1) 的误差
    e2_arr = S_neg_arr + (T_pos - S_pos_arr)

    # 5. 找出每个位置的最小误差
    errors = np.minimum(e1_arr, e2_arr)

    # 6. 找到全局最小误差的索引
    best_idx = np.argmin(errors)
    
    min_error = float(errors[best_idx])
    best_polarity = -1 if e1_arr[best_idx] < e2_arr[best_idx] else 1

    # 7. 计算阈值（原逻辑：当前值与下一值的中点）
    if best_idx + 1 < n:
        best_threshold = 0.5 * (sorted_vals[best_idx] + sorted_vals[best_idx + 1])
    else:
        best_threshold = sorted_vals[best_idx] + 0.5

    return float(best_threshold), int(best_polarity), min_error


# ──────────────────────────────────────────────────────────────────────────────
# 🚀 多进程共享内存设计：利用全局变量与 Linux 写时复制实现无拷贝共享
# ──────────────────────────────────────────────────────────────────────────────

_global_X = None
_global_y = None

def _init_mp_pool(X: np.ndarray, y: np.ndarray):
    """多进程初始化函数：子进程继承父进程的 X 和 y，零复制共享特征数据"""
    global _global_X, _global_y
    _global_X = X
    _global_y = y

def _eval_feature_worker(args: Tuple[int, np.ndarray]) -> Tuple[int, float, int, float]:
    """并行计算单特征最优阈值的 worker"""
    f_idx, weights = args
    # 直接在内存指针上切片提取对应特征列，避免跨进程传输大数据
    x_col = _global_X[:, f_idx]
    threshold, polarity, error = _find_best_threshold_for_feature(
        x_col, _global_y, weights
    )
    return f_idx, threshold, polarity, error


# ──────────────────────────────────────────────────────────────────────────────
# 核心函数：AdaBoost 训练（论文 Table 1）
# ──────────────────────────────────────────────────────────────────────────────

def train_adaboost(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int,
    verbose: bool = True,
) -> StrongClassifier:
    """
    训练单层强分类器（AdaBoost，论文 Table 1 完整流程）。

    参数：
        X : ndarray, shape (n_samples, n_features)
            所有样本的特征矩阵（行 = 样本，列 = 特征）。
            对应论文约 160,000 列特征。
        y : ndarray, shape (n_samples,), dtype int
            标签，1 = 正样本（人脸），0 = 负样本（非人脸）。
        n_features_to_select : int
            本层强分类器选择的弱分类器数量 T。
        verbose : bool
            是否打印每轮进度。

    返回：
        StrongClassifier — 训练好的强分类器（含所有弱分类器和默认阈值）
    """
    n_samples, n_features = X.shape
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    print(f"\n{'='*60}")
    print(f"[AdaBoost] 开始训练强分类器")
    print(f"  样本总数   : {n_samples}（正 {n_pos} / 负 {n_neg}）")
    print(f"  特征总数   : {n_features}")
    print(f"  目标弱分类器数 T = {n_features_to_select}")
    print(f"{'='*60}")

    # ── 论文 Table 1, 第一行 ──────────────────────────────────────────────────
    # 初始化权重：正样本各 1/(2l)，负样本各 1/(2m)
    # l = 正样本数，m = 负样本数
    weights = np.zeros(n_samples, dtype=np.float64)
    weights[y == 1] = 1.0 / (2.0 * n_pos)
    weights[y == 0] = 1.0 / (2.0 * n_neg)
    print(f"[AdaBoost] 权重初始化完成：正样本各 {1.0/(2*n_pos):.6f}，负样本各 {1.0/(2*n_neg):.6f}")

    # 🚀 初始化进程池：保留2个空闲核心，防止机器卡死，使用 fork 初始化进程池
    num_cores = max(1, multiprocessing.cpu_count() - 2)
    print(f"[AdaBoost] 正在初始化并行进程池，核心数: {num_cores} ...")
    pool = multiprocessing.Pool(
        processes=num_cores, 
        initializer=_init_mp_pool, 
        initargs=(X, y)
    )

    strong_clf = StrongClassifier()

    try:
        # ── 论文 Table 1, For t = 1,...,T ────────────────────────────────────────
        for t in range(1, n_features_to_select + 1):
            round_start = time.time()

            # ── Step 1：归一化权重 ────────────────────────────────────────────────
            # w_{t,i} ← w_{t,i} / Σ_j w_{t,j}
            weight_sum = weights.sum()
            weights /= weight_sum

            # ── Step 2：多进程并行扫描所有特征 ───────────────────
            # 准备参数元组 (特征索引, 当前轮权重)，传递少量数据 w
            tasks = [(f_idx, weights) for f_idx in range(n_features)]
            
            # 使用进程池计算并收集结果
            results = pool.map(_eval_feature_worker, tasks, chunksize=100)

            # 寻找加权误差最小的弱分类器
            best_error    = np.inf
            best_feat_idx = -1
            best_threshold = 0.0
            best_polarity  = 1

            for f_idx, threshold, polarity, error in results:
                if error < best_error:
                    best_error     = error
                    best_feat_idx  = f_idx
                    best_threshold = threshold
                    best_polarity  = polarity

            # ── Step 3：定义本轮弱分类器 h_t ─────────────────────────────────────
            # h_t(x) = h(x, f_t, p_t, θ_t)，其中 f_t, p_t, θ_t 为上面的最优值

            # ── Step 4：计算 α_t 和 β_t ──────────────────────────────────────────
            # 论文：β_t = ε_t / (1 - ε_t)
            #       α_t = log(1 / β_t)
            #
            # 数值稳定性处理：防止 ε=0（完美分类）或 ε=1 时取对数溢出
            eps = 1e-10
            epsilon_t = float(np.clip(best_error, eps, 1.0 - eps))
            beta_t  = epsilon_t / (1.0 - epsilon_t)
            alpha_t = np.log(1.0 / beta_t)

            # 记录弱分类器
            wc = WeakClassifier(
                feature_idx=best_feat_idx,
                threshold=best_threshold,
                polarity=best_polarity,
                alpha=alpha_t,
                error=epsilon_t,
            )
            strong_clf.weak_classifiers.append(wc)

            # ── Step 5：更新样本权重 ──────────────────────────────────────────────
            # w_{t+1,i} = w_{t,i} · β_t^{1 - e_i}
            # 其中 e_i = 0 若样本 x_i 被正确分类，e_i = 1 否则
            #
            # 即：分类正确的样本权重乘以 β_t（降权），分类错误的不变。

            # 计算本轮弱分类器对所有样本的预测
            f_vals = X[:, best_feat_idx]
            predictions = (best_polarity * f_vals < best_polarity * best_threshold).astype(np.int32)

            # e_i：分类正确为 0，分类错误为 1
            e_i = (predictions != y).astype(np.float64)

            # 权重更新：w_{t+1,i} = w_{t,i} · β_t^{1 - e_i}
            # 分类正确（e_i=0）→ 乘以 β_t^1 = β_t（降权）
            # 分类错误（e_i=1）→ 乘以 β_t^0 = 1.0（不变）
            weights *= np.power(beta_t, 1.0 - e_i)

            round_elapsed = time.time() - round_start

            # ── 打印本轮信息 ──────────────────────────────────────────────────────
            n_correct = int((predictions == y).sum())
            train_acc = n_correct / n_samples * 100

            if verbose:
                print(
                    f"[AdaBoost] 第 {t:3d}/{n_features_to_select} 轮 | "
                    f"特征索引={best_feat_idx:6d} | "
                    f"极性={best_polarity:+d} | "
                    f"阈值={best_threshold:+.4f} | "
                    f"ε={epsilon_t:.4f} | "
                    f"α={alpha_t:.4f} | "
                    f"β={beta_t:.4f} | "
                    f"训练准确率={train_acc:.2f}% | "
                    f"耗时={round_elapsed:.2f}s"
                )
            elif t % max(1, n_features_to_select // 10) == 0:
                # 非 verbose 模式下每 10% 打印一次进度
                print(
                    f"[AdaBoost] 进度 {t}/{n_features_to_select} "
                    f"({t/n_features_to_select*100:.0f}%) | "
                    f"ε={epsilon_t:.4f} | α={alpha_t:.4f}"
                )
    finally:
        # 🚀 无论训练是否异常，必须关闭并回收进程池
        pool.close()
        pool.join()

    # ── 设置默认阈值：½·sum_alpha_t（论文 Table 1 最终强分类器公式）──────────────
    strong_clf._update_threshold()

    # 打印强分类器汇总
    total_alpha = sum(wc.alpha for wc in strong_clf.weak_classifiers)
    print(f"\n[AdaBoost] 训练完成！")
    print(f"  弱分类器数量 : {len(strong_clf.weak_classifiers)}")
    print(f"  sum_alpha_t          : {total_alpha:.4f}")
    print(f"  threshold : {strong_clf.threshold:.4f}")

    # 在训练集上评估强分类器整体性能
    train_preds = strong_clf.classify(X)
    tp = int(((train_preds == 1) & (y == 1)).sum())
    fp = int(((train_preds == 1) & (y == 0)).sum())
    fn = int(((train_preds == 0) & (y == 1)).sum())
    tn = int(((train_preds == 0) & (y == 0)).sum())
    detection_rate = tp / n_pos * 100 if n_pos > 0 else 0.0
    false_positive_rate = fp / n_neg * 100 if n_neg > 0 else 0.0
    print(f"  训练集检测率 (TPR) : {detection_rate:.2f}%  ({tp}/{n_pos})")
    print(f"  训练集假正率 (FPR) : {false_positive_rate:.2f}%  ({fp}/{n_neg})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*60}\n")

    return strong_clf


# ──────────────────────────────────────────────────────────────────────────────
# 辅助函数：阈值调整（供级联训练模块调用）
# ──────────────────────────────────────────────────────────────────────────────

def adjust_threshold_for_detection_rate(
    strong_clf: StrongClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_detection_rate: float = 0.99,
) -> Tuple[float, float, float]:
    """
    在验证集上调整强分类器阈值，以达到目标检测率。

    参数：
        strong_clf            : StrongClassifier — 待调整的强分类器（原地修改）
        X_val                 : ndarray, (n_val, n_features) — 验证集特征
        y_val                 : ndarray, (n_val,)            — 验证集标签
        target_detection_rate : float — 目标最低检测率（默认 0.99）

    返回：
        (new_threshold, actual_detection_rate, actual_false_positive_rate)
    """
    print(f"\n[阈值调整] 目标检测率 ≥ {target_detection_rate*100:.1f}%")

    val_pos_mask = (y_val == 1)
    val_neg_mask = (y_val == 0)
    n_val_pos = val_pos_mask.sum()
    n_val_neg = val_neg_mask.sum()

    # 计算所有验证样本的连续得分
    scores = strong_clf.score(X_val)

    pos_scores = np.sort(scores[val_pos_mask])[::-1]  # 降序排列，从高到低尝试

    # 从高到低尝试每个正样本得分作为阈值
    for candidate_threshold in pos_scores:
        preds = (scores >= candidate_threshold).astype(np.int32)
        tp = int(((preds == 1) & val_pos_mask).sum())
        fp = int(((preds == 1) & val_neg_mask).sum())
        dr = tp / n_val_pos if n_val_pos > 0 else 0.0
        fpr = fp / n_val_neg if n_val_neg > 0 else 0.0

        if dr >= target_detection_rate:
            strong_clf.threshold = float(candidate_threshold)
            print(
                f"[阈值调整] 找到满足条件的阈值："
                f"θ={candidate_threshold:.4f} | "
                f"检测率={dr*100:.2f}% | 假正率={fpr*100:.2f}%"
            )
            return float(candidate_threshold), dr, fpr

    # 若无法达到目标，将阈值设为最小正样本得分
    min_threshold = float(pos_scores[0]) if len(pos_scores) > 0 else 0.0
    strong_clf.threshold = min_threshold
    preds = (scores >= min_threshold).astype(np.int32)
    tp = int(((preds == 1) & val_pos_mask).sum())
    fp = int(((preds == 1) & val_neg_mask).sum())
    dr  = tp / n_val_pos if n_val_pos > 0 else 0.0
    fpr = fp / n_val_neg if n_val_neg > 0 else 0.0

    print(
        f"[阈值调整] 警告：无法达到目标检测率！"
        f"当前最佳：θ={min_threshold:.4f} | 检测率={dr*100:.2f}% | 假正率={fpr*100:.2f}%"
    )
    return min_threshold, dr, fpr


def evaluate_strong_classifier(
    strong_clf: StrongClassifier,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = "验证集",
) -> Tuple[float, float]:
    """
    在给定数据集上评估强分类器的检测率和假正率。
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    preds = strong_clf.classify(X)
    tp = int(((preds == 1) & (y == 1)).sum())
    fp = int(((preds == 1) & (y == 0)).sum())
    fn = int(((preds == 0) & (y == 1)).sum())
    tn = int(((preds == 0) & (y == 0)).sum())

    dr  = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0

    print(
        f"[评估/{dataset_name}] "
        f"检测率={dr*100:.2f}% ({tp}/{n_pos}) | "
        f"假正率={fpr*100:.2f}% ({fp}/{n_neg}) | "
        f"TP={tp} FP={fp} FN={fn} TN={tn}"
    )
    return dr, fpr


# ──────────────────────────────────────────────────────────────────────────────
# 模型序列化工具
# ──────────────────────────────────────────────────────────────────────────────

def save_strong_classifier(strong_clf: StrongClassifier, path: str) -> None:
    """将强分类器序列化保存到磁盘。"""
    with open(path, "wb") as f:
        pickle.dump(strong_clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[序列化] 强分类器已保存到：{path}")


def load_strong_classifier(path: str) -> StrongClassifier:
    """从磁盘加载强分类器。"""
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"[序列化] 已加载强分类器：{path}（{len(clf.weak_classifiers)} 个弱分类器）")
    return clf


# ──────────────────────────────────────────────────────────────────────────────
# 单元测试 / 快速验证
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    合成数据冒烟测试。
    """
    print("=" * 60)
    print("AdaBoost 单元测试（合成数据）")
    print("=" * 60)

    np.random.seed(42)

    n_pos_train = 200
    n_neg_train = 400
    n_features  = 50

    X_pos = np.random.randn(n_pos_train, n_features) + 1.0
    X_neg = np.random.randn(n_neg_train, n_features) - 1.0

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.hstack([
        np.ones(n_pos_train, dtype=np.int32),
        np.zeros(n_neg_train, dtype=np.int32),
    ])

    shuffle_idx = np.random.permutation(len(y_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    print(f"\n训练集：{n_pos_train} 正样本 + {n_neg_train} 负样本，{n_features} 维特征\n")

    T = 5
    strong_clf = train_adaboost(X_train, y_train, n_features_to_select=T, verbose=True)

    print("\n── 在训练集上评估 ──")
    dr, fpr = evaluate_strong_classifier(strong_clf, X_train, y_train, "训练集")
    assert dr > 0.85, f"训练集检测率过低：{dr:.2f}"
    print(f"  ✓ 训练集检测率 {dr*100:.1f}% > 85%，通过")

    X_val_pos = np.random.randn(100, n_features) + 1.0
    X_val_neg = np.random.randn(200, n_features) - 1.0
    X_val = np.vstack([X_val_pos, X_val_neg])
    y_val = np.hstack([np.ones(100, dtype=np.int32), np.zeros(200, dtype=np.int32)])

    print("\n── 阈值调整测试 ──")
    new_thresh, dr_val, fpr_val = adjust_threshold_for_detection_rate(
        strong_clf, X_val, y_val, target_detection_rate=0.99
    )
    print(f"  调整后阈值={new_thresh:.4f}  验证集检测率={dr_val*100:.1f}%  假正率={fpr_val*100:.1f}%")

    scores = strong_clf.score(X_val)
    print(f"\n── score() 接口测试 ──")
    print(f"  得分范围：[{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  正样本平均得分：{scores[y_val==1].mean():.4f}（期望 > 负样本）")
    print(f"  负样本平均得分：{scores[y_val==0].mean():.4f}")
    assert scores[y_val==1].mean() > scores[y_val==0].mean(), \
        "正样本平均得分应高于负样本！"
    print("  ✓ 正样本得分高于负样本，通过")

    import tempfile, os
    print(f"\n── 序列化 / 反序列化测试 ──")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        save_strong_classifier(strong_clf, tmp_path)
        loaded_clf = load_strong_classifier(tmp_path)
        preds_orig   = strong_clf.classify(X_val)
        preds_loaded = loaded_clf.classify(X_val)
        assert np.array_equal(preds_orig, preds_loaded), "序列化前后预测结果不一致！"
        print("  ✓ 序列化/反序列化一致性通过")
    finally:
        os.unlink(tmp_path)

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)