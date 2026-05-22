"""
haar_features.py
================
Haar-like 矩形特征枚举与计算模块

位置：train/haar_features.py
被引用：
    - train/adaboost.py        —— 训练时对每个样本计算全部特征值矩阵
    - detect/detector.py       —— 检测时对每个子窗口计算特征值
    - train/train_cascade.py   —— 训练入口，调用 enumerate_features() 初始化

─────────────────────────────────────────────────────────────
论文对应关系（Section 2 & Fig.1）
─────────────────────────────────────────────────────────────
论文定义四种 Haar-like 特征（见 Fig.1 A/B/C/D）：

  类型 0 — 水平两矩形（Fig.1 A）：左右排列，最小单元 sx=2, sy=1
            特征值 = 右矩形像素和 − 左矩形像素和
            约束：x + W ≤ WIN（W 为 2 的倍数），y + H ≤ WIN
            总数：43,200

  类型 1 — 垂直两矩形（Fig.1 B）：上下排列，最小单元 sx=1, sy=2
            特征值 = 下矩形像素和 − 上矩形像素和
            约束：x + W ≤ WIN，y + H ≤ WIN（H 为 2 的倍数）
            总数：43,200

  类型 2 — 水平三矩形（Fig.1 C）：左中右，最小单元 sx=3, sy=1
            特征值 = 中间矩形像素和 − (左矩形 + 右矩形)像素和之和
            约束：x + W ≤ WIN（W 为 3 的倍数），y + H ≤ WIN
            总数：27,600

  类型 3 — 垂直三矩形（Fig.1 C 的转置）：上中下，最小单元 sx=1, sy=3
            特征值 = 中间矩形像素和 − (上矩形 + 下矩形)像素和之和
            约束：x + W ≤ WIN，y + H ≤ WIN（H 为 3 的倍数）
            总数：27,600
            ⚠ 论文 Fig.1 未单独画出此类型，但它是 162,336 精确总数的组成部分

  类型 4 — 四矩形（Fig.1 D）：2×2 排列，最小单元 sx=2, sy=2
            特征值 = (左上+右下) − (右上+左下)（对角线差）
            约束：x + W ≤ WIN（W 为 2 的倍数），y + H ≤ WIN（H 为 2 的倍数）
            总数：20,736

  精确总数（WIN=24）：43200 + 43200 + 27600 + 27600 + 20736 = 162,336
  论文原文写"about 160,000"是近似表述，Wikipedia 和 OpenCV 文档均确认精确值为 162,336。

  其中 x/y 为特征左上角列/行坐标，W/H 为整个特征组合的总宽/总高。

─────────────────────────────────────────────────────────────
特征描述符格式（FeatureDesc namedtuple）
─────────────────────────────────────────────────────────────
每条描述符是一个轻量 namedtuple：
    FeatureDesc(ftype, r, c, h, w)
        ftype : int，0/1/2/3，对应上面四种类型
        r     : int，矩形左上角行坐标（原始图像坐标，0-based）
        c     : int，矩形左上角列坐标
        h     : int，单个子矩形的高度
        w     : int，单个子矩形的宽度

描述符列表由 enumerate_features() 一次性生成，之后在训练/检测
全程复用，不重复计算。

─────────────────────────────────────────────────────────────
与 adaboost.py 的适配约定
─────────────────────────────────────────────────────────────
adaboost.py 使用以下两种调用方式：

  （A）单样本单特征（弱分类器评估阶段）：
        val = compute_feature(desc, iimg.ii)
        其中 iimg 是 IntegralImage 对象，iimg.ii 是 padded 积分图

  （B）批量预计算全部特征值矩阵（训练加速）：
        feat_matrix = compute_all_features(iimgs, features)
        返回 shape=(N_samples, N_features) 的 float32 矩阵，
        训练时直接按列切片取单个特征的全部样本值，避免重复计算积分图

─────────────────────────────────────────────────────────────
关键函数索引
─────────────────────────────────────────────────────────────
enumerate_features(win_size)     → List[FeatureDesc]
    枚举指定窗口大小内的全部合法特征描述符（~160,000 个）

compute_feature(desc, ii)        → float
    给定一条描述符和 padded 积分图，O(1) 计算单个特征值

compute_all_features(iimgs, features, scale) → np.ndarray  (N, F) float32
    批量计算 N 个样本的 F 个特征值，供 adaboost.py 训练时调用

参考论文：Viola & Jones, "Robust Real-Time Face Detection", IJCV 2004
         Section 2（Features）、Fig.1
"""

import numpy as np
from collections import namedtuple
from typing import List

# integral_image 模块提供底层积分图查询，保持松耦合（仅依赖 rect_sum 函数）
from integral_image import rect_sum, IntegralImage


# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────

BASE_WIN_SIZE = 24    # 论文规定的基础检测窗口边长（像素）

# 特征类型编号（与论文 Fig.1 对应）
# 注意：论文 Fig.1 画了 4 个图（A/B/C/D），但完整枚举时有 5 种类型。
# 水平三矩形（H3）和垂直三矩形（V3）是两种独立类型，
# V3 在 Fig.1 中未单独列出，但在 162,336 总数的精确计数中必须包含。
FEAT_H2 = 0    # 水平两矩形（Fig.1 A）：左右排列
FEAT_V2 = 1    # 垂直两矩形（Fig.1 B）：上下排列
FEAT_H3 = 2    # 水平三矩形（Fig.1 C）：左中右排列
FEAT_V3 = 3    # 垂直三矩形（Fig.1 C 的转置）：上中下排列
FEAT_D4 = 4    # 四矩形    （Fig.1 D）：2×2 对角差

# 五种类型的文字说明，供打印日志使用
FEAT_NAMES = {
    FEAT_H2: "水平两矩形",
    FEAT_V2: "垂直两矩形",
    FEAT_H3: "水平三矩形",
    FEAT_V3: "垂直三矩形",
    FEAT_D4: "四矩形",
}


# ─────────────────────────────────────────────────────────────
# 特征描述符数据结构
# ─────────────────────────────────────────────────────────────

# 用 namedtuple 而非 dataclass：轻量、可哈希、序列化友好（pickle/json）
# adaboost.py 将整条描述符存入弱分类器参数，需要可序列化
FeatureDesc = namedtuple("FeatureDesc", ["ftype", "r", "c", "h", "w"])
"""
Haar-like 特征描述符。

字段：
    ftype (int) : 特征类型，0=水平两矩形，1=垂直两矩形，2=水平三矩形，3=垂直三矩形，4=四矩形
    r     (int) : 矩形组左上角行坐标（0-based，相对检测窗口）
    c     (int) : 矩形组左上角列坐标
    h     (int) : 单个子矩形高度（像素）
    w     (int) : 单个子矩形宽度（像素）

注意：r/c/h/w 都是在 **基础 24×24 窗口** 中的坐标。
检测时若当前检测尺度为 scale，实际像素坐标需乘以 scale（由 detector.py 负责缩放）。
"""


# ─────────────────────────────────────────────────────────────
# 特征枚举
# ─────────────────────────────────────────────────────────────

def enumerate_features(win_size: int = BASE_WIN_SIZE) -> List[FeatureDesc]:
    """
    在给定正方形窗口内枚举全部合法的 Haar-like 特征描述符，共 5 种类型。

    枚举规则（MATLAB 风格，与 Wikipedia/OpenCV 实现一致）：
        每种类型由最小单元尺寸 (sx, sy) 参数化。
        对每个起点 (r, c) 和总尺寸 (H, W)，只要满足下列约束即为合法：
            c + W ≤ win_size   （不超出右边界）
            r + H ≤ win_size   （不超出下边界）
            W 是 sx 的倍数，H 是 sy 的倍数（保证可以等分为子矩形）

        FeatureDesc 中的 w/h 存储的是"单个子矩形"的宽/高：
            w = W / sx_units（总宽 / 水平子格数）
            h = H / sy_units（总高 / 垂直子格数）
        这样 compute_feature 可以直接用 w/h 定位各子矩形。

    五种类型及其精确数量（win_size=24）：
        FEAT_H2 (sx=2, sy=1)：43,200 个
        FEAT_V2 (sx=1, sy=2)：43,200 个
        FEAT_H3 (sx=3, sy=1)：27,600 个
        FEAT_V3 (sx=1, sy=3)：27,600 个   ← 垂直三矩形，易被遗漏
        FEAT_D4 (sx=2, sy=2)：20,736 个
        合计：162,336 个（论文近似称"约 160,000"）

    参数：
        win_size : 检测窗口边长，默认 24（论文设定）

    返回：
        List[FeatureDesc]，长度 162,336（win_size=24 时）
        列表顺序固定（按类型 0→4，同类型内按 r→c→h→w 字典序）。

    时间复杂度：O(win_size⁴)，只在训练开始前调用一次。
    """
    print(f"  [enumerate_features] 开始枚举特征，窗口大小={win_size}×{win_size}")
    features: List[FeatureDesc] = []

    # ── 类型 0：水平两矩形（H2，sx=2, sy=1）──────────────────
    # 总尺寸 W=2w（左右各 w），H=h
    # 约束：c + 2w ≤ win_size，r + h ≤ win_size
    # FeatureDesc.w = 单个子矩形宽，FeatureDesc.h = 高
    count_h2 = 0
    for r in range(win_size):
        for c in range(win_size):
            for h in range(1, win_size - r + 1):          # H = h，r+h ≤ win_size
                for w in range(1, (win_size - c) // 2 + 1):  # W = 2w，c+2w ≤ win_size
                    features.append(FeatureDesc(FEAT_H2, r, c, h, w))
                    count_h2 += 1
    print(f"    FEAT_H2 (水平两矩形): {count_h2:>7} 个  （期望 43200）")

    # ── 类型 1：垂直两矩形（V2，sx=1, sy=2）──────────────────
    # 总尺寸 W=w，H=2h（上下各 h）
    # 约束：c + w ≤ win_size，r + 2h ≤ win_size
    count_v2 = 0
    for r in range(win_size):
        for c in range(win_size):
            for h in range(1, (win_size - r) // 2 + 1):   # H = 2h，r+2h ≤ win_size
                for w in range(1, win_size - c + 1):       # W = w，c+w ≤ win_size
                    features.append(FeatureDesc(FEAT_V2, r, c, h, w))
                    count_v2 += 1
    print(f"    FEAT_V2 (垂直两矩形): {count_v2:>7} 个  （期望 43200）")

    # ── 类型 2：水平三矩形（H3，sx=3, sy=1）──────────────────
    # 总尺寸 W=3w（左中右各 w），H=h
    # 约束：c + 3w ≤ win_size，r + h ≤ win_size
    count_h3 = 0
    for r in range(win_size):
        for c in range(win_size):
            for h in range(1, win_size - r + 1):
                for w in range(1, (win_size - c) // 3 + 1):   # W = 3w，c+3w ≤ win_size
                    features.append(FeatureDesc(FEAT_H3, r, c, h, w))
                    count_h3 += 1
    print(f"    FEAT_H3 (水平三矩形): {count_h3:>7} 个  （期望 27600）")

    # ── 类型 3：垂直三矩形（V3，sx=1, sy=3）──────────────────
    # 总尺寸 W=w，H=3h（上中下各 h）
    # 特征值 = 中间矩形像素和 − (上矩形 + 下矩形)像素和之和
    # 约束：c + w ≤ win_size，r + 3h ≤ win_size
    # ⚠ 此类型是论文 Fig.1 未单独标注但必须纳入枚举的第五种类型
    count_v3 = 0
    for r in range(win_size):
        for c in range(win_size):
            for h in range(1, (win_size - r) // 3 + 1):   # H = 3h，r+3h ≤ win_size
                for w in range(1, win_size - c + 1):       # W = w，c+w ≤ win_size
                    features.append(FeatureDesc(FEAT_V3, r, c, h, w))
                    count_v3 += 1
    print(f"    FEAT_V3 (垂直三矩形): {count_v3:>7} 个  （期望 27600）")

    # ── 类型 4：四矩形（D4，sx=2, sy=2）──────────────────────
    # 总尺寸 W=2w，H=2h（2×2 排列）
    # 特征值 = (左上+右下) − (右上+左下)（对角线差）
    # 约束：c + 2w ≤ win_size，r + 2h ≤ win_size
    count_d4 = 0
    for r in range(win_size):
        for c in range(win_size):
            for h in range(1, (win_size - r) // 2 + 1):
                for w in range(1, (win_size - c) // 2 + 1):
                    features.append(FeatureDesc(FEAT_D4, r, c, h, w))
                    count_d4 += 1
    print(f"    FEAT_D4 (四矩形)    : {count_d4:>7} 个  （期望 20736）")

    total = len(features)
    print(f"  [enumerate_features] 枚举完成，共 {total} 个特征描述符")
    print(f"    精确值 162,336（论文近似称'约 160,000'）")
    return features


# ─────────────────────────────────────────────────────────────
# 单特征计算（O(1) per feature）
# ─────────────────────────────────────────────────────────────

def compute_feature(desc: FeatureDesc, ii: np.ndarray) -> float:
    """
    给定一条特征描述符和 padded 积分图，计算该特征的值。

    核心思想：所有子矩形的像素和均通过 rect_sum(ii, r, c, h, w) 在 O(1)
    时间内获取，最终以加减法组合得到特征值。

    参数：
        desc : FeatureDesc，包含 ftype/r/c/h/w
        ii   : padded 积分图（IntegralImage.ii），shape=(H+1, W+1)，dtype=float64
               注意传入的是 iimg.ii，不是 IntegralImage 对象本身

    返回：
        float，特征值（有正有负）

    各类型计算细节：

    FEAT_H2（水平两矩形）：
        左矩形覆盖 [r:r+h, c:c+w]
        右矩形覆盖 [r:r+h, c+w:c+2w]
        特征值 = sum(右) − sum(左)

    FEAT_V2（垂直两矩形）：
        上矩形覆盖 [r:r+h,   c:c+w]
        下矩形覆盖 [r+h:r+2h, c:c+w]
        特征值 = sum(下) − sum(上)

    FEAT_H3（三矩形）：
        左矩形覆盖 [r:r+h, c:c+w]
        中矩形覆盖 [r:r+h, c+w:c+2w]
        右矩形覆盖 [r:r+h, c+2w:c+3w]
        特征值 = sum(中) − sum(左) − sum(右)
        （等价于：中间 × 2 − 整体，也可用两次 rect_sum 实现，见下方注释）

    FEAT_D4（四矩形）：
        左上覆盖 [r:r+h,   c:c+w]
        右上覆盖 [r:r+h,   c+w:c+2w]
        左下覆盖 [r+h:r+2h, c:c+w]
        右下覆盖 [r+h:r+2h, c+w:c+2w]
        特征值 = sum(左上+右下) − sum(右上+左下)

    rect_sum 调用次数（与论文 Section 2.1 一致）：
        FEAT_H2 : 6 次（两矩形共用上下边，可优化为 6，这里用 6 次）
        FEAT_V2 : 6 次
        FEAT_H3 : 8 次（三矩形，两侧各 6-2=4 次，去公共边后 8 次）
        FEAT_D4 : 9 次（四矩形可共享 5 个角点，9 次）
    """
    r, c, h, w = desc.r, desc.c, desc.h, desc.w

    if desc.ftype == FEAT_H2:
        # ── 水平两矩形 ──
        # 左矩形：(r, c, h, w)      右矩形：(r, c+w, h, w)
        sum_left  = rect_sum(ii, r, c,     h, w)
        sum_right = rect_sum(ii, r, c + w, h, w)
        return sum_right - sum_left

    elif desc.ftype == FEAT_V2:
        # ── 垂直两矩形 ──
        # 上矩形：(r, c, h, w)      下矩形：(r+h, c, h, w)
        sum_top    = rect_sum(ii, r,     c, h, w)
        sum_bottom = rect_sum(ii, r + h, c, h, w)
        return sum_bottom - sum_top

    elif desc.ftype == FEAT_H3:
        # ── 水平三矩形 ──
        # 左矩形：(r, c, h, w)   中矩形：(r, c+w, h, w)   右矩形：(r, c+2w, h, w)
        # 特征值 = 中 − 左 − 右
        sum_left   = rect_sum(ii, r, c,          h, w)
        sum_center = rect_sum(ii, r, c +     w,  h, w)
        sum_right  = rect_sum(ii, r, c + 2 * w,  h, w)
        return sum_center - sum_left - sum_right

    elif desc.ftype == FEAT_V3:
        # ── 垂直三矩形 ──
        # 上矩形：(r,   c, h, w)   中矩形：(r+h,   c, h, w)   下矩形：(r+2h, c, h, w)
        # 特征值 = 中 − 上 − 下
        sum_top    = rect_sum(ii, r,           c, h, w)
        sum_center = rect_sum(ii, r +     h,   c, h, w)
        sum_bottom = rect_sum(ii, r + 2 * h,   c, h, w)
        return sum_center - sum_top - sum_bottom

    elif desc.ftype == FEAT_D4:
        # ── 四矩形（2×2）──
        # 左上：(r,   c,   h, w)    右上：(r,   c+w, h, w)
        # 左下：(r+h, c,   h, w)    右下：(r+h, c+w, h, w)
        # 对角线差：(左上+右下) − (右上+左下)
        sum_tl = rect_sum(ii, r,     c,     h, w)
        sum_tr = rect_sum(ii, r,     c + w, h, w)
        sum_bl = rect_sum(ii, r + h, c,     h, w)
        sum_br = rect_sum(ii, r + h, c + w, h, w)
        return (sum_tl + sum_br) - (sum_tr + sum_bl)

    else:
        raise ValueError(
            f"[compute_feature] 未知特征类型 ftype={desc.ftype}，"
            f"合法值为 0/1/2/3/4"
        )


# ─────────────────────────────────────────────────────────────
# 批量特征矩阵计算（供 adaboost.py 训练使用）
# ─────────────────────────────────────────────────────────────

def compute_all_features(
    iimgs: list,
    features: List[FeatureDesc],
    scale: float = 1.0,
) -> np.ndarray:
    """
    批量计算 N 个样本的全部 F 个特征值，返回 (N, F) float32 矩阵。

    训练时调用策略：
        训练开始前调用一次，将结果缓存到内存。
        AdaBoost 每轮迭代时，直接用 feat_matrix[:, feat_idx] 取单个特征
        的全部样本值，排序后计算最优阈值，无需重复查积分图。

    参数：
        iimgs    : List[IntegralImage]，长度 N，每个对应一张 24×24 样本的积分图
                   由 integral_image.build_batch() 生成
        features : List[FeatureDesc]，长度 F，由 enumerate_features() 生成
        scale    : float，检测器缩放比例（训练时固定为 1.0；检测时由 detector.py
                   处理缩放，不在此函数中缩放）

    返回：
        np.ndarray，shape=(N, F)，dtype=float32
        feat_matrix[i, j] = 第 i 个样本的第 j 个特征值

    打印：
        每处理 1000 个样本打印一次进度，并在结束时打印矩阵形状与内存占用。

    内存估算：
        N=14000（训练集+负样本），F=162336 个特征：
        14000 × 162336 × 4 bytes ≈ 9.1 GB — 超出普通机器内存
        实际训练时建议用 max_pos/max_neg 限制为 4000+10000=14000 样本，
        或改用在线（on-the-fly）特征计算（每轮 AdaBoost 重新算）以节省内存。
        此函数主要用于小规模快速验证或内存充足的环境。
    """
    N = len(iimgs)
    F = len(features)
    print(f"  [compute_all_features] 开始批量计算特征矩阵...")
    print(f"    样本数 N={N}，特征数 F={F}，scale={scale:.3f}")
    print(f"    预计内存占用: {N * F * 4 / 1024**2:.1f} MB (float32)")

    # 预分配结果矩阵（float32 节省内存，精度足够 AdaBoost 使用）
    feat_matrix = np.zeros((N, F), dtype=np.float32)

    for i, iimg in enumerate(iimgs):
        # 对第 i 个样本，计算全部 F 个特征值
        for j, desc in enumerate(features):
            feat_matrix[i, j] = compute_feature(desc, iimg.ii)

        # 进度打印（每 1000 个样本）
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"    进度: {i+1}/{N} ({(i+1)/N*100:.1f}%)")

    mem_mb = feat_matrix.nbytes / 1024 ** 2
    print(f"  [compute_all_features] 完成，矩阵 shape={feat_matrix.shape}，"
          f"实际占用 {mem_mb:.1f} MB")
    return feat_matrix


# ─────────────────────────────────────────────────────────────
# 检测时的单窗口特征计算（供 detector.py 使用）
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 检测时的特征计算（供 cascade_classifier.py 逐层调用）
# ─────────────────────────────────────────────────────────────

def compute_feature_at_scale(
    desc: FeatureDesc,
    ii: np.ndarray,
    scale: float,
    win_r: int,
    win_c: int,
) -> float:
    """
    检测阶段：在整张图像的积分图上，计算某个子窗口内单个特征的值。

    ── 论文 Section 5.5 的做法 ──────────────────────────────────
    论文缩放"检测器"而非图像：整张图像只构建一次积分图；
    对每个尺度 scale，检测窗口在图像上的实际像素大小是 24×scale，
    特征描述符的坐标也按同一 scale 放大后再加窗口偏移，
    就可以在同一张积分图上直接查询，无需重采样图像。

    坐标变换公式：
        实际行坐标 = win_r + round(desc.r × scale)
        实际列坐标 = win_c + round(desc.c × scale)
        实际高度   = max(1, round(desc.h × scale))
        实际宽度   = max(1, round(desc.w × scale))

    ── 在检测流程中的位置 ────────────────────────────────────────
    调用链（自顶向下）：

        detector.py
          └─ 对每个尺度 scale（从 1.0 按 1.25 倍递增）
               └─ 对每个窗口位置 (win_r, win_c)（步长 = round(scale × delta)）
                    └─ cascade_classifier.py
                         └─ 对级联第 i 层的每个弱分类器 t：
                              val = compute_feature_at_scale(
                                        desc_t, iimg.ii, scale, win_r, win_c)
                              若本层强分类器拒绝 → 立即跳出，不再计算后续层
                         └─ 全部层通过 → 记录为候选检测框

    这是级联结构效率的关键：大多数窗口在第 1 层（仅 2 个特征）就被拒绝，
    平均每个窗口只需计算约 8 个特征（论文 Section 5.3）。

    参数：
        desc   : FeatureDesc，特征描述符（坐标为 24×24 基础窗口内的坐标）
        ii     : 整张检测图像的 padded 积分图（IntegralImage.ii），
                 shape=(H_img+1, W_img+1)
                 注意：这是整张大图的积分图，不是裁剪后的小图
        scale  : float，当前检测尺度（相对 24×24 基础窗口的缩放倍数）
                 论文使用 1.0、1.25、1.25²、... 逐级递增
        win_r  : int，当前子窗口左上角在原始图像中的行坐标
        win_c  : int，当前子窗口左上角在原始图像中的列坐标

    返回：
        float，该特征在当前窗口/尺度下的值

    注意：
        不做越界检查（由 detector.py 保证 win_r/win_c 合法），
        保持函数极简以支持高频调用。
    """
    # 将描述符的 24×24 基础坐标 → 当前尺度下的实际图像坐标
    r_scaled = win_r + int(round(desc.r * scale))
    c_scaled = win_c + int(round(desc.c * scale))
    h_scaled = max(1, int(round(desc.h * scale)))
    w_scaled = max(1, int(round(desc.w * scale)))

    # 用缩放后的坐标构造临时描述符，复用 compute_feature 的计算逻辑
    scaled_desc = FeatureDesc(desc.ftype, r_scaled, c_scaled, h_scaled, w_scaled)
    return compute_feature(scaled_desc, ii)


def iter_window_positions(img_h: int, img_w: int,
                          scale: float, delta: float = 1.0):
    """
    生成器：按论文 Section 5.5 的规则，产出某一尺度下所有合法的窗口位置。

    论文原文：
        "The detector is also scanned across location. Subsequent locations
         are obtained by shifting the window some number of pixels Δ.
         This shifting process is affected by the scale of the detector:
         if the current scale is s the window is shifted by [sΔ]."

    也就是说步长不是固定像素数，而是 round(scale × delta)，
    scale 越大检测窗口越大，步长也相应变大，保持覆盖密度大致不变。

    参数：
        img_h  : 图像高度（像素）
        img_w  : 图像宽度（像素）
        scale  : 当前检测尺度
        delta  : 基础步长因子，论文使用 1.0（精度优先）或 1.5（速度优先）

    产出：
        (win_r, win_c) — 窗口左上角坐标（保证窗口完整在图像内）

    窗口实际大小：round(24 × scale) × round(24 × scale)
    """
    win_size = int(round(BASE_WIN_SIZE * scale))   # 当前尺度下窗口的像素边长
    step     = max(1, int(round(scale * delta)))   # 滑动步长，至少 1 像素

    print(f"  [iter_window_positions] scale={scale:.4f}, "
          f"win_size={win_size}px, step={step}px, "
          f"图像={img_h}×{img_w}")

    r = 0
    while r + win_size <= img_h:
        c = 0
        while c + win_size <= img_w:
            yield (r, c)
            c += step
        r += step


def iter_scales(img_h: int, img_w: int,
                scale_factor: float = 1.25,
                min_face_px: int = 24,
                max_face_px: int = None) -> list:
    """
    生成论文 Section 5.5 规定的多尺度序列。

    论文原文：
        "starting at the base scale in which faces are detected at a size of
         24×24 pixels, a 384 by 288 pixel image is scanned at 12 scales
         each a factor of 1.25 larger than the last."

    也就是 scale 从 1.0 开始，每次乘以 scale_factor（默认 1.25），
    直到检测窗口超出图像尺寸为止。

    参数：
        img_h        : 图像高度
        img_w        : 图像宽度
        scale_factor : 相邻尺度的比例因子，论文固定为 1.25
        min_face_px  : 最小检测人脸像素边长（对应 scale=1.0 时的 24px）
        max_face_px  : 最大检测人脸像素边长（None = 受图像尺寸限制）

    返回：
        List[float]，尺度序列，如 [1.0, 1.25, 1.5625, ...]
    """
    if max_face_px is None:
        max_face_px = min(img_h, img_w)

    scales = []
    scale  = min_face_px / BASE_WIN_SIZE   # 通常 = 1.0
    while True:
        win_size = int(round(BASE_WIN_SIZE * scale))
        if win_size > max_face_px or win_size > img_h or win_size > img_w:
            break
        scales.append(scale)
        scale *= scale_factor

    print(f"  [iter_scales] 图像={img_h}×{img_w}, "
          f"scale_factor={scale_factor}, "
          f"共 {len(scales)} 个尺度: "
          + ", ".join(f"{s:.4f}({int(round(24*s))}px)" for s in scales))
    return scales


# ─────────────────────────────────────────────────────────────
# 直接运行：快速自测
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  haar_features.py 快速自测")
    print("=" * 55)

    from integral_image import build

    # ── 1. 枚举特征 ──────────────────────────────────────────
    print("\n[1] 枚举 24×24 窗口内全部特征...")
    t0 = time.time()
    feats = enumerate_features(win_size=24)
    elapsed = time.time() - t0
    print(f"    总计 {len(feats)} 个（精确值 162,336），耗时 {elapsed:.3f}s")

    # ── 2. 手算验证单个特征 ───────────────────────────────────
    print("\n[2] 手算验证 compute_feature...")

    # 构造简单图像：左半亮（128），右半暗（64），24×24
    img_test = np.zeros((24, 24), dtype=np.float64)
    img_test[:, :12] = 128   # 左半亮
    img_test[:, 12:] = 64    # 右半暗
    iimg_test = build(img_test)

    # H2：右-左 = 64×288 - 128×288 = -18432
    desc_h2 = FeatureDesc(FEAT_H2, r=0, c=0, h=24, w=12)
    val_h2  = compute_feature(desc_h2, iimg_test.ii)
    expected_h2 = float(img_test[:, 12:24].sum() - img_test[:, 0:12].sum())
    print(f"    FEAT_H2: val={val_h2:.1f}, expected={expected_h2:.1f}, "
          f"{'✓' if abs(val_h2 - expected_h2) < 1e-6 else '✗'}")

    # V2：下-上，上下均匀，期望=0
    desc_v2 = FeatureDesc(FEAT_V2, r=0, c=0, h=12, w=24)
    val_v2  = compute_feature(desc_v2, iimg_test.ii)
    print(f"    FEAT_V2: val={val_v2:.1f}, expected=0.0, "
          f"{'✓' if abs(val_v2) < 1e-6 else '✗'}")

    # H3：w=8，左(0:8)=128，中(8:16)=128，右(16:24)=64
    #      中−左−右 = 128×192 − 128×192 − 64×192 = -12288
    desc_h3 = FeatureDesc(FEAT_H3, r=0, c=0, h=24, w=8)
    val_h3  = compute_feature(desc_h3, iimg_test.ii)
    expected_h3 = float(img_test[:, 8:16].sum() - img_test[:, 0:8].sum() - img_test[:, 16:24].sum())
    print(f"    FEAT_H3: val={val_h3:.1f}, expected={expected_h3:.1f}, "
          f"{'✓' if abs(val_h3 - expected_h3) < 1e-6 else '✗'}")

    # V3：构造上暗中亮下暗的图，验证垂直三矩形
    img_v3 = np.zeros((24, 24), dtype=np.float64)
    img_v3[0:8,  :] = 50    # 上 8 行暗
    img_v3[8:16, :] = 200   # 中 8 行亮
    img_v3[16:24,:] = 50    # 下 8 行暗
    iimg_v3 = build(img_v3)
    desc_v3 = FeatureDesc(FEAT_V3, r=0, c=0, h=8, w=24)
    val_v3  = compute_feature(desc_v3, iimg_v3.ii)
    # 中−上−下 = 200×192 − 50×192 − 50×192 = 19200
    expected_v3 = float(img_v3[8:16,:].sum() - img_v3[0:8,:].sum() - img_v3[16:24,:].sum())
    print(f"    FEAT_V3: val={val_v3:.1f}, expected={expected_v3:.1f}, "
          f"{'✓' if abs(val_v3 - expected_v3) < 1e-6 else '✗'}")

    # D4：左亮右暗，对角线差=0（上下对称）
    desc_d4 = FeatureDesc(FEAT_D4, r=0, c=0, h=12, w=12)
    val_d4  = compute_feature(desc_d4, iimg_test.ii)
    tl = float(img_test[0:12, 0:12].sum())
    tr = float(img_test[0:12, 12:24].sum())
    bl = float(img_test[12:24, 0:12].sum())
    br = float(img_test[12:24, 12:24].sum())
    expected_d4 = (tl + br) - (tr + bl)
    print(f"    FEAT_D4: val={val_d4:.1f}, expected={expected_d4:.1f}, "
          f"{'✓' if abs(val_d4 - expected_d4) < 1e-6 else '✗'}")

    print("\n  自测完成。运行 `python test.py --modules haar_features` 查看完整测试。")