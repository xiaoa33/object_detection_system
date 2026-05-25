# train/haar_features.py
# ============================================================
# Haar-like 矩形特征枚举与计算模块（支持自适应步长压缩）
# ============================================================

import numpy as np
from collections import namedtuple
from typing import List

from train.integral_image import rect_sum, IntegralImage

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
BASE_WIN_SIZE = 24

FEAT_H2 = 0    # 水平两矩形（Fig.1 A）
FEAT_V2 = 1    # 垂直两矩形（Fig.1 B）
FEAT_H3 = 2    # 水平三矩形（Fig.1 C）
FEAT_V3 = 3    # 垂直三矩形（Fig.1 C 的转置）
FEAT_D4 = 4    # 四矩形    （Fig.1 D）

FEAT_NAMES = {
    FEAT_H2: "水平两矩形",
    FEAT_V2: "垂直两矩形",
    FEAT_H3: "水平三矩形",
    FEAT_V3: "垂直三矩形",
    FEAT_D4: "四矩形",
}

FeatureDesc = namedtuple("FeatureDesc", ["ftype", "r", "c", "h", "w"])

# ─────────────────────────────────────────────────────────────
# 检测热路径优化：内联 rect_sum（消除函数调用开销）
# ─────────────────────────────────────────────────────────────
def _rect_sum_fast(ii: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """内联版 rect_sum，消除高频调用路径上的函数调用开销"""
    return float(ii[r + h, c + w] + ii[r, c] - ii[r, c + w] - ii[r + h, c])


# ─────────────────────────────────────────────────────────────
# 支持 stride 的特征枚举函数
# ─────────────────────────────────────────────────────────────
def enumerate_features(win_size: int = BASE_WIN_SIZE, stride: int = 1) -> List[FeatureDesc]:
    """
    在给定正方形窗口内枚举合法的 Haar-like 特征描述符。

    通过引入 stride (步长) 参数，可以等比例缩减特征空间：
        - stride = 1 : 论文原版穷举，共 162,336 个特征（最适合大样本，如 4000+ 正样本）
        - stride = 2 : 稀疏采样，约 12,000 ~ 15,000 个特征（最适合中等样本，如 1000+ 正样本）
        - stride = 3 : 高度稀疏，约 2,500 ~ 3,500 个特征（最适合极小样本快速调试，如 400 正样本）
    """
    print(f"  [enumerate_features] 开始枚举特征，窗口大小={win_size}×{win_size}，步长 stride={stride}")
    features: List[FeatureDesc] = []

    # ── 类型 0：水平两矩形（H2，sx=2, sy=1）──────────────────
    count_h2 = 0
    for r in range(0, win_size, stride):
        for c in range(0, win_size, stride):
            for h in range(1, win_size - r + 1, stride):
                for w in range(1, (win_size - c) // 2 + 1, stride):
                    features.append(FeatureDesc(FEAT_H2, r, c, h, w))
                    count_h2 += 1
    print(f"    FEAT_H2 (水平两矩形): {count_h2:>7} 个")

    # ── 类型 1：垂直两矩形（V2，sx=1, sy=2）──────────────────
    count_v2 = 0
    for r in range(0, win_size, stride):
        for c in range(0, win_size, stride):
            for h in range(1, (win_size - r) // 2 + 1, stride):
                for w in range(1, win_size - c + 1, stride):
                    features.append(FeatureDesc(FEAT_V2, r, c, h, w))
                    count_v2 += 1
    print(f"    FEAT_V2 (垂直两矩形): {count_v2:>7} 个")

    # ── 类型 2：水平三矩形（H3，sx=3, sy=1）──────────────────
    count_h3 = 0
    for r in range(0, win_size, stride):
        for c in range(0, win_size, stride):
            for h in range(1, win_size - r + 1, stride):
                for w in range(1, (win_size - c) // 3 + 1, stride):
                    features.append(FeatureDesc(FEAT_H3, r, c, h, w))
                    count_h3 += 1
    print(f"    FEAT_H3 (水平三矩形): {count_h3:>7} 个")

    # ── 类型 3：垂直三矩形（V3，sx=1, sy=3）──────────────────
    count_v3 = 0
    for r in range(0, win_size, stride):
        for c in range(0, win_size, stride):
            for h in range(1, (win_size - r) // 3 + 1, stride):
                for w in range(1, win_size - c + 1, stride):
                    features.append(FeatureDesc(FEAT_V3, r, c, h, w))
                    count_v3 += 1
    print(f"    FEAT_V3 (垂直三矩形): {count_v3:>7} 个")

    # ── 类型 4：四矩形（D4，sx=2, sy=2）──────────────────────
    count_d4 = 0
    for r in range(0, win_size, stride):
        for c in range(0, win_size, stride):
            for h in range(1, (win_size - r) // 2 + 1, stride):
                for w in range(1, (win_size - c) // 2 + 1, stride):
                    features.append(FeatureDesc(FEAT_D4, r, c, h, w))
                    count_d4 += 1
    print(f"    FEAT_D4 (四矩形)    : {count_d4:>7} 个")

    total = len(features)
    print(f"  [enumerate_features] 枚举完成，共 {total} 个特征描述符")
    return features


# ─────────────────────────────────────────────────────────────
# 自适应接口（根据样本数量，自动调节特征密度，防止过拟合）
# ─────────────────────────────────────────────────────────────
def enumerate_features_adaptive(n_pos_samples: int, win_size: int = BASE_WIN_SIZE) -> List[FeatureDesc]:
    """
    【动态自适应接口】根据训练集正样本数量，自动匹配最佳的特征步长（stride）。

    规则：
        - 正样本 < 1000（极小调试集）：开启 stride=3（特征约 3000 个，防止小样本高维过拟合）
        - 1000 <= 正样本 < 3000（中等数据集）：开启 stride=2（特征约 13000 个）
        - 正样本 >= 3000（原版大样本训练）：开启 stride=1（原版 162,336 全量特征）
    """
    if n_pos_samples <= 1000:
        stride = 3
        print(f"\n[INFO] [自适应模式] 当前正样本量极低 ({n_pos_samples} 张) -> 自动采用调试级步长 stride=3")
    elif n_pos_samples <= 3000:
        stride = 2
        print(f"\n[INFO] [自适应模式] 当前正样本量中等 ({n_pos_samples} 张) -> 自动采用中等步长 stride=2")
    else:
        stride = 1
        print(f"\n[INFO] [自适应模式] 样本量充足 ({n_pos_samples} 张) -> 启用论文原版穷举步长 stride=1")

    return enumerate_features(win_size, stride=stride)


# ─────────────────────────────────────────────────────────────
# 特征计算逻辑
# ─────────────────────────────────────────────────────────────
def compute_feature(desc: FeatureDesc, ii: np.ndarray) -> float:
    r, c, h, w = desc.r, desc.c, desc.h, desc.w
    if desc.ftype == FEAT_H2:
        sum_left  = rect_sum(ii, r, c,     h, w)
        sum_right = rect_sum(ii, r, c + w, h, w)
        return sum_right - sum_left
    elif desc.ftype == FEAT_V2:
        sum_top    = rect_sum(ii, r,     c, h, w)
        sum_bottom = rect_sum(ii, r + h, c, h, w)
        return sum_bottom - sum_top
    elif desc.ftype == FEAT_H3:
        sum_left   = rect_sum(ii, r, c,          h, w)
        sum_center = rect_sum(ii, r, c +     w,  h, w)
        sum_right  = rect_sum(ii, r, c + 2 * w,  h, w)
        return sum_center - sum_left - sum_right
    elif desc.ftype == FEAT_V3:
        sum_top    = rect_sum(ii, r,           c, h, w)
        sum_center = rect_sum(ii, r +     h,   c, h, w)
        sum_bottom = rect_sum(ii, r + 2 * h,   c, h, w)
        return sum_center - sum_top - sum_bottom
    elif desc.ftype == FEAT_D4:
        sum_tl = rect_sum(ii, r,     c,     h, w)
        sum_tr = rect_sum(ii, r,     c + w, h, w)
        sum_bl = rect_sum(ii, r + h, c,     h, w)
        sum_br = rect_sum(ii, r + h, c + w, h, w)
        return (sum_tl + sum_br) - (sum_tr + sum_bl)
    else:
        raise ValueError(f"未知特征类型 ftype={desc.ftype}")

def compute_all_features(iimgs: list, features: List[FeatureDesc], scale: float = 1.0) -> np.ndarray:
    N = len(iimgs)
    F = len(features)
    print(f"  [compute_all_features] 开始批量计算特征矩阵...")
    print(f"    样本数 N={N}，特征数 F={F}，scale={scale:.3f}")
    feat_matrix = np.zeros((N, F), dtype=np.float32)
    for i, iimg in enumerate(iimgs):
        for j, desc in enumerate(features):
            feat_matrix[i, j] = compute_feature(desc, iimg.ii)
        if (i + 1) % 1000 == 0 or (i + 1) == N:
            print(f"    进度: {i+1}/{N} ({(i+1)/N*100:.1f}%)")
    return feat_matrix

def compute_feature_at_scale(desc: FeatureDesc, ii: np.ndarray, scale: float, win_r: int, win_c: int) -> float:
    """
    检测阶段：计算子窗口内单个特征在给定尺度下的值。
    已做内联优化：rs = _rect_sum_fast，不创建临时 FeatureDesc，不调用 compute_feature。
    """
    r = win_r + int(round(desc.r * scale))
    c = win_c + int(round(desc.c * scale))
    h = max(1, int(round(desc.h * scale)))
    w = max(1, int(round(desc.w * scale)))

    rs = _rect_sum_fast

    if desc.ftype == FEAT_H2:
        return (rs(ii, r, c + w, h, w) - rs(ii, r, c, h, w))
    elif desc.ftype == FEAT_V2:
        return (rs(ii, r + h, c, h, w) - rs(ii, r, c, h, w))
    elif desc.ftype == FEAT_H3:
        return (rs(ii, r, c + w, h, w) -
                rs(ii, r, c, h, w) -
                rs(ii, r, c + 2 * w, h, w))
    elif desc.ftype == FEAT_V3:
        return (rs(ii, r + h, c, h, w) -
                rs(ii, r, c, h, w) -
                rs(ii, r + 2 * h, c, h, w))
    elif desc.ftype == FEAT_D4:
        return ((rs(ii, r, c, h, w) + rs(ii, r + h, c + w, h, w)) -
                (rs(ii, r, c + w, h, w) + rs(ii, r + h, c, h, w)))
    else:
        raise ValueError(f"[compute_feature_at_scale] 未知特征类型 ftype={desc.ftype}")

def iter_window_positions(img_h: int, img_w: int, scale: float, delta: float = 1.0):
    win_size = int(round(BASE_WIN_SIZE * scale))
    step     = max(1, int(round(scale * delta)))
    r = 0
    while r + win_size <= img_h:
        c = 0
        while c + win_size <= img_w:
            yield (r, c)
            c += step
        r += step

def iter_scales(img_h: int, img_w: int, scale_factor: float = 1.25, min_face_px: int = 24, max_face_px: int = None) -> list:
    if max_face_px is None:
        max_face_px = min(img_h, img_w)
    scales = []
    scale  = min_face_px / BASE_WIN_SIZE
    while True:
        win_size = int(round(BASE_WIN_SIZE * scale))
        if win_size > max_face_px or win_size > img_h or win_size > img_w:
            break
        scales.append(scale)
        scale *= scale_factor
    return scales
