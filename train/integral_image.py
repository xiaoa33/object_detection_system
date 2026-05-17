"""
integral_image.py
=================
积分图（Integral Image）构建与矩形区域查询模块

位置：train/integral_image.py
被引用：
    - train/haar_features.py   —— 计算 Haar-like 特征值（调用 rect_sum）
    - detect/detector.py       —— 检测时为每个尺度构建积分图（调用 build）
    - train/adaboost.py        —— 训练时对样本批量构建积分图

─────────────────────────────────────────────────────────────
核心数据结构 IntegralImage
─────────────────────────────────────────────────────────────
每个 IntegralImage 对象封装两张积分图：
    ii   : 像素值积分图          —— 用于计算区域像素均值 / Haar 特征值
    ii_sq: 像素值平方积分图      —— 与 ii 配合，O(1) 计算区域方差，支持方差归一化

论文 Section 5.4 的方差归一化公式：
    σ² = E[x²] − (E[x])²
       = (ii_sq_sum / N) − (ii_sum / N)²

─────────────────────────────────────────────────────────────
关键函数索引
─────────────────────────────────────────────────────────────
build(img)              → IntegralImage          构建单图的两张积分图
build_batch(imgs)       → List[IntegralImage]    批量构建（训练时用）
rect_sum(ii, r, c, h, w)   → float              O(1) 矩形像素和查询
rect_sum_sq(ii_sq, r, c, h, w) → float          O(1) 矩形像素平方和查询
window_variance(iimg, r, c, size) → float        O(1) 子窗口方差（用于归一化）

─────────────────────────────────────────────────────────────
与 haar_features.py 的适配约定
─────────────────────────────────────────────────────────────
haar_features.py 通过 rect_sum(iimg.ii, r, c, h, w) 形式调用：
    - iimg      : IntegralImage 对象
    - iimg.ii   : numpy float64 数组，形状 (H, W)，左上角填充零行零列
    - r, c      : 矩形左上角在 **原始图像** 中的行列坐标（0-based）
    - h, w      : 矩形高度和宽度（像素数）
    - 返回值    : float，该矩形内所有像素值之和

坐标系说明（与论文 Fig.3 一致）：
    积分图 ii 在原始图像外围加了一圈零填充（padded），尺寸为 (H+1, W+1)。
    这样 rect_sum 的四角访问永远不会出现负下标，无需边界检查，代码简洁。

    原始像素 img[r, c]  →  padded 积分图 ii[r+1, c+1]
    矩形左上 (r, c)，高 h，宽 w 的四角 padded 坐标：
        A = ii[r,     c    ]   ← 左上角之上、之左（不含矩形）
        B = ii[r,     c + w]   ← 右上角之上（不含矩形）
        C = ii[r + h, c    ]   ← 左下角之左（不含矩形）
        D = ii[r + h, c + w]   ← 右下角（含矩形）
    矩形像素和 = D + A - B - C   （容斥原理，与论文 Fig.3 完全一致）

参考论文：Viola & Jones, "Robust Real-Time Face Detection", IJCV 2004, Section 2.1
"""

import numpy as np
from dataclasses import dataclass
from typing import List


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class IntegralImage:
    """
    封装一张图像的两张积分图。

    属性：
        ii      (np.ndarray, float64, shape=(H+1, W+1))
                像素值积分图，首行首列为 0（零填充边界）。
        ii_sq   (np.ndarray, float64, shape=(H+1, W+1))
                像素值平方积分图，同样带零填充边界。
        H, W    原始图像的高度和宽度（不含填充）。

    使用示例：
        iimg = build(gray_image)
        s    = rect_sum(iimg.ii, r=5, c=3, h=8, w=6)    # 矩形像素和
        var  = window_variance(iimg, r=0, c=0, size=24)  # 子窗口方差
    """
    ii:   np.ndarray   # 像素积分图，   shape = (H+1, W+1)，dtype = float64
    ii_sq: np.ndarray  # 像素平方积分图，shape = (H+1, W+1)，dtype = float64
    H:    int          # 原始图像高度
    W:    int          # 原始图像宽度


# ─────────────────────────────────────────────────────────────
# 构建函数
# ─────────────────────────────────────────────────────────────

def build(img: np.ndarray) -> IntegralImage:
    """
    从灰度图像构建带零填充边界的积分图（及平方积分图）。

    算法（论文 Section 2.1，公式 1-2）：
        s(x, y) = s(x, y-1) + i(x, y)          # 行累积和
        ii(x, y) = ii(x-1, y) + s(x, y)         # 积分图
    等价于对图像先沿列方向 cumsum，再沿行方向 cumsum。
    NumPy 的 cumsum 实现与逐像素递推完全等价，且经过 C 层优化，速度更快。

    零填充策略：
        在结果外围加一圈 0（np.pad），使积分图尺寸变为 (H+1, W+1)。
        这样 rect_sum 的四角索引 ii[r, c] 在 r=0 或 c=0 时直接取 0，
        无需任何边界条件判断。

    参数：
        img  : 2D uint8 或 float32/float64 数组，形状 (H, W)
               应为灰度图像；若传入多通道图像将触发 ValueError。

    返回：
        IntegralImage 对象，ii 和 ii_sq 均为 float64，形状 (H+1, W+1)。

    时间复杂度：O(H × W)   ← NumPy cumsum 两次扫描
    空间复杂度：O(H × W)   ← 两张与原图等大的数组（+1 填充）
    """
    # ── 输入校验 ───────────────────────────────────────────
    if img.ndim != 2:
        raise ValueError(
            f"[integral_image.build] 输入必须是 2D 灰度图，"
            f"实际 shape={img.shape}，ndim={img.ndim}。"
            f"如果是彩色图请先用 cv2.cvtColor 转换为灰度。"
        )

    print(f"  [build] 构建积分图: 输入 shape={img.shape}, dtype={img.dtype}")

    # ── float64 转换（避免 uint8 溢出）─────────────────────
    # uint8 的最大值 255，24×24 子窗口累加最大为 255×576 ≈ 146,880，
    # 完整图像（384×288）累加最大约 2800 万，float64 可精确表示。
    img_f = img.astype(np.float64)

    # ── 构建像素积分图 ii ──────────────────────────────────
    # Step1: 沿列方向累积（等价于论文公式中的行累积和 s(x,y)）
    # Step2: 再沿行方向累积，得到完整积分图
    # np.cumsum axis 约定：axis=0 沿行（竖直方向），axis=1 沿列（水平方向）
    ii_raw = np.cumsum(np.cumsum(img_f, axis=0), axis=1)

    # ── 构建像素平方积分图 ii_sq ──────────────────────────
    ii_sq_raw = np.cumsum(np.cumsum(img_f ** 2, axis=0), axis=1)

    # ── 零填充：在顶部插入全零行，左侧插入全零列 ──────────
    # np.pad((1,0),(1,0)) → 上方填充 1 行，左侧填充 1 列；右下不填充
    ii    = np.pad(ii_raw,    ((1, 0), (1, 0)), mode='constant', constant_values=0)
    ii_sq = np.pad(ii_sq_raw, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    H, W = img.shape
    print(f"  [build] 完成: ii.shape={ii.shape}, "
          f"ii[H,W]={ii[H, W]:.1f} (应等于所有像素之和 {img_f.sum():.1f})")

    return IntegralImage(ii=ii, ii_sq=ii_sq, H=H, W=W)


def build_batch(imgs: np.ndarray) -> List[IntegralImage]:
    """
    批量构建积分图，供训练时对正负样本集使用。

    参数：
        imgs : float32/uint8 数组，形状 (N, H, W)，每张为灰度图

    返回：
        List[IntegralImage]，长度为 N

    打印：
        每 500 张打印一次进度，便于观察训练时的处理速度
    """
    if imgs.ndim != 3:
        raise ValueError(
            f"[integral_image.build_batch] 输入应为 (N, H, W)，"
            f"实际 shape={imgs.shape}"
        )
    N = len(imgs)
    print(f"  [build_batch] 开始批量构建积分图，共 {N} 张图像...")
    results = []
    for i, img in enumerate(imgs):
        results.append(build(img))
        # 批量时抑制每张的详细输出，改为阶段性汇报
        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  [build_batch] 进度: {i+1}/{N} ({(i+1)/N*100:.1f}%)")
    print(f"  [build_batch] 完成，共构建 {len(results)} 个 IntegralImage 对象")
    return results


# ─────────────────────────────────────────────────────────────
# 查询函数（O(1) 矩形区域求和）
# ─────────────────────────────────────────────────────────────

def rect_sum(ii: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """
    利用积分图在 O(1) 时间内计算任意矩形区域的像素和。

    坐标系（与论文 Fig.3、Section 2.1 一致）：
        ii 已含零填充，尺寸为 (H+1, W+1)。
        原始图像像素 img[r, c] 对应 ii[r+1, c+1]。

        矩形定义：左上角为原始图像坐标 (r, c)，高 h，宽 w，
                  即覆盖 img[r:r+h, c:c+w]。

        四角在 padded 积分图中的坐标：
            A = ii[r,     c    ]   ← 矩形左上角之上/左（已被填充为 0）
            B = ii[r,     c + w]   ← 矩形右上角之上
            C = ii[r + h, c    ]   ← 矩形左下角之左
            D = ii[r + h, c + w]   ← 矩形右下角（含矩形内所有像素）

        矩形像素和 = D + A - B - C   （容斥原理）

    参数：
        ii  : 带零填充的积分图，shape=(H+1, W+1)，dtype=float64
              即 IntegralImage.ii
        r   : 矩形左上角行坐标（0-based，相对原始图像）
        c   : 矩形左上角列坐标（0-based，相对原始图像）
        h   : 矩形高度（像素数，> 0）
        w   : 矩形宽度（像素数，> 0）

    返回：
        float — 矩形区域内所有像素值之和

    调用频率：
        这是整个系统中调用最频繁的函数（训练时约每轮 AdaBoost 调用数亿次）。
        实现保持纯 NumPy 数组索引，避免任何 Python 循环。
    """
    # 四角坐标（padded 积分图坐标，已自动处理边界）
    A = ii[r,     c    ]   # 左上：矩形外，值已被 pad=0 处理
    B = ii[r,     c + w]   # 右上
    C = ii[r + h, c    ]   # 左下
    D = ii[r + h, c + w]   # 右下

    return float(D + A - B - C)


def rect_sum_sq(ii_sq: np.ndarray, r: int, c: int, h: int, w: int) -> float:
    """
    利用平方积分图在 O(1) 时间内计算矩形区域的像素平方和。

    接口与 rect_sum 完全一致，只是传入 IntegralImage.ii_sq 而非 ii。
    用于配合 rect_sum 计算子窗口方差（论文 Section 5.4）。

    参数：
        ii_sq : 带零填充的像素平方积分图，shape=(H+1, W+1)
        r, c, h, w : 同 rect_sum

    返回：
        float — 矩形区域内所有像素值平方之和（即 Σ x²）
    """
    A = ii_sq[r,     c    ]
    B = ii_sq[r,     c + w]
    C = ii_sq[r + h, c    ]
    D = ii_sq[r + h, c + w]
    return float(D + A - B - C)


def window_variance(iimg: IntegralImage, r: int, c: int, size: int) -> float:
    """
    利用双积分图在 O(1) 时间内计算正方形子窗口的像素方差。

    论文 Section 5.4 公式：
        σ² = E[x²] - (E[x])²
           = (Σx² / N) - (Σx / N)²

    用途：
        检测时对每个候选子窗口做方差归一化，消除光照变化影响。
        训练时在 data_loader.py 的 _variance_normalize() 中已对样本做过一次，
        此函数主要供 detector.py 在推理阶段使用。

    参数：
        iimg : IntegralImage 对象（含 ii 和 ii_sq）
        r    : 子窗口左上角行坐标（0-based，相对整张图像）
        c    : 子窗口左上角列坐标
        size : 子窗口边长（正方形，通常为 24 × scale，取整后传入）

    返回：
        float — 方差 σ²（≥ 0）；纯色子窗口返回 0.0

    注意：
        返回的是方差 σ²，而非标准差 σ。
        若调用方需要做归一化（除以 σ），应先做 np.sqrt，并注意 σ≈0 时的除零保护。
    """
    N = size * size  # 子窗口像素总数

    # Σx 与 Σx²（O(1) 查询）
    sum_x  = rect_sum   (iimg.ii,    r, c, size, size)
    sum_x2 = rect_sum_sq(iimg.ii_sq, r, c, size, size)

    # 方差公式
    mean   = sum_x  / N
    mean_sq = sum_x2 / N
    variance = mean_sq - mean ** 2

    # 浮点误差可能导致极小负值，截断为 0
    return max(0.0, variance)


# ─────────────────────────────────────────────────────────────
# 便捷包装：直接接受 IntegralImage 对象（供 haar_features.py 调用）
# ─────────────────────────────────────────────────────────────

def compute_integral_image(img: np.ndarray) -> IntegralImage:
    """
    build() 的别名，提供更直观的函数名供外部模块导入。

    haar_features.py 推荐的调用方式：
        from integral_image import compute_integral_image, rect_sum
        iimg = compute_integral_image(gray_patch)
        val  = rect_sum(iimg.ii, r, c, h, w)
    """
    return build(img)


# ─────────────────────────────────────────────────────────────
# 直接运行：简单自测（非正式，完整测试见 test.py）
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  integral_image.py 快速自测")
    print("=" * 50)

    # 构造 3×4 测试图，便于手算验证
    img_test = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ], dtype=np.float64)

    print(f"\n  测试图像:\n{img_test}")

    iimg = build(img_test)
    print(f"\n  积分图 ii (shape={iimg.ii.shape}):\n{iimg.ii}")

    # 手算：整张图像像素和 = 1+2+...+12 = 78
    total = rect_sum(iimg.ii, 0, 0, 3, 4)
    print(f"\n  全图像素和 rect_sum(0,0,3,4) = {total}  （期望 78）")

    # 左上 2×2 子块：1+2+5+6 = 14
    sub = rect_sum(iimg.ii, 0, 0, 2, 2)
    print(f"  左上 2×2 像素和 rect_sum(0,0,2,2) = {sub}  （期望 14）")

    # 右下 2×2 子块：7+8+11+12 = 38
    sub2 = rect_sum(iimg.ii, 1, 2, 2, 2)
    print(f"  右下 2×2 像素和 rect_sum(1,2,2,2) = {sub2}  （期望 38）")

    # 方差（全图）
    var = window_variance(iimg, 0, 0, 3)   # 3×3 左上子块
    print(f"  左上 3×3 子窗口方差 = {var:.4f}")

    print("\n  自测完成。运行 `python test.py --modules integral_image` 查看完整测试。")