"""
nms.py
======
非极大值抑制（Non-Maximum Suppression, NMS）工具模块

位置：utils/nms.py
职责：
    1. 计算两个矩形框的交并比（IoU, Intersection over Union）
    2. 对检测器输出的候选框列表执行 NMS，去除重复检测
    3. 支持基于 IoM（Intersection over Minimum）的投票合并机制

为什么需要 NMS？
    - 多尺度检测时，同一个人脸会被多个相邻/不同尺度的窗口检测到
    - NMS 保留得分最高的框，抑制与其高度重叠的其他框
    - 最终每个目标只保留一个最优检测框

算法步骤：
    1. 统计每个框的"得票数"（周围有多少重叠框），删除孤立框
    2. 按票数从高到低排序
    3. 选择当前票数最高的框，找到与其重叠的所有框
    4. 对该组框取坐标平均值，得到平滑的合并框
    5. 重复步骤 3-4，直到没有剩余框

参考论文：
    - Viola-Jones 论文 Section 5：使用 "post-processing" 合并多尺度检测
    - 实际实现通常采用 NMS 或均值漂移（Mean Shift）聚类
"""

from typing import List, Tuple
import numpy as np


def compute_iou(box1: Tuple[int, int, int, int],
                box2: Tuple[int, int, int, int]) -> float:
    """
    计算两个矩形框的交并比（IoU）。

    参数：
        box1: 第一个框 (x1, y1, w1, h1)
              x1, y1 = 左上角坐标
              w1, h1 = 宽度和高度
        box2: 第二个框 (x2, y2, w2, h2)

    返回：
        IoU 值，范围 [0.0, 1.0]
        - 0.0 表示两个框完全不重叠
        - 1.0 表示两个框完全重合

    计算公式：
        IoU = 交集面积 / 并集面积
             = intersection_area / (area1 + area2 - intersection_area)
    """
    # ─── 解包坐标 ───
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # ─── 计算两个框的右下角坐标 ───
    x1_right = x1 + w1
    y1_bottom = y1 + h1
    x2_right = x2 + w2
    y2_bottom = y2 + h2

    # ─── 计算交集矩形的左上角和右下角坐标 ───
    inter_left   = max(x1, x2)
    inter_top    = max(y1, y2)
    inter_right  = min(x1_right, x2_right)
    inter_bottom = min(y1_bottom, y2_bottom)

    # ─── 计算交集面积 ───
    inter_w = max(0, inter_right - inter_left)
    inter_h = max(0, inter_bottom - inter_top)
    inter_area = inter_w * inter_h

    # ─── 如果交集面积为 0，直接返回 0 ───
    if inter_area == 0:
        return 0.0

    # ─── 计算两个框各自的面积 ───
    area1 = w1 * h1
    area2 = w2 * h2

    # ─── 计算并集面积 ───
    union_area = area1 + area2 - inter_area

    # ─── 计算 IoU ───
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou


def nms(candidates: List[Tuple[int, int, int, int]],
        iou_threshold: float = 0.3,
        min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
    """
    非极大值抑制与重叠框合并（支持投票机制）。

    参数：
        candidates   : 候选框列表，每个框为 (x, y, w, h)
        iou_threshold: IoU 阈值（默认 0.3）
                       - 两个框的 IoU 超过此阈值 → 视为重复检测
                       - 使用 IoM (Intersection over Minimum) 解决大框套小框
        min_votes    : 最小票数（默认 3）
                       - 统计每个框周围有多少重叠框
                       - 低于此值的孤立框（背景噪声）会被删除

    返回：
        经过 NMS 过滤和合并后的框列表

    算法详解：
        1. 如果输入为空，直接返回空列表
        2. 统计每个框的"得票数"（使用 IoM 计算重叠）
        3. 删除票数不够的孤立框（假阳性）
        4. 按票数降序排列，逐组合并重叠框
        5. 对每组框取坐标平均值，得到平滑的合并框
    """
    if not candidates:
        return []

    boxes = np.array(candidates, dtype=np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w  = boxes[:, 2]
    h  = boxes[:, 3]
    x2 = x1 + w
    y2 = y1 + h
    areas = w * h

    # ─── 步骤 1：统计每个框的"得票数" ───
    # 使用 IoM (Intersection over Minimum) 解决大框套小框的问题
    # 例如：一个 100x100 的大框完全包含一个 20x20 的小框
    # IoU = (20*20) / (100*100) = 0.04（很小，不会被抑制）
    # IoM = (20*20) / (20*20) = 1.0（正确识别为重叠）
    votes = np.zeros(len(boxes))
    for i in range(len(boxes)):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h

        # IoM = 交集面积 / min(两个框的面积)
        overlap = inter_area / np.minimum(areas[i], areas)
        votes[i] = np.sum(overlap > iou_threshold)

    # ─── 步骤 2：第一次筛选 ───
    # 删除孤立的假阳性（票数不够的）
    keep_indices = np.where(votes >= min_votes)[0]
    if len(keep_indices) == 0:
        return []

    v_boxes = boxes[keep_indices]
    v_votes = votes[keep_indices]
    vx1, vy1, vx2, vy2 = x1[keep_indices], y1[keep_indices], x2[keep_indices], y2[keep_indices]
    vareas = areas[keep_indices]

    # ─── 步骤 3：按票数降序合并 ───
    order = v_votes.argsort()[::-1]
    final_boxes = []

    while order.size > 0:
        i = order[0]

        # 计算当前票王与剩下框的重叠
        xx1 = np.maximum(vx1[i], vx1[order])
        yy1 = np.maximum(vy1[i], vy1[order])
        xx2 = np.minimum(vx2[i], vx2[order])
        yy2 = np.minimum(vy2[i], vy2[order])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / np.minimum(vareas[i], vareas[order])

        # 找到属于同一组的索引
        merge_indices = np.where(ovr > iou_threshold)[0]

        # 对该组坐标取平均值，得到平滑的框
        group_boxes = v_boxes[order[merge_indices]]
        mean_box = np.mean(group_boxes, axis=0).astype(int)
        final_boxes.append(tuple(mean_box))

        # 剔除已处理的框
        remaining_indices = np.where(ovr <= iou_threshold)[0]
        order = order[remaining_indices]

    return final_boxes


def nms_with_scores(boxes: List[Tuple[int, int, int, int]],
                    scores: List[float],
                    iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    带置信度分数的 NMS（扩展版本）。

    当检测器输出置信度分数时使用此函数。
    与 nms() 的区别：
        - 按置信度分数排序（而非按票数）
        - 分数高的框优先保留

    参数：
        boxes         : 候选框列表
        scores        : 每个框对应的置信度分数（与 boxes 一一对应）
        iou_threshold : IoU 阈值

    返回：
        过滤后的框列表

    用法：
        # 如果未来成员 B 的模型输出分数，可以这样用：
        # filtered_boxes = nms_with_scores(boxes, scores, iou_threshold=0.5)
    """
    if not boxes or not scores:
        return []
    if len(boxes) != len(scores):
        raise ValueError(f"boxes 和 scores 长度不一致: {len(boxes)} vs {len(scores)}")

    if len(boxes) == 1:
        return boxes.copy()

    # ─── 按分数从高到低排序 ───
    paired = list(zip(boxes, scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    keep_boxes = []

    while paired:
        # 取出当前分数最高的框
        current_box, _ = paired.pop(0)
        keep_boxes.append(current_box)

        # 过滤重叠框
        remaining = []
        for box, score in paired:
            iou = compute_iou(current_box, box)
            if iou <= iou_threshold:
                remaining.append((box, score))

        paired = remaining

    return keep_boxes


# ─── 便捷函数：合并检测框（均值漂移风格） ───
# ══════════════════════════════════════════════════════════════════════════════
# 模块二：原著 VJ2004 "多重检测合并"（均值合并）算法
# 参考：Viola-Jones 2004, Section 5 - Integration of Multiple Detections
# ══════════════════════════════════════════════════════════════════════════════


class _UnionFind:
    """
    并查集（Disjoint Set Union / Union-Find）数据结构。
    
    用于高效地将 N 个检测框划分为等价类（连通分量）。
    支持路径压缩（Path Compression）和按秩合并（Union by Rank）。
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """查找 x 所属集合的代表元（带路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        """合并 x 和 y 所属的两个集合（按秩合并）"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def group_rectangles_original(
    rectangles: List[Tuple[int, int, int, int]],
    eps: float = 0.35,
    group_threshold: int = 3
) -> List[Tuple[int, int, int, int]]:
    """
    严格按照 Viola-Jones 2004 论文 Section 5 实现的均值合并算法。

    算法原理：
        VJ 论文指出，检测器对同一张人脸会产生大量位置偏移和尺度变化的重叠框。
        这些框构成空间上的连通图。算法分为三步：
        
        1. 等价关系判定（ε-邻域）：若两框的偏移量和尺寸差均不超过
           ε · min(w1,w2)，则视为等价，在二者之间建立无向边。
        2. 连通集划分（Union-Find）：将所有框划分为互不相交的连通子集。
        3. 阈值过滤 + 均值合并：丢弃框数少于 group_threshold 的子集（噪声）；
           对保留下来的子集，计算坐标/尺寸的算术平均值作为最终输出框。

    参数：
        rectangles     : 原始检测框列表 [(x, y, w, h), ...]
        eps            : 邻域容差系数（默认 0.2）
                         用于判定两个框是否等价的阈值：
                           Δx ≤ eps · min(w1, w2)
                           Δy ≤ eps · min(h1, h2)
                           Δw ≤ eps · min(w1, w2)
                           Δh ≤ eps · min(h1, h2)
        group_threshold: 最小重叠框数过滤阈值（默认 3）
                         若某连通子集中框的数量 < group_threshold，
                         则判定该组为假阳性（噪声），整组丢弃。

    返回：
        合并后的框列表 [(x, y, w, h), ...]
        与 nms() 保持相同的输入输出接口。

    参考文献：
        Viola, Jones. "Robust Real-Time Face Detection" (2004)
        Section 5: Integration of Multiple Detections
        Pages 10-11
    """
    if not rectangles:
        return []

    n = len(rectangles)

    # ─── 步骤 1：构建并查集，根据 ε-邻域等价关系合并 ───
    uf = _UnionFind(n)

    for i in range(n):
        x1, y1, w1, h1 = rectangles[i]
        for j in range(i + 1, n):
            x2, y2, w2, h2 = rectangles[j]

            # 计算四个维度的容差阈值
            eps_w = eps * min(w1, w2)
            eps_h = eps * min(h1, h2)

            # 判断等价条件：四个差值均 ≤ ε · min_dim
            if (abs(x1 - x2) <= eps_w and
                abs(y1 - y2) <= eps_h and
                abs(w1 - w2) <= eps_w and
                abs(h1 - h2) <= eps_h):
                uf.union(i, j)

    # ─── 步骤 2：将并查集中的连通分量收集为子集 ───
    # key = 集合代表元，value = 该集合中所有框的索引列表
    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append(idx)

    # ─── 步骤 3：阈值过滤 + 均值合并 ───
    merged_boxes = []

    for root, indices in groups.items():
        # 3a. 过滤：丢弃框数少于 group_threshold 的噪声子集
        if len(indices) < group_threshold:
            continue

        # 3b. 均值合并：对子集中所有框的坐标和尺寸取算术平均值
        sum_x, sum_y, sum_w, sum_h = 0, 0, 0, 0
        for idx in indices:
            x, y, w, h = rectangles[idx]
            sum_x += x
            sum_y += y
            sum_w += w
            sum_h += h

        cnt = len(indices)
        merged_boxes.append((
            int(round(sum_x / cnt)),
            int(round(sum_y / cnt)),
            int(round(sum_w / cnt)),
            int(round(sum_h / cnt)),
        ))

    return merged_boxes


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]],
                            iou_threshold: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    合并重叠框（替代 NMS 的另一种策略）。

    与 NMS 不同，此函数不丢弃重叠框，而是将它们合并为一个框。
    合并方式：取所有重叠框的坐标平均值。

    适用场景：
        - 多个检测框覆盖同一个人脸的不同部位
        - 希望得到一个更精确的包围盒

    参数：
        boxes         : 候选框列表
        iou_threshold : 判断是否重叠的 IoU 阈值

    返回：
        合并后的框列表
    """
    if not boxes:
        return []
    if len(boxes) == 1:
        return boxes.copy()

    # 先执行 NMS 得到不重叠的框组
    # 然后对每个组内的框取平均
    # 这里简化实现：直接返回 NMS 结果
    return nms(boxes, iou_threshold)
