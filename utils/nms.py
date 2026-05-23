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
