# utils/nms.py
import numpy as np
from typing import List, Tuple

def nms(candidates: List[Tuple[int, int, int, int]], iou_threshold: float = 0.3, min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
    """
    非极大值抑制与重叠框合并
    :param candidates: [(x, y, w, h), ...]
    :param iou_threshold: 合并阈值
    :param min_votes: 最小票数，低于此值的孤立框（背景噪声）会被删除
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

    # 1. 统计每个框的"得票数"（周围有多少重叠框）
    votes = np.zeros(len(boxes))
    for i in range(len(boxes)):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)
        
        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)
        inter_area = inter_w * inter_h
        
        # 使用 IoM (Intersection over Minimum) 解决大框套小框
        overlap = inter_area / np.minimum(areas[i], areas)
        votes[i] = np.sum(overlap > iou_threshold)

    # 2. 第一次筛选：删除孤立的假阳性（票数不够的）
    keep_indices = np.where(votes >= min_votes)[0]
    if len(keep_indices) == 0:
        return []
    
    v_boxes = boxes[keep_indices]
    v_votes = votes[keep_indices]
    vx1, vy1, vx2, vy2 = x1[keep_indices], y1[keep_indices], x2[keep_indices], y2[keep_indices]
    vareas = areas[keep_indices]

    # 3. 第二阶段：按票数降序合并
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