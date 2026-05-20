import numpy as np

def nms(candidates, overlap_thresh=0.3):
    """
    非极大值抑制
    candidates: [(x, y, w, h), ...]
    overlap_thresh: 重叠度阈值，超过则判定为同一个目标
    """
    if len(candidates) == 0:
        return []

    # 转换为 numpy 数组: [x1, y1, x2, y2]
    boxes = np.array([[c[0], c[1], c[0]+c[2], c[1]+c[3]] for c in candidates])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    
    # 按照 y2 坐标排序（或者置信度，这里假设没有置信度则按面积排序）
    idxs = np.argsort(y2)
    
    keep = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        keep.append(candidates[i])
        
        # 计算当前框与剩余框的 IoU
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[:last]]
        
        # 删除重叠度过高的
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
    return keep