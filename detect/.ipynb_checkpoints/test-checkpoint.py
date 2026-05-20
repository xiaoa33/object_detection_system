import cv2
import os
from detect.detector import Detector
from detect.cascade_classifier import CascadeClassifier

# 1. 在 test.py 中定义这个函数
def non_max_suppression(rects, overlapThresh=0.3):
    if len(rects) == 0:
        return []
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=float)
    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2) # 如果有置信度，按置信度排序更好

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    return [rects[i] for i in pick]

# 2. 在检测循环之后使用它
# 原本的代码可能是：for r in rects: draw_rect(r)

    
def test_single_image(image_path, model_path):
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误：找不到图片 {image_path}")
        return
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}，请确保已训练模型")
        return

    # 2. 初始化分类器 (加载模型)
    print(f"[Test] 正在加载模型: {model_path} ...")
    cascade = CascadeClassifier(model_path)
    
    # 3. 初始化检测器
    detector = Detector(cascade)
    
    # 4. 读取图片
    img = cv2.imread(image_path)
    print(f"[Test] 成功读取图片，尺寸: {img.shape}")
    
    # 5. 执行检测
    print("[Test] 开始检测...")
    candidates = detector.detect(img)
    candidates = non_max_suppression(candidates, overlapThresh=0.3)
    
    # 6. 输出结果
    print(f"[Test] 检测完成！共找到 {len(candidates)} 个候选框。")
    for i, box in enumerate(candidates):
        x, y, w, h = box
        print(f"  框 {i+1}: 左上角({x}, {y}), 宽{w}, 高{h}")
        # 可选：绘制并在窗口显示
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if len(candidates) > 0:
        cv2.imwrite("output_result.jpg", img)
        print("图片已保存为 output_result.jpg，请在左侧文件树中双击查看。")

if __name__ == "__main__":
    # 请确保这里有你的一张测试图片
    test_image = "D:\cv\大作业\image.jpg" 
    model_file = "models/cascade_model.pkl"
    
    test_single_image(test_image, model_file)