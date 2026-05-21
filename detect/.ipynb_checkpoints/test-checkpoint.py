import cv2
import os
from detect.detector import Detector
from detect.cascade_classifier import CascadeClassifier
import numpy as np

    
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
        # 5. 执行检测 (这里可以手动传入 NMS 参数进行调优)
    print("[Test] 开始检测...")
    
    # min_votes 越大，过滤越严格（图1的背景框就会消失）
    # iou_threshold 越小，合并越积极（图2的重叠框就会变成一个大框）
    candidates = detector.detect(img, iou_threshold=0.2, min_votes=5)
    
    
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