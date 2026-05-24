"""
main.py
=======
实时人脸检测系统 —— 主入口

位置：main.py
职责：
    1. 解析命令行参数（可选）
    2. 初始化 Detector（真实模式，加载 Viola-Jones 模型）
    3. 创建 PyQt5 应用和 MainWindow
    4. 启动主事件循环

使用方式：
    # 基本运行
    python main.py

    # 指定摄像头 ID
    python main.py --camera 1

    # 指定参数
    python main.py --min-face 40 --nms-threshold 0.4

系统架构：
    main.py
      └── Detector (detect/detector.py)
            └── Viola-Jones (积分图 + Haar + 级联)
      └── MainWindow (ui/main_window.py)
            └── VideoThread (ui/video_thread.py)
                  ├── Detector.detect() → 检测
                  ├── nms() / group_rectangles_original() → 去重
                  └── pyqtSignal → 更新 UI
      └── QApplication → 主事件循环

依赖：
    - PyQt5
    - OpenCV (cv2)
    - numpy
    - 项目内部模块：detect.detector, ui.main_window, ui.video_thread, utils.nms
"""

import sys
import os
import argparse

# ─── 确保项目根目录在 Python 路径中 ───
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    """
    解析命令行参数。

    支持以下参数：
        --model          : 模型文件路径（默认: models/cascade_model.pkl）
        --camera         : 摄像头 ID（默认: 0）
        --min-face       : 最小人脸尺寸（默认: 24）
        --max-face       : 最大人脸尺寸（默认: 500）
        --scale-factor   : 图像金字塔缩放因子（默认: 1.25，越大越快但可能漏检）
        --nms-threshold  : NMS IoU 阈值（默认: 0.5，范围 0.0~1.0）

    返回：
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="实时人脸检测系统 —— 基于 Viola-Jones 算法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python main.py                          # 默认运行
  python main.py --camera 1 --min-face 40 --nms-threshold 0.4
        """
    )

    # ─── 模型参数 ───
    parser.add_argument(
        "--model", type=str, default="models/cascade_model.pkl",
        help="级联模型文件路径（默认: models/cascade_model.pkl）"
    )

    # ─── 摄像头参数 ───
    parser.add_argument(
        "--camera", type=int, default=0,
        help="摄像头设备 ID（默认: 0，即内置摄像头）"
    )

    # ─── 检测参数 ───
    parser.add_argument(
        "--min-face", type=int, default=24,
        help="最小人脸尺寸（像素，默认: 24）"
    )
    parser.add_argument(
        "--max-face", type=int, default=500,
        help="最大人脸尺寸（像素，默认: 500）"
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.25,
        help="图像金字塔缩放因子（默认: 1.25，越大越快但可能漏检）"
    )
    parser.add_argument(
        "--nms-threshold", type=float, default=0.5,
        help="NMS IoU 阈值（默认: 0.5，范围 0.0~1.0）"
    )

    return parser.parse_args()


def main():
    """
    主函数 —— 系统入口。

    执行流程：
        1. 解析命令行参数
        2. 初始化 Detector（真实 Viola-Jones 模式）
        3. 创建 PyQt5 QApplication
        4. 创建 MainWindow（包含 VideoThread）
        5. 启动主事件循环
    """
    # ─── 步骤 1：解析命令行参数 ───
    args = parse_args()

    # ─── 步骤 2：初始化 Detector ───
    print("=" * 60)
    print("  实时人脸检测系统 v1.0")
    print("  基于 Viola-Jones 算法复现")
    print("=" * 60)

    model_path = os.path.join(_project_root, args.model)
    print(f"[main] 加载模型: {model_path}")

    try:
        from detect.detector import Detector
        detector = Detector(
            model_path=model_path,
            scale_factor=args.scale_factor,
            min_face_size=args.min_face,
            max_face_size=args.max_face
        )
    except Exception as e:
        print(f"[main] ❌ 初始化检测器失败: {e}")
        print("[main] 请确保已训练模型并安装依赖: pip install -r requirements.txt")
        sys.exit(1)

    # ─── 步骤 3：创建 PyQt5 应用 ───
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("实时人脸检测系统")

    # ─── 步骤 4：创建主窗口 ───
    from ui.main_window import MainWindow
    window = MainWindow(
        detector=detector,
        camera_id=args.camera
    )
    window.show()

    # ─── 步骤 5：启动主事件循环 ───
    print("[main] 系统启动成功！")
    print(f"[main] 检测模式: Viola-Jones 真实模式")
    print(f"[main] 模型文件: {model_path}")
    print(f"[main] 摄像头 ID: {args.camera}")
    print(f"[main] 最小人脸尺寸: {args.min_face}px")
    print(f"[main] NMS 阈值: {args.nms_threshold}")
    print("[main] 按 Ctrl+C 或关闭窗口退出")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
