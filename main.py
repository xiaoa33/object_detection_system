"""
main.py
=======
实时人脸检测系统 —— 主入口

位置：main.py
职责：
    1. 解析命令行参数（可选）
    2. 初始化 Detector（自动选择占位模式或真实模式）
    3. 创建 PyQt5 应用和 MainWindow
    4. 启动主事件循环

使用方式：
    # 基本运行（自动选择模式）
    python main.py

    # 强制使用占位模式（即使有 .pkl 文件）
    python main.py --placeholder

    # 强制使用真实模式（即使没有 .pkl 文件）
    python main.py --real --model models/cascade_model.pkl

    # 指定摄像头 ID
    python main.py --camera 1

    # 指定参数
    python main.py --min-face 40 --nms-threshold 0.4

系统架构：
    main.py
      ├── Detector (detect/detector.py)
      │     ├── 占位模式 → OpenCV Haar Cascade
      │     └── 真实模式 → Viola-Jones (积分图 + Haar + 级联)
      ├── MainWindow (ui/main_window.py)
      │     └── VideoThread (ui/video_thread.py)
      │           ├── Detector.detect() → 检测
      │           ├── nms() → 去重
      │           └── pyqtSignal → 更新 UI
      └── QApplication → 主事件循环

依赖：
    - PyQt5
    - OpenCV (cv2)
    - numpy
    - 项目内部模块：detect.detector, ui.main_window, ui.video_thread, utils.nms

注意事项：
    - 如果 models/cascade_model.pkl 不存在，自动使用占位模式
    - 占位模式使用 OpenCV 内置的 haarcascade_frontalface_default.xml
    - 将来成员 B 上传 .pkl 后，无需修改任何代码即可切换到真实模式
"""

import sys
import os
import argparse

# ─── 确保项目根目录在 Python 路径中 ───
# 这样无论从哪个目录运行 python main.py，都能正确导入项目模块
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args():
    """
    解析命令行参数。

    支持以下参数：
        --placeholder    : 强制使用占位模式（OpenCV 内置检测器）
        --real           : 强制使用真实模式（Viola-Jones 模型）
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
  python main.py                          # 自动模式
  python main.py --placeholder            # 强制占位模式
  python main.py --real --model models/cascade_model.pkl  # 强制真实模式
  python main.py --camera 1 --min-face 40 --nms-threshold 0.4
        """
    )

    # ─── 检测模式参数 ───
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--placeholder", action="store_true",
        help="强制使用占位模式（OpenCV 内置 Haar Cascade）"
    )
    mode_group.add_argument(
        "--real", action="store_true",
        help="强制使用真实模式（Viola-Jones 模型，需要 .pkl 文件）"
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
        2. 初始化 Detector（自动选择模式）
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

    # 确定使用哪种模式
    if args.placeholder:
        use_placeholder = True
        print("[main] 强制使用占位模式")
    elif args.real:
        use_placeholder = False
        print(f"[main] 强制使用真实模式，模型: {args.model}")
    else:
        # 自动判断
        model_path = os.path.join(_project_root, args.model)
        use_placeholder = not os.path.exists(model_path)
        if use_placeholder:
            print(f"[main] 自动模式：未找到模型文件 {args.model}，使用占位模式")
        else:
            print(f"[main] 自动模式：找到模型文件 {args.model}，使用真实模式")

    # 创建检测器
    try:
        from detect.detector import Detector
        detector = Detector(
            model_path=args.model,
            scale_factor=args.scale_factor,
            min_face_size=args.min_face,
            max_face_size=args.max_face,
            use_placeholder=use_placeholder
        )
    except Exception as e:
        print(f"[main] ❌ 初始化检测器失败: {e}")
        print("[main] 请确保已安装依赖: pip install -r requirements.txt")
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
    print(f"[main] 检测模式: {'占位模式 (OpenCV)' if detector.is_placeholder else '真实模式 (Viola-Jones)'}")
    print(f"[main] 摄像头 ID: {args.camera}")
    print(f"[main] 最小人脸尺寸: {args.min_face}px")
    print(f"[main] NMS 阈值: {args.nms_threshold}")
    print("[main] 按 Ctrl+C 或关闭窗口退出")
    print("=" * 60)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
