"""
video_thread.py
===============
视频采集与检测线程模块

位置：ui/video_thread.py
职责：
    1. 在独立线程中读取摄像头帧（避免阻塞 UI 主线程）
    2. 对每帧调用 Detector.detect() 进行人脸检测
    3. 调用 NMS 合并重复检测框
    4. 在帧上绘制检测框和 FPS 信息
    5. 通过 PyQt5 信号将处理后的帧发送到主界面显示

为什么使用 QThread？
    - PyQt5 的主线程负责 UI 事件循环（鼠标点击、窗口绘制等）
    - 如果摄像头读取和检测也在主线程执行，UI 会卡顿
    - QThread 在后台线程执行耗时操作，通过信号（Signal）安全地传递数据到主线程

工作流程：
    OpenCV 读取摄像头 → 灰度转换 → Detector.detect() → NMS → 绘制框 → pyqtSignal → UI 显示

依赖：
    - PyQt5.QtCore.QThread, QThread, pyqtSignal
    - OpenCV (cv2)
    - detect.detector.Detector
    - utils.nms.nms
"""

import cv2
import time
import numpy as np
from typing import List, Tuple

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

# ─── 导入项目模块 ───
from detect.detector import Detector
from utils.nms import nms, group_rectangles_original


def draw_detection_frame(frame: np.ndarray,
                         boxes: List[Tuple[int, int, int, int]],
                         fps: float = 0.0,
                         detect_enabled: bool = True) -> np.ndarray:
    """
    在帧上绘制检测框和状态信息（模块级函数，供 VideoThread 和 MainWindow 共用）。

    参数：
        frame          : BGR 格式图像（原地修改）
        boxes          : 人脸框列表 [(x, y, w, h), ...]
        fps            : 帧率（0 表示静态图像）
        detect_enabled : 检测是否启用（控制状态文字颜色）

    返回：
        绘制后的帧（即输入的 frame）
    """
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Face", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status_color = (0, 255, 0) if detect_enabled else (0, 0, 255)
    cv2.putText(frame, f"Detection: {'ON' if detect_enabled else 'OFF'}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, f"Faces: {len(boxes)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


class VideoThread(QThread):
    """
    视频采集与检测线程。

    在后台线程中循环执行：
        读取帧 → 检测 → NMS → 绘制 → 发射信号

    通过信号与主界面通信：
        frame_signal: 发送处理后的帧（QImage 格式）给 UI 显示
        stats_signal: 发送检测统计信息（FPS、检测人数等）
    """

    # ─── PyQt5 信号定义 ───
    # frame_signal: 发送 QImage 给主界面显示
    #   参数: QImage（RGB 格式的图像数据）
    frame_signal = pyqtSignal(QImage)

    # stats_signal: 发送检测统计信息
    #   参数: (fps, face_count, frame_width, frame_height)
    #         fps        - 当前帧率（帧/秒）
    #         face_count - 当前帧检测到的人脸数量
    #         frame_w    - 帧宽度（像素）
    #         frame_h    - 帧高度（像素）
    stats_signal = pyqtSignal(float, int, int, int)

    def __init__(self, detector: Detector, camera_id: int = 0, parent=None):
        """
        初始化视频线程。

        参数：
            detector  : Detector 实例（占位模式或真实模式均可）
            camera_id : 摄像头设备 ID
                        - 0 = 默认摄像头（通常是笔记本内置摄像头）
                        - 1, 2, ... = 外接摄像头
            parent    : 父 QObject（通常传入 MainWindow）
        """
        super().__init__(parent)
        self.detector = detector
        self.camera_id = camera_id

        # ─── 运行时状态 ───
        self._running = False          # 线程是否在运行
        self._detect_enabled = True    # 是否执行检测（可开关）

        # ─── 可调参数（UI 通过 setter 方法修改） ───
        self._nms_threshold = 0.5      # NMS 的 IoU 阈值
        self._min_face_size = 24       # 最小人脸尺寸（像素）
        self._max_face_size = 500      # 最大人脸尺寸（像素）
        self._min_votes = 2            # 最少重叠票数/邻近框数
        self._scale_factor = 1.25      # 金字塔缩放因子

        # ─── FPS 计算相关 ───
        self._fps = 0.0                # 当前帧率
        self._frame_count = 0          # 已处理帧数
        self._fps_start_time = time.time()  # FPS 统计起始时间

        # ═══ 性能优化：跳帧检测 ═══
        # 每 DETECT_INTERVAL 帧才做一次检测，中间帧复用上次结果
        # 这样检测速度翻倍，但显示仍然流畅
        self._detect_interval = 2      # 每 2 帧检测 1 次
        self._frame_counter = 0        # 帧计数器
        self._last_face_boxes = []     # 上次检测结果缓存

        # ═══ 检测精度（步长系数） ═══
        # step_delta 控制滑动窗口步长：
        #   1.0 = 精度优先（窗口密，检测慢）
        #   1.5 = 平衡模式（默认）
        #   2.0 = 速度优先（窗口疏，检测快）
        self._step_delta = 1.5

        # ═══ 原著 VJ2004 步长因子 ═══
        # 严格遵循论文 Section 4.1: step = max(1, round(scale * step_factor))
        # 默认 3.0（等效于旧版 step_delta=1.5，保证性能不退化）
        # 旧公式: step = round(scale * step_delta * 2)
        # 新公式: step = round(scale * step_factor)
        self._step_factor = 3.0

        # ═══ 后处理算法模式 ═══
        # 0 = 现代 IoU NMS（nms 函数）
        # 1 = 原著均值合并（group_rectangles_original 函数）
        self._nms_mode = 0

    # ─── 属性访问器（供 UI 调用） ───

    @property
    def nms_threshold(self) -> float:
        """获取当前 NMS 阈值"""
        return self._nms_threshold

    @nms_threshold.setter
    def nms_threshold(self, value: float):
        """
        设置 NMS 阈值。
        参数 value: 0.0 ~ 1.0 之间的浮点数
        """
        self._nms_threshold = max(0.0, min(1.0, value))

    @property
    def min_face_size(self) -> int:
        """获取当前最小人脸尺寸"""
        return self._min_face_size

    @min_face_size.setter
    def min_face_size(self, value: int):
        """
        设置最小人脸尺寸。
        参数 value: 像素值，通常 20 ~ 200
        """
        self._min_face_size = max(10, min(500, value))

    @property
    def max_face_size(self) -> int:
        """获取当前最大人脸尺寸"""
        return self._max_face_size

    @max_face_size.setter
    def max_face_size(self, value: int):
        """
        设置最大人脸尺寸。
        参数 value: 像素值，通常 100 ~ 1000
        """
        self._max_face_size = max(50, min(2000, value))

    @property
    def min_votes(self) -> int:
        """获取当前最少重叠票数"""
        return self._min_votes

    @min_votes.setter
    def min_votes(self, value: int):
        """
        设置最少重叠票数/邻近框数。
        参数 value: 1 ~ 10
        """
        self._min_votes = max(1, min(300, value))

    @property
    def scale_factor(self) -> float:
        """获取金字塔缩放因子"""
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value: float):
        """
        设置金字塔缩放因子。
        参数 value: 1.05 ~ 2.00
        """
        self._scale_factor = max(1.01, min(3.0, value))

    @property
    def detect_enabled(self) -> bool:
        """检测是否启用"""
        return self._detect_enabled

    @detect_enabled.setter
    def detect_enabled(self, enabled: bool):
        """
        启用/禁用检测。
        禁用时只显示视频流，不画检测框。
        """
        self._detect_enabled = enabled

    @property
    def step_delta(self) -> float:
        """获取当前检测步长系数"""
        return self._step_delta

    @step_delta.setter
    def step_delta(self, value: float):
        """
        设置检测步长系数。
        参数 value: 1.0（精度优先）~ 2.0（速度优先）
        """
        self._step_delta = max(1.0, min(2.0, value))
        # 同步到检测器
        self.detector.step_delta = self._step_delta

    @property
    def step_factor(self) -> float:
        """获取原著 VJ2004 步长因子 Δ"""
        return self._step_factor

    @step_factor.setter
    def step_factor(self, value: float):
        """
        设置原著 VJ2004 步长因子 Δ。
        参数 value: 1.0（高精度）~ 5.0（高速度）
        """
        self._step_factor = max(0.5, min(5.0, value))
        # 同步到检测器
        self.detector.step_factor = self._step_factor

    @property
    def nms_mode(self) -> int:
        """获取后处理算法模式：0=NMS, 1=原著均值合并"""
        return self._nms_mode

    @nms_mode.setter
    def nms_mode(self, value: int):
        """
        设置后处理算法模式。
        参数 value: 0 = 现代 IoU NMS, 1 = 原著均值合并
        """
        self._nms_mode = 1 if value == 1 else 0

    # ─── 线程控制 ───

    def start(self, priority: QThread.Priority = QThread.NormalPriority):
        """
        启动视频采集线程。
        在调用 start() 后，run() 方法会在新线程中执行。
        """
        self._running = True
        self._frame_count = 0
        self._fps_start_time = time.time()
        super().start(priority)
        print(f"[VideoThread] 线程已启动，摄像头 ID={self.camera_id}")

    def stop(self):
        """
        停止视频采集线程。
        设置 _running = False，run() 中的循环会自然退出。
        """
        self._running = False
        print("[VideoThread] 线程正在停止...")

    def run(self):
        """
        线程主循环（在后台线程中执行）。

        流程：
            1. 打开摄像头
            2. 循环读取帧直到 _running = False：
               a. 读取一帧
               b. 如果检测启用：调用 Detector.detect() + NMS
               c. 在帧上绘制检测框和 FPS
               d. 将帧转换为 QImage 并通过信号发送
               e. 更新 FPS 统计
            3. 释放摄像头资源
        """
        # ─── 打开摄像头 ───
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[VideoThread] 无法打开摄像头 ID={self.camera_id}")
            # 发送一个空信号通知 UI
            self.stats_signal.emit(0.0, 0, 0, 0)
            return

        # 设置摄像头属性（可选）
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 设置宽度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
        cap.set(cv2.CAP_PROP_FPS, 30)             # 设置目标帧率

        print(f"[VideoThread] 摄像头已打开，分辨率: "
              f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        # ─── 主循环 ───
        while self._running:
            # 步骤 a：读取一帧
            ret, frame = cap.read()
            if not ret:
                print("[VideoThread] ⚠ 读取帧失败，尝试重新连接...")
                time.sleep(0.1)
                continue

            # 保存原始帧尺寸用于统计信号
            frame_h, frame_w = frame.shape[:2]

            # 步骤 b：人脸检测 + NMS
            face_boxes = []
            if self._detect_enabled:
                # 更新检测器参数（与 UI 同步）
                self.detector.min_face_size = self._min_face_size
                self.detector.max_face_size = self._max_face_size
                self.detector.scale_factor = self._scale_factor

                # ═══ 性能优化：跳帧检测 ═══
                # 每 _detect_interval 帧才做一次检测，中间帧复用上次结果
                # 这样检测速度翻倍，但显示仍然流畅
                self._frame_counter += 1
                if self._frame_counter % self._detect_interval == 0:
                    # ═══ 性能优化：缩小检测图像 ═══
                    # 原始帧 640×480 → 缩小到 480×360（面积缩小约 44%）
                    # 在保持可检测最小人脸 24px 的前提下平衡速度与精度
                    # 若原始人脸过小（< 32px 在原始帧中），缩小后可能低于 24px 最小检测窗口
                    DETECT_WIDTH = 480
                    DETECT_HEIGHT = 360
                    detect_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
                    gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

                    # 调用检测器（内部已包含 NMS 后处理）
                    small_boxes = self.detector.detect(
                        gray,
                        iou_threshold=self._nms_threshold,
                        min_votes=self._min_votes
                    )

                    # 将检测框坐标从缩小后的图像还原到原始尺寸
                    scale_x = frame.shape[1] / DETECT_WIDTH
                    scale_y = frame.shape[0] / DETECT_HEIGHT
                    self._last_face_boxes = [
                        (int(x * scale_x), int(y * scale_y),
                         int(w * scale_x), int(h * scale_y))
                        for (x, y, w, h) in small_boxes
                    ]

                face_boxes = self._last_face_boxes

            # 步骤 c：在帧上绘制检测框和 FPS
            display_frame = self._draw_detection(frame, face_boxes)

            # 步骤 d：转换为 QImage 并通过信号发送
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
            self.frame_signal.emit(qt_image)

            # 步骤 e：发送统计信息
            self.stats_signal.emit(self._fps, len(face_boxes), frame_w, frame_h)

            # 更新 FPS 统计
            self._frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:  # 每秒更新一次 FPS
                self._fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = time.time()

        # ─── 释放资源 ───
        cap.release()
        print("[VideoThread] 线程已停止，摄像头已释放")

    def _draw_detection(self, frame: np.ndarray,
                        boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        return draw_detection_frame(frame, boxes, self._fps, self._detect_enabled)
