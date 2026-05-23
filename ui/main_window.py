"""
main_window.py
==============
主界面窗口模块

位置：ui/main_window.py
职责：
    1. 创建 PyQt5 主窗口，包含视频显示和参数控制面板
    2. 左侧：实时视频显示区域（QLabel）
    3. 右侧：参数控制面板
       - NMS 阈值滑动条（0.0 ~ 1.0）
       - 最小人脸尺寸滑动条（20 ~ 300 像素）
       - FPS 和检测人数实时显示
       - 检测启停按钮
    4. 连接 VideoThread 的信号，实时更新界面

布局说明：
    ┌──────────────────────────────────────────────┐
    │  实时人脸检测系统 v1.0                         │
    ├──────────────────────┬───────────────────────┤
    │                      │    参数控制            │
    │                      │  ─────────────         │
    │   视频显示区域        │  NMS 阈值: [═══●══] 0.5  │
    │   (QLabel)           │  最小人脸: [══●═══] 80   │
    │                      │                       │
    │                      │  FPS: 30.5            │
    │                      │  检测人数: 2           │
    │                      │                       │
    │                      │  [■ 停止检测]          │
    │                      │  [  退出程序  ]        │
    └──────────────────────┴───────────────────────┘

依赖：
    - PyQt5.QtWidgets (QMainWindow, QWidget, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout)
    - PyQt5.QtCore (Qt, QTimer)
    - PyQt5.QtGui (QImage, QPixmap)
    - ui.video_thread.VideoThread
"""

import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QSlider, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QApplication, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

# ─── 导入项目模块 ───
from ui.video_thread import VideoThread
from detect.detector import Detector


class MainWindow(QMainWindow):
    """
    实时人脸检测系统主窗口。

    功能：
        - 显示摄像头实时视频流
        - 实时调节 NMS 阈值和最小人脸尺寸
        - 显示 FPS 和检测人数
        - 启停检测
        - 优雅退出（释放摄像头资源）
    """

    def __init__(self, detector: Detector, camera_id: int = 0):
        """
        初始化主窗口。

        参数：
            detector  : Detector 实例（占位模式或真实模式均可）
            camera_id : 摄像头设备 ID（默认 0）
        """
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id

        # ─── 视频线程（在 start_detection 中创建） ───
        self.video_thread: Optional[VideoThread] = None

        # ─── 初始化 UI ───
        self._init_ui()

        # ─── 启动检测 ───
        self.start_detection()

    def _init_ui(self):
        """
        初始化用户界面。

        创建左侧视频显示区域和右侧控制面板。
        """
        self.setWindowTitle("实时人脸检测系统 v1.0")
        self.setMinimumSize(960, 600)

        # ─── 中央部件 ───
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # ─── 主布局：水平排列（左：视频 | 右：控制面板） ───
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ═══════════════════════════════════════════════
        # 左侧：视频显示区域
        # ═══════════════════════════════════════════════
        video_group = QGroupBox("视频显示")
        video_layout = QVBoxLayout(video_group)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 8px;
                color: #ffffff;
                font-size: 16px;
            }
        """)
        self.video_label.setText("等待摄像头启动...")
        video_layout.addWidget(self.video_label)

        main_layout.addWidget(video_group, stretch=3)

        # ═══════════════════════════════════════════════
        # 右侧：控制面板
        # ═══════════════════════════════════════════════
        control_group = QGroupBox("⚙ 参数控制")
        control_layout = QVBoxLayout(control_group)
        control_layout.setSpacing(15)

        # ─── 1. NMS 阈值滑动条 ───
        nms_group = QVBoxLayout()
        nms_label = QLabel("NMS 阈值 (IoU)")
        nms_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.nms_value_label = QLabel("0.50")
        self.nms_value_label.setAlignment(Qt.AlignRight)
        self.nms_value_label.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")

        nms_header = QHBoxLayout()
        nms_header.addWidget(nms_label)
        nms_header.addStretch()
        nms_header.addWidget(self.nms_value_label)

        self.nms_slider = QSlider(Qt.Horizontal)
        self.nms_slider.setRange(0, 100)          # 0 ~ 100 对应 0.00 ~ 1.00
        self.nms_slider.setValue(50)               # 默认 0.50
        self.nms_slider.setTickPosition(QSlider.TicksBelow)
        self.nms_slider.setTickInterval(10)
        self.nms_slider.valueChanged.connect(self._on_nms_changed)

        nms_group.addLayout(nms_header)
        nms_group.addWidget(self.nms_slider)
        control_layout.addLayout(nms_group)

        # ─── 2. 最小人脸尺寸滑动条 ───
        face_size_group = QVBoxLayout()
        face_size_label = QLabel("最小人脸尺寸 (像素)")
        face_size_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.face_size_value_label = QLabel("24")
        self.face_size_value_label.setAlignment(Qt.AlignRight)
        self.face_size_value_label.setStyleSheet("color: #00ff00; font-size: 14px; font-weight: bold;")

        face_size_header = QHBoxLayout()
        face_size_header.addWidget(face_size_label)
        face_size_header.addStretch()
        face_size_header.addWidget(self.face_size_value_label)

        self.face_size_slider = QSlider(Qt.Horizontal)
        self.face_size_slider.setRange(20, 300)    # 20 ~ 300 像素
        self.face_size_slider.setValue(24)          # 默认 24（Viola-Jones 基础窗口大小）
        self.face_size_slider.setTickPosition(QSlider.TicksBelow)
        self.face_size_slider.setTickInterval(20)
        self.face_size_slider.valueChanged.connect(self._on_face_size_changed)

        face_size_group.addLayout(face_size_header)
        face_size_group.addWidget(self.face_size_slider)
        control_layout.addLayout(face_size_group)

        # ─── 分隔线 ───
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator)

        # ─── 3. 状态显示 ───
        status_group = QVBoxLayout()
        status_title = QLabel("实时状态")
        status_title.setFont(QFont("Arial", 10, QFont.Bold))

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
        self.fps_label.setFont(QFont("Consolas", 14))

        self.face_count_label = QLabel("检测人数: 0")
        self.face_count_label.setStyleSheet("color: #00ff00; font-size: 16px; font-weight: bold;")
        self.face_count_label.setFont(QFont("Consolas", 14))

        self.resolution_label = QLabel("分辨率: -")
        self.resolution_label.setStyleSheet("color: #aaaaaa; font-size: 12px;")

        status_group.addWidget(status_title)
        status_group.addWidget(self.fps_label)
        status_group.addWidget(self.face_count_label)
        status_group.addWidget(self.resolution_label)
        control_layout.addLayout(status_group)

        # ─── 分隔线 ───
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        control_layout.addWidget(separator2)

        # ─── 4. 按钮区域 ───
        button_group = QVBoxLayout()
        button_group.setSpacing(10)

        # 检测启停按钮
        self.toggle_button = QPushButton("■ 停止检测")
        self.toggle_button.setMinimumHeight(40)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:checked {
                background-color: #27ae60;
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._on_toggle_detection)

        # 退出按钮
        self.exit_button = QPushButton("✕ 退出程序")
        self.exit_button.setMinimumHeight(40)
        self.exit_button.setStyleSheet("""
            QPushButton {
                background-color: #7f8c8d;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #95a5a6;
            }
        """)
        self.exit_button.clicked.connect(self._on_exit)

        button_group.addWidget(self.toggle_button)
        button_group.addWidget(self.exit_button)
        control_layout.addLayout(button_group)

        # ─── 弹性空间（将控件推到顶部） ───
        control_layout.addStretch()

        # ─── 检测模式提示 ───
        mode_text = "占位模式 (OpenCV)" if self.detector.is_placeholder else "真实模式 (Viola-Jones)"
        mode_label = QLabel(f"检测模式: {mode_text}")
        mode_label.setStyleSheet("color: #f39c12; font-size: 11px; font-style: italic;")
        mode_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(mode_label)

        main_layout.addWidget(control_group, stretch=1)

        # ─── 设置窗口样式 ───
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f23;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #e0e0e0;
                border: 2px solid #2d2d5e;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2d2d5e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ff00;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #00aa00;
                border-radius: 3px;
            }
        """)

    # ─── 信号槽：参数变化 ───

    def _on_nms_changed(self, value: int):
        """
        NMS 阈值滑动条值变化时的回调。

        将滑动条的值（0~100）映射到 NMS 阈值（0.00~1.00），
        并实时更新 VideoThread 的参数。
        """
        threshold = value / 100.0
        self.nms_value_label.setText(f"{threshold:.2f}")

        if self.video_thread is not None:
            self.video_thread.nms_threshold = threshold

    def _on_face_size_changed(self, value: int):
        """
        最小人脸尺寸滑动条值变化时的回调。

        实时更新 VideoThread 的最小人脸尺寸参数。
        """
        self.face_size_value_label.setText(str(value))

        if self.video_thread is not None:
            self.video_thread.min_face_size = value

    def _on_toggle_detection(self, checked: bool):
        """
        检测启停按钮点击回调。

        切换检测的启用/禁用状态。
        """
        if self.video_thread is not None:
            self.video_thread.detect_enabled = not checked
            if checked:
                # 当前是停止状态
                self.toggle_button.setText("▶ 开始检测")
                self.toggle_button.setStyleSheet("""
                    QPushButton {
                        background-color: #27ae60;
                        color: white;
                        font-size: 14px;
                        font-weight: bold;
                        border-radius: 6px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #2ecc71;
                    }
                """)
            else:
                # 当前是运行状态
                self.toggle_button.setText("■ 停止检测")
                self.toggle_button.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        color: white;
                        font-size: 14px;
                        font-weight: bold;
                        border-radius: 6px;
                        padding: 8px;
                    }
                    QPushButton:hover {
                        background-color: #c0392b;
                    }
                """)

    # ─── 视频线程管理 ───

    def start_detection(self):
        """
        启动视频检测线程。

        创建 VideoThread 实例，连接信号，启动线程。
        """
        # 如果已有线程在运行，先停止
        if self.video_thread is not None:
            self.stop_detection()

        # 创建新线程
        self.video_thread = VideoThread(
            detector=self.detector,
            camera_id=self.camera_id,
            parent=self
        )

        # 连接信号
        self.video_thread.frame_signal.connect(self._update_frame)
        self.video_thread.stats_signal.connect(self._update_stats)

        # 同步当前参数
        self.video_thread.nms_threshold = self.nms_slider.value() / 100.0
        self.video_thread.min_face_size = self.face_size_slider.value()

        # 启动线程
        self.video_thread.start()

    def stop_detection(self):
        """
        停止视频检测线程。

        安全地停止线程并等待其结束。
        """
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait(2000)  # 最多等待 2 秒
            self.video_thread = None

    # ─── 信号槽：更新 UI ───

    def _update_frame(self, qimage: QImage):
        """
        更新视频帧显示。

        由 VideoThread.frame_signal 触发，在主线程中执行。

        参数：
            qimage: RGB 格式的 QImage（已绘制检测框和 FPS）
        """
        pixmap = QPixmap.fromImage(qimage)

        # 缩放以适应 QLabel 的大小，保持宽高比
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _update_stats(self, fps: float, face_count: int, frame_w: int, frame_h: int):
        """
        更新统计信息显示。

        由 VideoThread.stats_signal 触发，在主线程中执行。

        参数：
            fps        : 当前帧率
            face_count : 当前帧检测到的人脸数
            frame_w    : 帧宽度
            frame_h    : 帧高度
        """
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.face_count_label.setText(f"检测人数: {face_count}")
        self.resolution_label.setText(f"分辨率: {frame_w} × {frame_h}")

    # ─── 窗口事件 ───

    def closeEvent(self, event):
        """
        窗口关闭事件。

        在关闭窗口前安全地停止视频线程。
        """
        self.stop_detection()
        event.accept()

    def _on_exit(self):
        """
        退出程序按钮回调。
        """
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出实时人脸检测系统吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.stop_detection()
            QApplication.quit()
