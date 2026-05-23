"""
main_window.py
==============
主界面窗口模块 — Instagram 风格 UI

位置：ui/main_window.py
职责：
    1. 创建 PyQt5 主窗口，包含视频显示和参数控制面板
    2. 左侧：实时视频显示区域（QLabel）
    3. 右侧：参数控制面板（Ins 暗色风格）
       - NMS 阈值滑动条（0.0 ~ 1.0）
       - 最小人脸尺寸滑动条（20 ~ 300 像素）
       - 检测精度下拉框（真实模式）
       - FPS 和检测人数实时显示
       - 检测启停按钮
    4. 连接 VideoThread 的信号，实时更新界面

布局说明：
    ┌──────────────────────────────────────────────┐
    │  📷 FACE DETECT · INSIGHT                    │
    ├──────────────────────┬───────────────────────┤
    │                      │  ⚙ CONTROL            │
    │   视频显示区域        │  ─────────────         │
    │   (圆角毛玻璃)        │  NMS THRESHOLD  0.50   │
    │                      │  [═══●══════════]      │
    │                      │  MIN FACE SIZE  24     │
    │                      │  [══●═══════════]      │
    │                      │                       │
    │                      │  ● FPS  30.5          │
    │                      │  ● FACES  2           │
    │                      │                       │
    │                      │  [ ■ STOP ]           │
    │                      │  [  EXIT  ]           │
    └──────────────────────┴───────────────────────┘

依赖：
    - PyQt5.QtWidgets (QMainWindow, QWidget, QLabel, QSlider, QPushButton, QVBoxLayout, QHBoxLayout)
    - PyQt5.QtCore (Qt, QTimer)
    - PyQt5.QtGui (QImage, QPixmap, QFont, QLinearGradient, QPalette, QBrush)
    - ui.video_thread.VideoThread
"""

import sys
from typing import Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QSlider, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout,
    QApplication, QMessageBox, QFrame, QComboBox,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QFont, QFontDatabase,
    QColor, QPalette, QLinearGradient, QBrush, QGradient
)

# ─── 导入项目模块 ───
from ui.video_thread import VideoThread
from detect.detector import Detector


# ═══════════════════════════════════════════════════════════
# 清新风格颜色常量
# ═══════════════════════════════════════════════════════════
# 主色调：薄荷绿 + 天蓝（清新自然）
CLR_PRIMARY = "#2ecc71"      # 薄荷绿
CLR_SECONDARY = "#3498db"    # 天蓝
CLR_ACCENT = "#1abc9c"       # 青绿
CLR_WARM = "#f39c12"         # 暖橙（点缀）

# 背景色
BG_MAIN = "#f0f4f8"         # 浅灰蓝主背景
BG_CARD = "#ffffff"          # 纯白卡片
BG_INPUT = "#f7f9fc"         # 浅灰输入区域

# 文字色
TEXT_PRIMARY = "#2c3e50"     # 深蓝灰主文字
TEXT_SECONDARY = "#7f8c8d"   # 灰绿次要文字
TEXT_ACCENT = "#2ecc71"      # 薄荷绿强调

# 状态色
STATUS_ON = "#2ecc71"        # 检测中（薄荷绿）
STATUS_OFF = "#e74c3c"       # 停止（柔和红）


class MainWindow(QMainWindow):
    """
    实时人脸检测系统主窗口 — Ins 暗色风格。
    """

    def __init__(self, detector: Detector, camera_id: int = 0):
        super().__init__()
        self.detector = detector
        self.camera_id = camera_id
        self.video_thread: Optional[VideoThread] = None
        self._init_ui()
        self.start_detection()

    # ═══════════════════════════════════════════════════════
    # UI 初始化
    # ═══════════════════════════════════════════════════════

    def _init_ui(self):
        """初始化 Ins 风格用户界面"""
        self.setWindowTitle("实时人脸检测系统 · INSIGHT")
        self.setMinimumSize(1100, 680)

        # ─── 全局字体 ───
        # 使用微软雅黑（黑体风格），清晰大方
        font = QFont("Microsoft YaHei", 14)
        font.setHintingPreference(QFont.PreferNoHinting)
        self.setFont(font)

        # ─── 中央部件 ───
        central = QWidget()
        self.setCentralWidget(central)

        # ─── 主布局 ───
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # ═══════════════════════════════════════════════════
        # 左侧：视频显示区域（毛玻璃卡片）
        # ═══════════════════════════════════════════════════
        video_card = QFrame()
        video_card.setObjectName("videoCard")
        video_card.setStyleSheet(f"""
            QFrame#videoCard {{
                background-color: {BG_CARD};
                border: 1px solid #e0e6ed;
                border-radius: 16px;
            }}
        """)
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(12, 12, 12, 12)

        # 视频标签
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("◉ 等待摄像头启动...")
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1a1a2e;
                border: none;
                border-radius: 12px;
                color: #8899aa;
                font-size: 14px;
                font-weight: 300;
            }}
        """)
        video_layout.addWidget(self.video_label)

        main_layout.addWidget(video_card, stretch=3)

        # ═══════════════════════════════════════════════════
        # 右侧：控制面板（Ins 风格卡片）
        # ═══════════════════════════════════════════════════
        control_card = QFrame()
        control_card.setObjectName("controlCard")
        control_card.setStyleSheet(f"""
            QFrame#controlCard {{
                background-color: {BG_CARD};
                border: 1px solid #e0e6ed;
                border-radius: 16px;
            }}
        """)
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 24, 20, 24)
        control_layout.setSpacing(18)

        # ─── 标题 ───
        title_label = QLabel("⚙ 参数控制")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 3px;
                padding: 0px;
            }}
        """)
        control_layout.addWidget(title_label)

        # ─── 分隔线 ───
        control_layout.addWidget(self._make_divider())

        # ─── 1. NMS 阈值 ───
        self._nms_value_label = self._make_value_label("0.50")
        control_layout.addLayout(self._build_slider_group(
            label_text="NMS 阈值 (IoU)",
            value_label=self._nms_value_label,
            slider=self._make_slider(0, 100, 50, self._on_nms_changed),
            suffix=""
        ))

        # ─── 2. 最小人脸尺寸 ───
        self._face_size_value_label = self._make_value_label("24")
        control_layout.addLayout(self._build_slider_group(
            label_text="最小人脸尺寸",
            value_label=self._face_size_value_label,
            slider=self._make_slider(20, 300, 24, self._on_face_size_changed),
            suffix="px"
        ))

        # ─── 分隔线 ───
        control_layout.addWidget(self._make_divider())

        # ─── 3. 检测精度（仅真实模式） ───
        if not self.detector.is_placeholder:
            precision_group = QVBoxLayout()
            precision_group.setSpacing(8)

            precision_header = QHBoxLayout()
            precision_label = QLabel("检测精度")
            precision_label.setStyleSheet(f"""
                QLabel {{
                    color: {TEXT_SECONDARY};
                    font-size: 10px;
                    font-weight: 600;
                    letter-spacing: 2px;
                }}
            """)
            precision_header.addWidget(precision_label)
            precision_header.addStretch()
            precision_group.addLayout(precision_header)

            self.precision_combo = QComboBox()
            self.precision_combo.addItem("🌱  速度优先 (step=2.0)", 2.0)
            self.precision_combo.addItem("🌿  平衡模式 (step=1.5)", 1.5)
            self.precision_combo.addItem("🌳  精度优先 (step=1.0)", 1.0)
            self.precision_combo.setCurrentIndex(1)
            self.precision_combo.setStyleSheet(f"""
                QComboBox {{
                    background-color: {BG_INPUT};
                    color: {TEXT_PRIMARY};
                    font-size: 12px;
                    font-weight: 500;
                    padding: 10px 14px;
                    border: 1px solid #d0d7de;
                    border-radius: 10px;
                }}
                QComboBox:hover {{
                    border: 1px solid {CLR_PRIMARY};
                }}
                QComboBox::drop-down {{
                    border: none;
                    width: 30px;
                }}
                QComboBox::down-arrow {{
                    image: none;
                    border: none;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {BG_CARD};
                    color: {TEXT_PRIMARY};
                    font-size: 12px;
                    selection-background-color: #e8f8f0;
                    selection-color: {CLR_PRIMARY};
                    border: 1px solid #d0d7de;
                    border-radius: 8px;
                    padding: 4px;
                    outline: none;
                }}
            """)
            self.precision_combo.currentIndexChanged.connect(self._on_precision_changed)
            precision_group.addWidget(self.precision_combo)
            control_layout.addLayout(precision_group)

            control_layout.addWidget(self._make_divider())

        # ─── 4. 状态显示 ───
        status_group = QVBoxLayout()
        status_group.setSpacing(12)

        status_title = QLabel("实时状态")
        status_title.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 2px;
            }}
        """)
        status_group.addWidget(status_title)

        # FPS
        fps_row = QHBoxLayout()
        fps_dot = QLabel("●")
        fps_dot.setStyleSheet(f"color: {STATUS_ON}; font-size: 8px;")
        fps_dot.setFixedWidth(16)
        fps_label = QLabel("帧率")
        fps_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 700;
            }}
        """)
        fps_row.addWidget(fps_dot)
        fps_row.addWidget(fps_label)
        fps_row.addStretch()
        fps_row.addWidget(self.fps_label)
        status_group.addLayout(fps_row)

        # 检测人数
        face_row = QHBoxLayout()
        face_dot = QLabel("●")
        face_dot.setStyleSheet(f"color: {CLR_SECONDARY}; font-size: 8px;")
        face_dot.setFixedWidth(16)
        face_label = QLabel("检测人数")
        face_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 11px;")
        self.face_count_label = QLabel("0")
        self.face_count_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 700;
            }}
        """)
        face_row.addWidget(face_dot)
        face_row.addWidget(face_label)
        face_row.addStretch()
        face_row.addWidget(self.face_count_label)
        status_group.addLayout(face_row)

        # 分辨率
        self.resolution_label = QLabel("分辨率: -- × --")
        self.resolution_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 400;
            }}
        """)
        status_group.addWidget(self.resolution_label)

        control_layout.addLayout(status_group)

        # ─── 弹性空间 ───
        control_layout.addStretch()

        # ─── 5. 按钮区域 ───
        button_group = QVBoxLayout()
        button_group.setSpacing(10)

        # 检测启停按钮（Ins 风格渐变）
        self.toggle_button = QPushButton("■  停止检测")
        self.toggle_button.setMinimumHeight(48)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLR_PRIMARY}, stop:1 {CLR_ACCENT});
                color: white;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 2px;
                border: none;
                border-radius: 12px;
                padding: 12px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #27ae60, stop:1 #16a085);
            }}
            QPushButton:pressed {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #219a52, stop:1 #148f77);
            }}
            QPushButton:checked {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {STATUS_ON}, stop:1 #27ae60);
            }}
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.clicked.connect(self._on_toggle_detection)

        # 退出按钮
        self.exit_button = QPushButton("退出程序")
        self.exit_button.setMinimumHeight(44)
        self.exit_button.setCursor(Qt.PointingHandCursor)
        self.exit_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 2px;
                border: 1px solid #d0d7de;
                border-radius: 12px;
                padding: 12px;
            }}
            QPushButton:hover {{
                border: 1px solid {CLR_PRIMARY};
                color: {CLR_PRIMARY};
                background-color: #e8f8f0;
            }}
        """)
        self.exit_button.clicked.connect(self._on_exit)

        button_group.addWidget(self.toggle_button)
        button_group.addWidget(self.exit_button)
        control_layout.addLayout(button_group)

        # ─── 检测模式提示 ───
        mode_text = "占位模式" if self.detector.is_placeholder else "Viola-Jones 真实模式"
        mode_label = QLabel(f"● {mode_text}")
        mode_label.setAlignment(Qt.AlignCenter)
        mode_label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 9px;
                font-weight: 400;
                letter-spacing: 1px;
                padding: 4px;
            }}
        """)
        control_layout.addWidget(mode_label)

        main_layout.addWidget(control_card, stretch=1)

        # ─── 全局样式 ───
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {BG_MAIN};
            }}
            QWidget {{
                background-color: transparent;
            }}
            QMessageBox {{
                background-color: {BG_CARD};
                color: {TEXT_PRIMARY};
                font-size: 13px;
                border: 1px solid #d0d7de;
                border-radius: 12px;
            }}
            QMessageBox QLabel {{
                color: {TEXT_PRIMARY};
                font-size: 13px;
                padding: 10px;
            }}
            QMessageBox QPushButton {{
                background-color: {BG_INPUT};
                color: {TEXT_PRIMARY};
                font-size: 12px;
                font-weight: 600;
                border: 1px solid #d0d7de;
                border-radius: 8px;
                padding: 8px 24px;
                min-width: 80px;
            }}
            QMessageBox QPushButton:hover {{
                background-color: #e8ecf0;
                border: 1px solid {CLR_PRIMARY};
            }}
            QMessageBox QPushButton:pressed {{
                background-color: #d0d7de;
            }}
        """)

    # ═══════════════════════════════════════════════════════
    # UI 辅助方法
    # ═══════════════════════════════════════════════════════

    def _make_divider(self) -> QFrame:
        """创建清新风格分隔线"""
        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("""
            QFrame {
                color: #dce1e8;
                border: none;
                border-top: 1px solid #dce1e8;
                max-height: 1px;
            }
        """)
        return div

    def _make_value_label(self, text: str) -> QLabel:
        """创建数值显示标签"""
        label = QLabel(text)
        label.setAlignment(Qt.AlignRight)
        label.setStyleSheet(f"""
            QLabel {{
                color: {CLR_PRIMARY};
                font-size: 14px;
                font-weight: 700;
            }}
        """)
        return label

    def _make_slider(self, min_val: int, max_val: int,
                     default: int, callback) -> QSlider:
        """创建清新风格滑动条"""
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setStyleSheet(f"""
            QSlider {{
                height: 24px;
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: #dce1e8;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLR_PRIMARY}, stop:1 {CLR_ACCENT});
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                width: 22px;
                height: 22px;
                margin: -9px 0;
                border-radius: 11px;
            }}
            QSlider::sub-page:horizontal {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLR_PRIMARY}, stop:1 {CLR_ACCENT});
                border-radius: 2px;
            }}
            QSlider::add-page:horizontal {{
                background: #dce1e8;
                border-radius: 2px;
            }}
        """)
        slider.valueChanged.connect(callback)
        return slider

    def _build_slider_group(self, label_text: str,
                            value_label: QLabel,
                            slider: QSlider,
                            suffix: str = "") -> QVBoxLayout:
        """构建一组标签 + 滑动条布局"""
        group = QVBoxLayout()
        group.setSpacing(8)

        # 标题行
        header = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {TEXT_SECONDARY};
                font-size: 10px;
                font-weight: 600;
                letter-spacing: 2px;
            }}
        """)
        header.addWidget(label)
        header.addStretch()
        header.addWidget(value_label)
        if suffix:
            suffix_label = QLabel(suffix)
            suffix_label.setStyleSheet(f"""
                QLabel {{
                    color: {TEXT_SECONDARY};
                    font-size: 11px;
                    font-weight: 400;
                }}
            """)
            header.addWidget(suffix_label)

        group.addLayout(header)
        group.addWidget(slider)
        return group

    # ═══════════════════════════════════════════════════════
    # 信号槽：参数变化
    # ═══════════════════════════════════════════════════════

    def _on_precision_changed(self, index: int):
        """检测精度下拉框变化回调"""
        step_delta = self.precision_combo.currentData()
        if self.video_thread is not None:
            self.video_thread.step_delta = step_delta

    def _on_nms_changed(self, value: int):
        """NMS 阈值滑动条变化回调"""
        threshold = value / 100.0
        # 直接更新保存的数值标签引用
        self._nms_value_label.setText(f"{threshold:.2f}")
        if self.video_thread is not None:
            self.video_thread.nms_threshold = threshold

    def _on_face_size_changed(self, value: int):
        """最小人脸尺寸滑动条变化回调"""
        # 直接更新保存的数值标签引用
        self._face_size_value_label.setText(str(value))
        if self.video_thread is not None:
            self.video_thread.min_face_size = value

    def _on_toggle_detection(self, checked: bool):
        """检测启停按钮回调"""
        if self.video_thread is not None:
            self.video_thread.detect_enabled = not checked
            if checked:
                self.toggle_button.setText("▶  开始检测")
            else:
                self.toggle_button.setText("■  停止检测")

    # ═══════════════════════════════════════════════════════
    # 视频线程管理
    # ═══════════════════════════════════════════════════════

    def start_detection(self):
        """启动视频检测线程"""
        if self.video_thread is not None:
            self.stop_detection()

        self.video_thread = VideoThread(
            detector=self.detector,
            camera_id=self.camera_id,
            parent=self
        )

        self.video_thread.frame_signal.connect(self._update_frame)
        self.video_thread.stats_signal.connect(self._update_stats)

        # 同步当前参数
        self.video_thread.nms_threshold = 0.5
        self.video_thread.min_face_size = 24

        self.video_thread.start()

    def stop_detection(self):
        """停止视频检测线程"""
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread.wait(2000)
            self.video_thread = None

    # ═══════════════════════════════════════════════════════
    # 信号槽：更新 UI
    # ═══════════════════════════════════════════════════════

    def _update_frame(self, qimage: QImage):
        """更新视频帧显示"""
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def _update_stats(self, fps: float, face_count: int,
                      frame_w: int, frame_h: int):
        """更新统计信息显示"""
        self.fps_label.setText(f"{fps:.1f}")
        self.face_count_label.setText(str(face_count))
        self.resolution_label.setText(f"分辨率: {frame_w} × {frame_h}")

    # ═══════════════════════════════════════════════════════
    # 窗口事件
    # ═══════════════════════════════════════════════════════

    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_detection()
        event.accept()

    def _on_exit(self):
        """退出程序按钮回调"""
        reply = QMessageBox.question(
            self, "确认退出",
            "确定要退出实时人脸检测系统吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.stop_detection()
            QApplication.quit()
