"""
detector.py
===========
多尺度检测器模块

位置：detect/detector.py
职责：
    1. 接收彩色/灰度图像。
    2. 计算整张图的积分图。
    3. 按照 scale_factor 遍历多个尺度。
    4. 在每个尺度下按步长滑动窗口，计算方差并调用 CascadeClassifier 判决。
    5. 返回候选的 bounding boxes，供 NMS 过滤。

===== 占位模式说明（成员 C 添加）=====
由于成员 B 的 cascade_model.pkl 尚未生成/上传，
本模块提供两种运行模式：
  1. 真实模式：使用成员 B 的 CascadeClassifier + 积分图 + Haar 特征（完整 Viola-Jones）
  2. 占位模式：使用 OpenCV 内置的 Haar Cascade（haarcascade_frontalface_default.xml）
              作为临时替代，保证 UI、NMS、视频流等模块可以独立测试。

切换方式：
  - 当 models/cascade_model.pkl 存在时 → 自动使用真实模式
  - 否则 → 自动使用占位模式（OpenCV 内置检测器）
  - 也可通过 Detector(use_placeholder=True/False) 强制指定

将来成员 B 上传 .pkl 后，只需将文件放到 models/ 目录下，
系统会自动切换到真实模式，无需修改任何 UI 代码。
=====================================
"""

import cv2
import numpy as np
import os
from typing import List, Tuple

# ─── 尝试导入成员 A/B 的模块（可能因缺少 .pkl 而失败） ───
try:
    from train.integral_image import build, window_variance
    from train.haar_features import iter_scales, BASE_WIN_SIZE
    from detect.cascade_classifier import CascadeClassifier as RealCascade
    _HAS_REAL_MODULES = True
except (ImportError, ModuleNotFoundError):
    _HAS_REAL_MODULES = False

# ─── 导入 NMS 模块 ───
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nms import nms, group_rectangles_original


class Detector:
    """
    多尺度人脸检测器。

    支持两种模式：
      - 真实模式：使用 Viola-Jones 完整算法（积分图 + Haar + 级联）
      - 占位模式：使用 OpenCV 内置 Haar Cascade 作为临时替代

    对外接口统一为 detect(gray_frame) -> List[[x, y, w, h], ...]
    """

    def __init__(self,
                 model_path: str = "models/cascade_model.pkl",
                 scale_factor: float = 1.25,
                 step_delta: float = 1.5,
                 step_factor: float = None,
                 min_face_size: int = 40,
                 max_face_size: int = 500,
                 use_placeholder: bool = None):
        """
        初始化检测器。

        参数：
            model_path     : 成员 B 训练好的级联模型路径（.pkl 文件）
            scale_factor   : 图像金字塔缩放因子（越大越快，但可能漏检）
            step_delta     : [已弃用] 旧版步长系数，保留仅用于兼容性
            step_factor    : 原著 VJ2004 步长因子 Δ
                             步长 = max(1, round(scale * step_factor))
                             默认 3.0（与旧版 step_delta=1.5 性能完全一致）
                             1.5 = 高精度（偏慢），3.0 = 平衡（默认），4.5 = 高速度（偏快）
            min_face_size  : 最小人脸尺寸（像素）
            max_face_size  : 最大人脸尺寸（像素）
            use_placeholder: 是否强制使用占位模式
                             - True  → 强制使用 OpenCV 内置检测器
                             - False → 强制使用真实 Viola-Jones 模型
                             - None  → 自动判断（优先尝试加载 .pkl）
        """
        self.scale_factor = scale_factor
        self.step_delta = step_delta
        # ─── 兼容性说明 ───
        # 旧版公式: step = max(1, round(scale * step_delta * 2))
        # 新版公式: step = max(1, round(scale * step_factor))
        # 旧版默认 step_delta=1.5 → 步长 = round(scale * 3.0)
        # 因此 step_factor=3.0 与旧版默认行为完全一致，保证性能不退化
        self.step_factor = 3.0 if step_factor is None else step_factor
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size

        # ═══ 后处理算法模式 ═══
        # 0 = 现代 IoU NMS（nms 函数）
        # 1 = 原著均值合并（group_rectangles_original 函数）
        self._nms_mode = 0

        # ─── 判断使用哪种模式 ───
        if use_placeholder is None:
            # 自动模式：检查 .pkl 文件是否存在
            pkl_exists = os.path.exists(model_path)
            use_placeholder = not pkl_exists

        self._is_placeholder = use_placeholder

        if use_placeholder:
            # ═══ 占位模式：使用 OpenCV 内置 Haar Cascade ═══
            print("[Detector] ⚠ 使用占位模式（OpenCV 内置 Haar Cascade）")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(
                    f"OpenCV 内置级联文件不存在: {cascade_path}\n"
                    f"请检查 opencv-python 安装是否完整。"
                )
            self._placeholder_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"[Detector] 占位检测器加载成功: {cascade_path}")
        else:
            # ═══ 真实模式：使用成员 B 的 Viola-Jones 模型 ═══
            if not _HAS_REAL_MODULES:
                raise ImportError(
                    "无法导入成员 A/B 的模块（train.integral_image, train.haar_features, "
                    "detect.cascade_classifier）。\n"
                    "请确保这些模块存在且可导入。"
                )
            print(f"[Detector] 使用真实 Viola-Jones 模型: {model_path}")
            self._real_cascade = RealCascade(model_path)
            print(f"[Detector] 真实模型加载成功！共 {len(self._real_cascade.stages)} 层级联")

    @property
    def is_placeholder(self) -> bool:
        """当前是否处于占位模式"""
        return self._is_placeholder

    @property
    def nms_mode(self) -> int:
        """获取后处理算法模式：0=现代 IoU NMS, 1=原著均值合并"""
        return self._nms_mode

    @nms_mode.setter
    def nms_mode(self, value: int):
        """
        设置后处理算法模式。
        参数 value: 0 = 现代 IoU NMS, 1 = 原著均值合并
        """
        self._nms_mode = 1 if value == 1 else 0

    def _post_process(self, candidates: List[Tuple[int, int, int, int]],
                      iou_threshold: float = 0.3,
                      min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        统一后处理入口：根据 nms_mode 选择算法。

        参数：
            candidates   : 候选框列表 [(x, y, w, h), ...]
            iou_threshold: NMS 的 IoU 阈值（仅现代 NMS 使用）
            min_votes    : 最小票数（仅现代 NMS 使用）

        返回：
            过滤合并后的框列表
        """
        if not candidates:
            return []

        if self._nms_mode == 1:
            # ═══ 原著均值合并算法 ═══
            # 使用 group_threshold=min_votes 作为连通分支过滤阈值
            return group_rectangles_original(
                candidates,
                eps=0.2,
                group_threshold=min_votes
            )
        else:
            # ═══ 现代 IoU NMS ═══
            return nms(candidates, iou_threshold=iou_threshold, min_votes=min_votes)

    def detect(self, img_bgr: np.ndarray,
               iou_threshold: float = 0.3,
               min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        对输入图像执行人脸检测。

        参数：
            img_bgr      : 输入图像（BGR 格式，uint8）
                           可以是彩色图（3通道）或灰度图（1通道）
            iou_threshold: NMS 的 IoU 阈值（默认 0.3）
                           两个框的 IoU 超过此阈值 → 视为重复检测
            min_votes    : 最小票数（默认 3）
                           同一位置至少被 N 个不同尺度/位移的窗口命中才输出

        返回：
            List[[x, y, w, h], ...]
            每个元素是一个检测到的人脸框：
              x, y : 框左上角坐标
              w, h : 框的宽度和高度

        注意：
            - 返回的坐标是相对于原始输入图像的
            - 返回空列表表示未检测到人脸
        """
        # ─── 统一转为灰度图 ───
        if len(img_bgr.shape) == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        if self._is_placeholder:
            return self._detect_placeholder(gray, iou_threshold, min_votes)
        else:
            return self._detect_real(gray, iou_threshold, min_votes)

    def _detect_placeholder(self, gray: np.ndarray,
                            iou_threshold: float = 0.3,
                            min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        占位模式检测：使用 OpenCV 内置 Haar Cascade。

        步骤：
            1. 调用 OpenCV 的 detectMultiScale 进行多尺度检测
            2. 将结果转换为统一的 [x, y, w, h] 格式
            3. 按最小人脸尺寸过滤过小的框
            4. 调用 NMS 合并重复框
        """
        # OpenCV 的 detectMultiScale 参数说明：
        #   scaleFactor : 图像金字塔缩放因子（与真实模式保持一致）
        #   minNeighbors: 每个候选框需要保留的最小邻近框数量（值越大，误检越少）
        #   minSize     : 最小检测窗口大小
        #   maxSize     : 最大检测窗口大小
        # 占位模式使用更宽松的参数，确保能检测到人脸
        # minNeighbors=1 表示每个候选框只需 1 个邻近框即可保留，
        # 这样检测率更高（虽然可能多一些误检，但 NMS 会过滤）
        boxes = self._placeholder_cascade.detectMultiScale(
            image=gray,
            scaleFactor=self.scale_factor,
            minNeighbors=1,          # 宽松值，提高检测率（NMS 会过滤重复框）
            minSize=(self.min_face_size, self.min_face_size),
            maxSize=(self.max_face_size, self.max_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # ─── 转换为统一的 List[[x, y, w, h]] 格式 ───
        candidates = []
        if boxes is not None and len(boxes) > 0:
            candidates = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]

        # ─── 统一后处理（支持双算法切换） ───
        # 占位模式：OpenCV 的 detectMultiScale 已经做了内部 NMS，
        # 返回的框数量通常很少（1~3个），不需要投票过滤。
        # 使用 min_votes=1 只做合并（取重叠框的平均值），不做投票过滤。
        return self._post_process(candidates, iou_threshold=iou_threshold, min_votes=1)

    def _detect_real(self, gray: np.ndarray,
                     iou_threshold: float = 0.3,
                     min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        真实模式检测：使用完整的 Viola-Jones 算法。

        流程（对应论文 Section 4）：
            1. 构建积分图（Integral Image）
            2. 按 scale_factor 生成多尺度图像金字塔
            3. 在每个尺度下按步长滑动窗口
            4. 对每个窗口计算方差，快速过滤平滑区域
            5. 调用级联分类器 CascadeClassifier.predict_window()
            6. 收集所有通过级联的候选框
            7. 调用 NMS 合并重复框

        返回：
            List[[x, y, w, h], ...]  坐标相对于原始图像尺寸
        """
        H, W = gray.shape

        # 步骤 1：构建积分图（O(HW) 时间）
        iimg = build(gray)

        # 步骤 2：生成所有检测尺度
        scales = iter_scales(
            H, W,
            scale_factor=self.scale_factor,
            min_face_px=self.min_face_size,
            max_face_px=self.max_face_size
        )

        candidates = []

        # 步骤 3：遍历每个尺度
        for scale in scales:
            win_size_px = int(round(BASE_WIN_SIZE * scale))

            # ═══ 原著 VJ2004 步长公式（Section 4.1）═══
            # 论文原文： "If the current scale is s, the window is shifted by [sΔ]"
            # 其中 Δ = step_factor（默认 1.0）
            # 步长随着检测框变大而等比例变大，避免大尺度下的冗余扫描
            step = max(1, int(round(scale * self.step_factor)))

            # 步骤 4：滑动窗口遍历
            for r in range(0, H - win_size_px + 1, step):
                for c in range(0, W - win_size_px + 1, step):

                    # ─── 边界安全检查（成员 B 添加） ───
                    # 基础窗口是 24x24，特征最大坐标不会超过 24
                    max_feat_offset = 24
                    if (r + int(round(max_feat_offset * scale)) >= H or
                        c + int(round(max_feat_offset * scale)) >= W):
                        continue

                    # 4.1 O(1) 计算子窗口方差（用于光照归一化）
                    var = window_variance(iimg, r, c, win_size_px)

                    # 4.2 低方差快速拒识（优化技巧）
                    # 纯色区域（如墙壁、天空）方差极小，绝不可能是人脸
                    if var < 10.0:
                        continue

                    # 4.3 调用级联分类器判决
                    is_face = self._real_cascade.predict_window(
                        ii=iimg.ii,
                        scale=scale,
                        win_r=r,
                        win_c=c,
                        variance=var
                    )

                    # 4.4 记录候选框 (x, y, w, h) = (c, r, win_size, win_size)
                    if is_face:
                        candidates.append((c, r, win_size_px, win_size_px))

        # ─── 统一后处理（支持双算法切换） ───
        # min_votes=3 表示同一位置至少被3个不同尺度/位移的窗口命中，才输出最终结果
        final_faces = self._post_process(candidates, iou_threshold=iou_threshold, min_votes=min_votes)

        return final_faces


# ─── 快捷函数：直接创建默认检测器 ───
def create_detector(model_path: str = "models/cascade_model.pkl",
                    use_placeholder: bool = None) -> Detector:
    """
    创建并返回一个配置好的 Detector 实例。

    这是给 UI 模块使用的快捷工厂函数。

    参数：
        model_path     : 模型文件路径
        use_placeholder: 是否强制使用占位模式（None=自动判断）

    用法：
        detector = create_detector()  # 自动选择模式
        detector = create_detector(use_placeholder=True)   # 强制占位
        detector = create_detector(use_placeholder=False)  # 强制真实
    """
    return Detector(
        model_path=model_path,
        use_placeholder=use_placeholder
    )
