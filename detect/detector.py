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
    5. 返回候选的 bounding boxes，供 NMS/均值合并过滤。

对外接口统一为 detect(gray_frame) -> List[(x, y, w, h), ...]
"""

import cv2
import numpy as np
from typing import List, Tuple

# ─── 强行导入 Viola-Jones 真实模块 ───
from train.integral_image import build, window_variance
from train.haar_features import iter_scales, BASE_WIN_SIZE
from detect.cascade_classifier import CascadeClassifier as RealCascade

# ─── 导入后处理模块 ───
from utils.nms import nms, group_rectangles_original


class Detector:
    """
    多尺度人脸检测器 —— 纯正的 Viola-Jones 实现。

    架构：
        积分图 + Haar-like 特征 + 级联分类器

    对外接口统一为 detect(gray_frame) -> List[(x, y, w, h), ...]
    """

    def __init__(self,
                 model_path: str = "models/cascade_model.pkl",
                 scale_factor: float = 1.25,
                 step_delta: float = 1.5,
                 step_factor: float = None,
                 min_face_size: int = 40,
                 max_face_size: int = 500,
                 max_image_dim: int = 640,
                 verbose: bool = False):
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
            max_image_dim  : 检测前将图像长边缩放到此尺寸以内以加速检测（0=不缩放）
            verbose        : 是否打印逐窗口调试日志
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
        self.max_image_dim = max_image_dim
        self._verbose = verbose

        # ═══ 后处理算法模式 ═══
        # 0 = 现代 IoU NMS（nms 函数）
        # 1 = 原著均值合并（group_rectangles_original 函数）
        self._nms_mode = 0

        # ═══ 加载真实 Viola-Jones 级联模型 ═══
        # 兼容性：支持 str 路径 或 已实例化的 CascadeClassifier 对象
        if isinstance(model_path, str):
            print(f"[Detector] 加载 Viola-Jones 模型: {model_path}")
            self._cascade = RealCascade(model_path, verbose=verbose)
        else:
            self._cascade = model_path
            print(f"[Detector] 使用已加载的 CascadeClassifier（共 {len(self._cascade.stages)} 层级联）")
        print(f"[Detector] 模型加载成功！共 {len(self._cascade.stages)} 层级联")

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
            min_votes    : 最小票数 / 连通分支过滤阈值

        返回：
            过滤合并后的框列表
        """
        if not candidates:
            return []

        if self._nms_mode == 1:
            # ═══ 原著均值合并算法 ═══
            # group_threshold=min_votes 作为连通分支过滤阈值
            return group_rectangles_original(
                candidates,
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
            List[(x, y, w, h), ...]
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

        # ─── 大图自动降采样 ───
        # 长边超过 max_image_dim 时等比缩放，大幅减少滑动窗口数量
        # 检测后将框坐标还原到原始尺寸
        H_orig, W_orig = gray.shape
        scale_img = 1.0
        if self.max_image_dim > 0 and max(H_orig, W_orig) > self.max_image_dim:
            scale_img = self.max_image_dim / max(H_orig, W_orig)
            new_w = int(W_orig * scale_img)
            new_h = int(H_orig * scale_img)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        faces = self._detect_real(gray, iou_threshold, min_votes)

        # 坐标还原到原始图像尺寸
        if scale_img != 1.0:
            faces = [(int(x / scale_img), int(y / scale_img),
                      int(w / scale_img), int(h / scale_img)) for (x, y, w, h) in faces]

        return faces

    def _detect_real(self, gray: np.ndarray,
                     iou_threshold: float = 0.3,
                     min_votes: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        完整 Viola-Jones 检测流程（对应论文 Section 4）。

        流程：
            1. 构建积分图（Integral Image）
            2. 按 scale_factor 生成多尺度图像金字塔
            3. 在每个尺度下按步长滑动窗口
            4. 对每个窗口计算方差，快速过滤平滑区域
            5. 调用级联分类器 CascadeClassifier.predict_window()
            6. 收集所有通过级联的候选框
            7. 调用后处理（NMS/均值合并）合并重复框

        返回：
            List[(x, y, w, h), ...]  坐标相对于原始图像尺寸
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
            # 其中 Δ = step_factor
            step = max(1, int(round(scale * self.step_factor)))

            # 步骤 4：滑动窗口遍历
            for r in range(0, H - win_size_px + 1, step):
                for c in range(0, W - win_size_px + 1, step):

                    # 边界安全检查：特征坐标各自 round 后再相加，
                    # 可能比 win_size_px 多出最多 3 px 的舍入误差，
                    # 若窗口紧贴图像右/下边缘则会导致积分图越界。
                    max_feat_r = r + win_size_px + 3
                    max_feat_c = c + win_size_px + 3
                    if max_feat_r >= H or max_feat_c >= W:
                        continue

                    # 4.1 O(1) 计算子窗口方差（用于光照归一化）
                    var = window_variance(iimg, r, c, win_size_px)

                    # 4.2 低方差快速拒识（优化技巧）
                    if var < 10.0:
                        continue

                    # 4.3 调用级联分类器判决
                    is_face = self._cascade.predict_window(
                        ii=iimg.ii,
                        scale=scale,
                        win_r=r,
                        win_c=c,
                        variance=var
                    )

                    # 4.4 记录候选框
                    if is_face:
                        candidates.append((c, r, win_size_px, win_size_px))

        # ─── 统一后处理（支持双算法切换） ───
        # min_votes 作为后处理的过滤阈值：
        #   NMS 模式：同一位置至少被 min_votes 个窗口命中
        #   均值合并模式：连通分支至少包含 min_votes 个框
        final_faces = self._post_process(
            candidates,
            iou_threshold=iou_threshold,
            min_votes=min_votes
        )

        return final_faces
