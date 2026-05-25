# 实时人脸检测系统 —— 详细设计文档

> 作者：成员 C
> 日期：2026-05-25
> 范围：`main.py` / `detect/` / `utils/nms.py` / `ui/`

---

## 一、系统总览

本系统是基于 Viola-Jones 2004 论文的实时人脸检测应用，完整复现了从积分图构建、Haar 特征计算、级联分类器推理到多尺度滑动窗口检测的全链路。我负责的部分涵盖**系统入口**、**多尺度检测推理**、**后处理去重**以及**多线程前端界面**，将核心算法封装为一个可交互、可调参的桌面应用。

```
main.py  (入口，参数解析，组装)
  │
  ├─ Detector (detect/detector.py)        ← 多尺度检测编排
  │    ├─ integral_image.build()          ← O(1) 积分图
  │    ├─ haar_features.iter_scales()     ← 图像金字塔
  │    ├─ CascadeClassifier.predict_window()  ← 级联推理 + 早期拒绝
  │    └─ _post_process()                 ← 后处理分派
  │         ├─ nms()                       ← 现代 IoU NMS
  │         └─ group_rectangles_original() ← 原著均值合并
  │
  └─ MainWindow (ui/main_window.py)       ← PyQt5 主界面
       ├─ VideoThread (ui/video_thread.py) ← 后台采集 + 检测
       └─ 手动触发检测 (图片模式)          ← 主线程同步检测
```

系统有两种运行路径：
1. **实时摄像头**：`QThread` 后台循环采集 → 跳帧检测 → `pyqtSignal` 异步渲染
2. **静态图片**：停止摄像头 → 上传图片 → 点击按钮 → 主线程同步检测 → 结果叠加

---

## 二、双检测模式实现

### 2.1 实时摄像头检测 —— QThread + 跳帧 + 异步信号槽

摄像头检测的核心挑战是：**Viola-Jones 全链路在纯 Python 下的单帧检测耗时远高于 33ms（30fps 的帧间隔），必须在主线程不被阻塞的前提下尽可能提高视觉流畅度**。解决方案是三管齐下：

#### (a) QThread 后台线程

`VideoThread` 继承 `QThread`，在独立的 `run()` 方法中执行摄像头读取和检测的完整循环。主线程只负责接收 `frame_signal` 信号并刷新 QLabel 的 QPixmap。

```python
# ui/video_thread.py —— 信号定义
frame_signal = pyqtSignal(QImage)           # 帧数据 → UI
stats_signal = pyqtSignal(float, int, int, int)  # FPS, 人数, 分辨率
```

这种设计使检测耗时完全不影响 UI 响应——即使单帧检测花费 200ms，窗口仍然可以拖动、按钮仍然可点击。

#### (b) 跳帧检测 + 结果缓存

不是每帧都做检测。`_detect_interval = 2` 意味着每 2 帧检测 1 次，中间帧直接复用上一帧的检测框。这相当于将有效检测帧率翻倍，且人眼几乎感觉不到检测框的"滞后"。

```python
# ui/video_thread.py:356-383
self._frame_counter += 1
if self._frame_counter % self._detect_interval == 0:
    # 仅在检测帧执行检测
    small_boxes = self.detector.detect(gray, ...)
    self._last_face_boxes = [...]  # 坐标还原
face_boxes = self._last_face_boxes  # 中间帧复用缓存
```

#### (c) 检测前降采样

摄像头原始分辨率 640×480 在检测前被缩小到 480×360（面积减少 ~44%）。滑动窗口数量与图像面积成正比，因此这一步直接将候选窗口数削减近一半。检测完成后，框坐标按缩放比例还原回原始分辨率：

```python
# ui/video_thread.py:362-380
DETECT_WIDTH, DETECT_HEIGHT = 480, 360
detect_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
# ... 检测 ...
scale_x = frame.shape[1] / DETECT_WIDTH
scale_y = frame.shape[0] / DETECT_HEIGHT
boxes = [(int(x * scale_x), int(y * scale_y), ...) for ...]
```

#### (d) QImage 悬垂指针修复

一个隐藏较深的 bug：`QImage(rgb_frame.data, ...)` 只是引用了 numpy 数组的内存，并不持有它。当 `rgb_frame` 离开作用域被 Python GC 回收后，QImage 内部指针悬垂，导致随机崩溃。修复方法是显式调用 `.copy()` 强制 QImage 持有独立内存：

```python
# ui/video_thread.py:392 —— 修复前后对比
# 修复前（随机崩溃）:
qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
# 修复后:
qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
```

### 2.2 静态图片检测 —— 同步检测 + 自适应降采样 + 坐标还原

图片模式的设计从最初"上传即检测"的自动触发，经历了三次重构（ImageDetectThread → 独立 Detector → 当前方案）最终收敛为**手动触发 + 主线程同步检测**。

核心流程（`main_window.py:_run_image_detection`）：

```
加载原图 → 长边缩放到 ≤600px → 同步 detect() → 坐标还原 → 叠加框 → 显示
```

降采样策略与摄像头模式不同：图片的长边超过 600px 时才等比缩放，缩放完成后在检测函数内部也设了 `max_image_dim=0`（因为图片层面已处理过缩放），避免双重缩放：

```python
# main_window.py:816-824
MAX_DIM = 600
if max(h_orig, w_orig) > MAX_DIM:
    scale = MAX_DIM / max(h_orig, w_orig)
    detect_img = cv2.resize(self._static_image, (int(w_orig * scale), int(h_orig * scale)))
else:
    scale = 1.0
    detect_img = self._static_image
```

按钮的五态机设计保证了用户操作路径的清晰：

| 状态 | 按钮文字 | 行为 |
|------|---------|------|
| 刚上传，未检测 | `▶ 开始检测` | 执行检测 |
| 检测完成 | `■ 隐藏检测框` | 切换原图/带框图 |
| 检测框已隐藏 | `▶ 显示检测框` | 重新显示检测框 |
| 调参后结果过期 | `▶ 重新检测` | 用新参数检测 |
| 检测框已显示（再次点击） | `▶ 显示检测框` | 隐藏检测框 |

参数变化时通过 `_mark_image_stale()` 自动将按钮回退到「重新检测」状态，不需要用户记住当前结果是否有效。

---

## 三、双后处理（去重）机制

多尺度滑动窗口检测必然产生大量重叠候选框——同一个人脸在相邻尺度和相邻位移下会被反复命中。后处理的目标是：将这些碎片化的候选框合并为每张人脸一个精确框。

我实现了两种独立的去重算法，通过 `Detector._nms_mode` 切换。

### 3.1 现代 IoU NMS（`nms()`）

这是对标准 NMS 的改进版本，核心创新是用 **IoM（Intersection over Minimum）** 替代传统 IoU 来衡量重叠程度。

**为什么需要 IoM？** 考虑一个典型场景：100×100 的大框完全包含一个 20×20 的小框。传统 IoU = 400 / 10000 = 0.04，远低于任何合理阈值，两个框会被视为"不重叠"。但实际上小框是大框在更精细尺度下的冗余检测，应该被合并。IoM = 400 / min(10000, 400) = 1.0，正确捕捉了这种包含关系。

```python
# utils/nms.py:144-146 —— IoM 计算
overlap = inter_area / np.minimum(areas[i], areas)  # IoM, 不是 IoU
votes[i] = np.sum(overlap > iou_threshold)
```

算法三步：
1. **投票**：统计每个框周围有多少个重叠框（IoM > 阈值）。这是对"候选框置信度"的代理指标——票数越高，说明越多尺度/位移的窗口同时认为这里有人脸。
2. **过滤**：删除票数 < `min_votes` 的孤立框（通常是背景纹理误触发）。
3. **逐组合并**：按票数降序，每次取票王，将其与所有重叠框取坐标均值，然后从候选集中移除该组，继续下一组。

该算法的优点是无需依赖检测器提供置信度分数（Viola-Jones 级联本身不输出概率），仅利用几何一致性即可完成去重。

### 3.2 原著均值合并（`group_rectangles_original()`）

这是基于 Viola-Jones 2004 论文 Section 5 数学思想重构的算法，经过一次重大 Bug 修复。

**数据结构**：内部使用 `_UnionFind` 并查集，支持路径压缩和按秩合并，将 O(N²) 的连通分量查找优化到接近 O(N α(N))。

**判定逻辑**（修复后的核心）：

```python
# utils/nms.py:341-355 —— ε-邻域坐标差值判定
dx = abs(x1s[i] - x1s[j])          # 左上角 x 差值
dy = abs(y1s[i] - y1s[j])          # 左上角 y 差值
dw = abs(x2s[i] - x2s[j])          # 右下角 x 差值
dh = abs(y2s[i] - y2s[j])          # 右下角 y 差值

limit_w = eps * min(ws[i], ws[j])  # 容差 = ε × 较小框宽度
limit_h = eps * min(hs[i], hs[j])  # 容差 = ε × 较小框高度

if dx <= limit_w and dy <= limit_h and dw <= limit_w and dh <= limit_h:
    uf.union(i, j)
```

关键设计决策：**四个坐标差值必须全部在容差范围内才归组**。这产生的是非传递性判定——框 A 和框 B 在 ε-邻域内，框 B 和框 C 也在 ε-邻域内，但框 A 和框 C 可能不在同一邻域。并查集会通过 B 将三者连通，但由于容差基于较小框的尺寸，且要求四个角点全部对齐，实际中链式传播被严格限制。

**容差基准**：`eps * min(w_i, w_j)` —— 使用较小框的尺寸作为基准，意味着两个框尺寸差异越大，判定越严格。这符合直觉：100×100 和 20×20 的两个框不太可能属于同一个人脸。

三步流程：
1. **归组**：上述 ε-邻域判定 + 并查集
2. **投票过滤**：组内框数 < `group_threshold` 的整组丢弃（孤立噪声）
3. **均值合并**：组内所有框的坐标和尺寸取算术平均，得到平滑的最终框

### 3.3 两种算法的对比

| 维度 | 现代 IoU NMS | 原著均值合并 |
|------|-------------|-------------|
| 重叠度量 | IoM（交集/最小面积） | ε-邻域坐标差值 |
| 分组方式 | 贪心逐组（按票数降序） | 并查集连通分量 |
| 合并方式 | 组内坐标均值 | 组内坐标算术平均 |
| 过滤阈值 | `iou_threshold` + `min_votes` | `eps` + `group_threshold` |
| 传递性 | 贪心（组间无传递） | 并查集（有传递但受 ε 约束） |
| 适用场景 | 通用目标检测 | 人脸检测（框尺寸相对均匀） |

两种算法的输入输出接口完全一致，通过 `Detector._post_process()` 统一分派：

```python
# detect/detector.py:106-132
def _post_process(self, candidates, iou_threshold, min_votes):
    if self._nms_mode == 1:
        return group_rectangles_original(candidates, group_threshold=min_votes)
    else:
        return nms(candidates, iou_threshold=iou_threshold, min_votes=min_votes)
```

---

## 四、参数动态调节面板

右侧控制面板提供 7 个可调参数，滑动条变化立即通过信号槽同步到检测流。

### 4.1 参数详解

#### NMS 阈值（IoU）/ IOU Threshold
- **范围**：0.00 ~ 1.00，默认 0.15
- **意义**：判断两个候选框是否"重叠"的门槛。使用 IoM（而非传统 IoU）计算重叠度
- **调大**：归组更宽松 → 更多框被合并 → 漏检率降低但同一人脸可能出现多个框
- **调小**：归组更严格 → 只有高度重叠的框才合并 → 框更精准但可能产生碎片化检测
- **实时性**：影响 NMS 的 `iou_threshold` 参数，摄像头模式下直接传给 `VideoThread`，图片模式下缓存为 `_current_nms_threshold`

#### 最小人脸尺寸 / Min Face Size
- **范围**：20 ~ 300 px，默认 60 px
- **意义**：图像金字塔的最小检测窗口尺寸。检测器不会尝试检测比这更小的人脸
- **调大**：跳过小尺度 → 金字塔层数减少 → **FPS 大幅提升**，但远处小人脸漏检
- **调小**：覆盖小尺度 → 金字塔层数增加 → 检测更全面但耗时显著增加
- **数学关系**：金字塔层数 ≈ log(W / min_face) / log(scale_factor)，min_face 翻倍约减少 2~3 层

#### 最大人脸尺寸 / Max Face Size
- **范围**：0 ~ 1000 px，默认 140 px
- **意义**：金字塔的最大检测窗口尺寸。值为 0 时无上限
- **调大**：覆盖近处大人脸 → 金字塔层数增加
- **调小**：限制最大尺度 → 加速检测，适合自拍/半身照场景
- **注意**：如果图像中的人脸尺寸超过此值，将完全不会被检测到

#### 最少重叠票数 / Min Votes
- **范围**：2 ~ 3000，默认 45
- **意义**：
  - NMS 模式：候选框周围至少有多少个重叠框才被认为有效（低于此值为噪声）
  - 均值合并模式：连通分支至少包含多少个框才保留（低于此值整组丢弃）
- **调大**：过滤更激进 → 误检率大幅下降，但真阳性也可能被过滤
- **调小**：过滤更宽松 → 召回率提升但可能引入背景噪声
- **工程洞察**：这个参数是抵御 Viola-Jones 高误检率的第一道防线。在模型本身判别力有限的情况下，用几何一致性（多个尺度/位移的窗口同时命中）作为"置信度"的替代指标

#### 滑动步长比例 / Step Factor (Δ)
- **范围**：0.5 ~ 5.0，默认 3.0
- **意义**：滑动窗口的位移步长，公式为 `step = max(1, round(scale × Δ))`，遵循 VJ2004 论文 Section 4.1 的原文公式
- **Δ = 1.0**：窗口几乎逐像素滑动，精度最高但极慢
- **Δ = 3.0**：平衡模式（默认），每 3 个像素滑动一次
- **Δ = 5.0**：大步长快速扫描，可能跳过小尺寸人脸
- **数学影响**：检测耗时 ∝ 1/Δ²（步长加倍，窗口数约为 1/4）

#### 金字塔缩放因子 / Scale Factor
- **范围**：1.05 ~ 2.00，默认 1.25
- **意义**：图像金字塔的层间缩放比例。每一层将图像缩小为上一层的 `1/scale_factor`
- **1.05**：精细金字塔（~60 层），几乎不漏检但极慢
- **1.25**：标准金字塔（~15 层），论文推荐值
- **2.00**：粗糙金字塔（~7 层），速度快但尺度跳跃大，易漏检中间尺寸的人脸
- **FPS 影响**：层数与 `log(1/scale_factor)` 成反比，1.05 → 1.25 可减少约 70% 的层数

#### 后处理算法 / NMS Mode
- **选项**：现代 IoU NMS（默认）/ 原著均值合并
- **意义**：选择去重算法的实现。详见第三章对比

### 4.2 参数同步路径

```
用户拖动滑动条
  → valueChanged 信号
    → _on_xxx_changed() 回调
      → 更新 self._current_xxx（缓存）
      → 如果摄像头在运行：video_thread.xxx = value（实时生效）
      → 如果是图片模式 + 已检测：_mark_image_stale()（标记过期）

摄像头 run() 循环中每帧：
  → self.detector.min_face_size = self._min_face_size（参数注入）
  → self.detector.detect(gray, iou_threshold=..., min_votes=...)

图片检测触发时：
  → self.detector.min_face_size = self._current_min_face_size（参数注入）
  → self.detector.detect(detect_img, iou_threshold=..., min_votes=...)
```

参数注入到 Detector 后，影响 `_detect_real()` 中的 `iter_scales()`（控制金字塔层数）和滑动窗口步长计算，最终决定候选框的数量和质量。

---

## 五、级联分类器推理

`CascadeClassifier` 负责加载训练好的模型并对单个滑动窗口做出"人脸/非人脸"的二分类判决。

### 5.1 模型加载与兼容性

训练代码可能在不同 numpy 版本下运行，产生了两个兼容性问题：

1. **numpy 2.x → 1.x 路径映射**：numpy 2.x 将 C 扩展从 `numpy.core` 迁移到 `numpy._core`，pickle 序列化时记录了新路径。在 numpy 1.x 环境下加载会报 `ModuleNotFoundError`。通过自定义 `_CompatUnpickler` 在 `find_class()` 中实时重映射解决。

2. **模块路径别名**：旧版训练代码以 `adaboost.StrongClassifier` 路径序列化类，当前项目结构为 `train.adaboost.StrongClassifier`。通过在 `sys.modules` 中注册 `adaboost` 别名模块解决。

```python
# detect/cascade_classifier.py:61-64
_adaboost_holder = type(_sys)('adaboost')
_adaboost_holder.StrongClassifier = StrongClassifier
_adaboost_holder.WeakClassifier = WeakClassifier
_sys.modules['adaboost'] = _adaboost_holder
```

### 5.2 早期拒绝机制

级联分类器的核心思想是：绝大多数窗口是非人脸，应该在级联的早期阶段就被快速拒绝，避免计算后续层的所有特征。每一层（Stage）是一个强分类器，包含若干弱分类器：

```python
# detect/cascade_classifier.py:147-183
for stage_idx, stage in enumerate(stages):
    stage_score = 0.0
    for wc in stage.weak_classifiers:
        raw_feat_val = compute_feature_at_scale(...)
        normalized_feat = raw_feat_val * inv_norm  # 尺度+方差归一化
        if wc.polarity * normalized_feat < wc.polarity * wc.threshold:
            stage_score += wc.alpha
    if stage_score < stage.threshold:
        return 0  # 早期拒绝：不计算后续层的特征
return 1  # 通过所有层
```

归一化策略：特征值除以 `(scale² × σ)`，其中 σ 是窗口像素标准差。这与训练时对图像整体做方差归一化的逻辑等效，确保了训练-推理的统计一致性。

---

## 六、系统入口 (`main.py`)

`main.py` 负责参数解析和模块组装，核心逻辑简洁：

```python
detector = Detector(
    model_path=model_path,
    scale_factor=args.scale_factor,
    step_factor=args.step_factor,
    min_face_size=args.min_face,
    max_face_size=args.max_face,
    max_image_dim=0,  # VideoThread 已做缩放，不再重复
)
app = QApplication(sys.argv)
window = MainWindow(detector=detector, camera_id=args.camera)
window.show()
sys.exit(app.exec_())
```

`max_image_dim=0` 是一个关键设计决策：摄像头模式下 VideoThread 已经在帧级别做了 640→480 的降采样，如果 Detector 内部再做一次降采样会导致双重缩放和坐标映射混乱。图片模式下的降采样在 `MainWindow._run_image_detection()` 中单独处理。

---

## 七、线程安全设计

系统的线程安全模型遵循一个简单原则：**摄像头运行时不切换模式，切换模式时先彻底停止摄像头**。

- `stop_detection()`：`_running = False` → `wait(3000)` → 超时则 `terminate()` → `deleteLater()`
- 线程停止后，`self.video_thread = None`，此后的任何操作都不会意外访问已销毁的线程
- 图片检测在主线程同步执行，不存在任何竞态条件
- Detector 实例在摄像头线程和主线程之间共享，但只在摄像头**已停止**后才被主线程的图片检测使用
