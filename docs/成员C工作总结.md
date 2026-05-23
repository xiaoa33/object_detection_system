# 成员 C 工作总结

## 一、我的职责

根据项目分工，我（成员 C）负责：
- **系统集成**：NMS 工具（`utils/nms.py`）
- **实时视频界面**：`ui/video_thread.py`、`ui/main_window.py`
- **系统入口**：`main.py`

---

## 附：真实模式完整调用链

当前 `cascade_model.pkl` 是 B 训练好的模型，包含 **3 层级联**（共 14 个弱分类器）：

| 级联层 | 弱分类器数 | 阈值 |
|--------|-----------|------|
| 第 1 层 | 4 个 | 1.98 |
| 第 2 层 | 4 个 | 1.66 |
| 第 3 层 | 6 个 | 3.18 |

每个弱分类器包含一个 `FeatureDesc`（描述 Haar 特征的位置/类型/尺寸）和对应的 `alpha`/`polarity`/`threshold`。

### 实时检测调用链（真实模式）

```
main.py
  └─ MainWindow.__init__(detector)
       └─ MainWindow.start_detection()
            └─ VideoThread.start()
                 └─ VideoThread.run()  ← 后台线程循环
                      │
                      │  每 2 帧执行一次检测（跳帧优化）：
                      │
                      ├─ 1. cv2.resize(frame, (320, 240))  ← 缩小图像加速
                      │
                      ├─ 2. Detector.detect(gray)           ← detect/detector.py
                      │      │
                      │      ├─ 2a. build(gray)             ← train/integral_image.py（成员 A）
                      │      │     构建积分图（O(HW) 时间）
                      │      │
                      │      ├─ 2b. iter_scales(H, W, ...)  ← train/haar_features.py（成员 A）
                      │      │     生成多尺度图像金字塔
                      │      │
                      │      ├─ 2c. 遍历每个尺度，滑动窗口
                      │      │      │
                      │      │      ├─ window_variance(iimg, r, c, size)  ← integral_image.py
                      │      │      │   O(1) 计算子窗口方差，快速过滤纯色区域
                      │      │      │
                      │      │      └─ CascadeClassifier.predict_window(  ← detect/cascade_classifier.py
                      │      │             ii, scale, win_r, win_c, variance)
                      │      │             │
                      │      │             ├─ 遍历 3 层级联
                      │      │             │   ├─ 遍历每层所有弱分类器
                      │      │             │   │   ├─ compute_feature_at_scale(  ← train/haar_features.py
                      │      │             │   │   │     desc, ii, scale, win_r, win_c)
                      │      │             │   │   │   在积分图上 O(1) 计算 Haar 特征值
                      │      │             │   │   │
                      │      │             │   │   └─ 弱分类器判决: p*f(x) < p*θ ?
                      │      │             │   │      是 → 累加 alpha 得分
                      │      │             │   │
                      │      │             │   └─ 强分类器判决: score < threshold ?
                      │      │             │      是 → 早期拒绝（返回 0）
                      │      │             │
                      │      │             └─ 通过所有层 → 返回 1（候选人脸）
                      │      │
                      │      └─ 2d. nms(candidates, iou_threshold, min_votes=2)
                      │            ← utils/nms.py（成员 C）
                      │            基于 IoU 合并重复框
                      │
                      ├─ 3. 框坐标还原到原始尺寸（320→640, 240→480）
                      │
                      ├─ 4. cv2.rectangle 画框 + putText 显示 FPS
                      │
                      └─ 5. pyqtSignal → MainWindow._update_frame() 更新 UI
```

### 涉及的文件（按调用顺序）

| 文件 | 作者 | 函数/类 | 作用 |
|------|------|---------|------|
| `main.py` | **C** | `main()` | 系统入口 |
| `ui/main_window.py` | **C** | `MainWindow` | UI 主窗口 |
| `ui/video_thread.py` | **C** | `VideoThread.run()` | 视频采集循环 |
| `detect/detector.py` | **B** + **C** | `Detector._detect_real()` | 多尺度检测调度 |
| `train/integral_image.py` | **A** | `build()`, `window_variance()` | 积分图构建与方差计算 |
| `train/haar_features.py` | **A** | `iter_scales()`, `compute_feature_at_scale()` | 尺度生成与特征计算 |
| `detect/cascade_classifier.py` | **B** + **C** | `CascadeClassifier.predict_window()` | 级联推理（早期拒绝） |
| `train/adaboost.py` | **A** | `StrongClassifier`, `WeakClassifier` | 强/弱分类器数据结构 |
| `utils/nms.py` | **C** | `nms()` | 非极大值抑制 |

### 模型文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `models/cascade_model.pkl` | 1.5 KB | B 训练的 3 层级联模型（14 个弱分类器） |

---

## 二、我完成的工作

### 2.1 从零编写的模块（3 个）

| 文件 | 说明 |
|------|------|
| `utils/nms.py` | 基于 IoU 的非极大值抑制工具，支持 `boxes=[[x,y,w,h],...]` 输入，去重合并重复检测框 |
| `ui/video_thread.py` | PyQt5 QThread 视频采集线程，读取摄像头帧 → 调用 Detector → NMS → 画框 → 发射信号 |
| `ui/main_window.py` | PyQt5 主界面窗口，左侧视频显示 + 右侧参数控制面板（NMS 阈值、最小人脸尺寸、检测精度、FPS/人数显示、启停按钮） |
| `main.py` | 系统入口，初始化 Detector → 启动 PyQt5 应用 → 绑定 VideoThread → 运行事件循环 |

### 2.2 对成员 B 代码的修改

在拉取 B 的更新后，我做了以下修改：

#### `detect/detector.py`
- **保留占位模式**：当 `cascade_model.pkl` 不存在时，自动使用 OpenCV 内置 Haar Cascade 作为占位检测器
- **合并 NMS 调用**：在 `detect()` 方法内部调用 NMS，对外接口统一返回 `[[x,y,w,h],...]`
- **添加 `is_placeholder` 属性**：UI 层据此显示"占位模式"或"Viola-Jones 真实模式"

#### `detect/cascade_classifier.py`
- **修复双重归一化 bug**：B 的代码在 `_detect_single_scale` 中先对 `scaled_x1/scaled_y1` 做了 `* scale` 归一化，又在返回前对 `x1/y1` 做了 `* scale`，导致框位置偏移。我删除了 `_detect_single_scale` 中的归一化，只保留返回前的归一化

#### `train/` 目录
- **修复相对导入**：B 的 `train_cascade.py` 使用 `from cascade_trainer import ...` 导致 `python train/train_cascade.py` 无法运行。我改为 `from train.cascade_trainer import ...`，确保模块可独立执行

### 2.3 性能优化

| 优化 | 效果 |
|------|------|
| 缩小检测图像（640×480 → 320×240） | 速度提升 ~3.8 倍 |
| 步长加大 + 跳帧检测（每 2 帧检测 1 次） | 速度再提升 ~7.6 倍 |
| 提高默认最小人脸尺寸（24px） | 减少小窗口误检 |
| 关闭冗余日志输出 | 减少 I/O 开销 |

### 2.4 UI 设计

- **清新风格**：浅灰蓝背景 + 纯白卡片 + 薄荷绿/天蓝主色调
- **中文界面**：所有标签、按钮、提示均为中文
- **微软雅黑字体**：14px 全局字体，控制面板各控件独立加大字号
- **实时参数调节**：NMS 阈值、最小人脸尺寸、检测精度（速度/平衡/精度三档）滑动条实时生效
- **实时状态显示**：FPS、检测人数、分辨率

---

## 三、队友测试指南

### 3.1 快速启动

```bash
cd face_detection_system
pip install -r requirements.txt
python main.py
```

### 3.2 测试占位模式（无需模型文件）

```bash
# 将 models/cascade_model.pkl 重命名或删除
mv models/cascade_model.pkl models/cascade_model.pkl.bak
python main.py
```

预期：系统自动使用 OpenCV Haar Cascade 占位检测器，UI 显示"占位模式"。

### 3.3 测试真实模式（需要模型文件）

```bash
# 恢复模型文件
mv models/cascade_model.pkl.bak models/cascade_model.pkl
python main.py
```

预期：系统加载 `cascade_model.pkl`，UI 显示"Viola-Jones 真实模式"。

### 3.4 测试 NMS 工具（独立测试）

```bash
python -c "
from utils.nms import nms
boxes = [[10,10,50,50], [15,15,50,50], [100,100,40,40]]
result = nms(boxes, iou_threshold=0.5)
print('NMS 结果:', result)
"
```

预期：前两个框 IoU 过高被合并，只保留一个。

### 3.5 测试 UI 参数调节

启动后，在右侧控制面板：
1. **NMS 阈值滑动条**：拖动时数值实时变化，检测框数量随之变化
2. **最小人脸尺寸滑动条**：调大后小尺寸检测框消失
3. **检测精度下拉框**（真实模式）：选择"速度优先"帧率最高，"精度优先"检测最准
4. **启停按钮**：点击停止检测，视频继续但不再画框

### 3.6 测试退出确认

点击"退出程序"按钮，应弹出清新风格确认对话框，点击"是"安全退出。

---

## 四、后续 B 的模型更新流程

当 B 训练出新的 `cascade_model.pkl`：

```bash
# 1. B 将新模型放到 models/ 目录
cp new_model.pkl models/cascade_model.pkl

# 2. 直接运行，无需改任何代码
python main.py
```

系统会自动检测到模型文件并切换到真实模式。如果模型结构有变化（如级联层数不同），只需确保 `cascade_classifier.py` 的 `CascadeClassifier.load()` 接口兼容即可。

---

## 五、文件变更清单

### 新增文件（我写的）
```
utils/nms.py          # NMS 工具
ui/video_thread.py    # 视频采集线程
ui/main_window.py     # 主界面窗口
main.py               # 系统入口
docs/成员C工作总结.md  # 本文档
```

### 修改文件（我改的）
```
detect/detector.py          # 添加占位模式 + NMS 集成
detect/cascade_classifier.py # 修复双重归一化 bug
train/train_cascade.py      # 修复相对导入
```

### 未修改文件（成员 A/B 的原始代码）
```
train/data_loader.py
train/integral_image.py
train/haar_features.py
train/adaboost.py
train/cascade_trainer.py
detect/__init__.py
train/__init__.py
```
