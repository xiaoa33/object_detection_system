"""
data_loader.py
==============
数据加载与准备模块（Viola-Jones 算法复现）

本文件位于 train/ 目录下，负责：
    1. 正样本准备：下载 LFW funneled 数据集，处理为 24×24 灰度图
    2. 负样本准备：下载 COCO 2017 val 数据集，裁取 24×24 非人脸子窗口
    3. 测试集准备：整理 MIT+CMU 正面人脸测试集图像与标注
    4. 数据读取接口：供训练脚本调用，返回正负样本 numpy 数组

数据保存目录（与 train/ 同级的 data/ 文件夹）：
    data/
    ├── downloads/        原始数据集压缩包缓存
    ├── train/
    │   ├── positive/    4000 张 24×24 灰度 PNG（方差归一化后保存）
    │   └── negative/   10000 张 24×24 灰度 PNG（方差归一化后保存）
    ├── val/
    │   ├── positive/     500 张 24×24 灰度 PNG
    │   └── negative/    2000 张 24×24 灰度 PNG
    └── test/
        ├── images/       完整测试图像（灰度 JPG，不做 24×24 处理）
        ├── annotations/  XML 标注文件
        └── all_faces.json  汇总标注

核心逻辑：
    - 若目标目录内文件数量已达标 → 直接跳过，不重复处理
    - 若文件不足 → 自动下载原始数据集并处理

关于方差归一化（论文 Section 5.4）：
    论文公式：σ² = m² - (1/N)Σx²，其中 m 是均值
    等价于统计学常用写法：σ² = E[x²] - (E[x])²，完全一致。
    归一化操作：normed = (x - mean) / std
    本脚本在保存时做一次归一化，训练时读取后直接使用，不再重复归一化。
    检测时对每个子窗口独立做相同归一化，与训练保持一致。

关于 LFW 数据集：
    LFW funneled 版本图像已由 Viola-Jones 检测器居中，并扩大 2.2 倍，
    效果与论文规范（人脸区域向外扩大 50%）类似，直接缩放到 24×24 即可，
    不需要再做中心裁剪。

关于 MIT+CMU 测试集：
    可自动下载，地址为：
        图像 tar 包: https://www.cs.cmu.edu/afs/cs/project/vision/vasc/idb/images/face/frontal_images/images.tar
        标注文件:    https://www.cs.cmu.edu/afs/cs/project/vision/vasc/idb/images/face/frontal_images/list.html
    标注格式为关键点坐标文本（非 XML），每行：
        文件名  左眼x 左眼y  右眼x 右眼y  鼻x 鼻y  嘴左x 嘴左y  嘴中x 嘴中y  嘴右x 嘴右y
    脚本从两眼/鼻关键点推算人脸 bounding box（x, y, w, h）。
    也支持手动下载后放入 data/test/ 目录跳过自动下载。

使用方式：
    # 作为脚本直接运行（准备全部数据）
    python train/data_loader.py

    # 只准备某一部分
    python train/data_loader.py --only positive
    python train/data_loader.py --only negative
    python train/data_loader.py --only test

    # 作为模块调用（供训练脚本使用）
    from data_loader import load_train_data, load_val_data, load_test_data
    X_pos, X_neg = load_train_data()   # 返回 float32 ndarray (N, 24, 24)

参考论文：Viola & Jones, "Robust Real-Time Face Detection", IJCV 2004
"""

import os
import sys
import json
import random
import tarfile
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# ============================================================
# 路径配置
# ============================================================

# train/data_loader.py 所在目录
TRAIN_DIR = Path(__file__).parent.resolve()

# data/ 目录与 train/ 同级
DATA_DIR = TRAIN_DIR.parent / "data"

# 各子目录
TRAIN_POS_DIR = DATA_DIR / "train" / "positive"
TRAIN_NEG_DIR = DATA_DIR / "train" / "negative"
VAL_POS_DIR   = DATA_DIR / "val"   / "positive"
VAL_NEG_DIR   = DATA_DIR / "val"   / "negative"
TEST_IMG_DIR  = DATA_DIR / "test"  / "images"
TEST_ANN_DIR  = DATA_DIR / "test"  / "annotations"
TEST_JSON     = DATA_DIR / "test"  / "all_faces.json"

# 原始数据集下载缓存目录
DOWNLOAD_DIR  = DATA_DIR / "downloads"

# ============================================================
# 数量配置
# ============================================================

TRAIN_POS_COUNT   = 4000    # 训练集正样本数量（论文使用约 4916 张）
TRAIN_NEG_COUNT   = 10000   # 训练集负样本数量（论文从 9500 张图随机裁取）
VAL_POS_COUNT     = 500     # 验证集正样本数量
VAL_NEG_COUNT     = 2000    # 验证集负样本数量

OUTPUT_SIZE       = 24      # 目标图像尺寸（论文规定 24×24）
PATCHES_PER_IMAGE = 8       # 每张 COCO 图最多裁取的 patch 数
MIN_VAR_THRESHOLD = 50.0    # patch 最低方差（过滤纯色/过暗区域）

SEED = 42  # 随机种子，保证可复现

# ============================================================
# 下载地址
# ============================================================

# LFW funneled：提供多个备选地址，自动逐一尝试
LFW_URLS = [
    "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz",   # 官方（http，部分网络可访问）
    "https://ndownloader.figstatic.com/files/5976018",     # figshare 镜像
]
LFW_ARCHIVE = "lfw-funneled.tgz"
LFW_SUBDIR  = "lfw"  # 解压后子目录名

# COCO val2017：官方 CDN，一般较稳定
COCO_IMG_URL    = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNO_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_ZIP    = "val2017.zip"
COCO_ANNO_ZIP   = "annotations_trainval2017.zip"
COCO_IMG_SUBDIR = "val2017"
COCO_ANNO_FILE  = "annotations/instances_val2017.json"

# MIT+CMU：正确地址（旧地址 vasc.ri.cmu.edu 已失效，现托管于 cs.cmu.edu/afs/...）
MITCMU_BASE_URL  = "https://www.cs.cmu.edu/afs/cs/project/vision/vasc/idb/images/face/frontal_images"
MITCMU_TAR_URL   = f"{MITCMU_BASE_URL}/images.tar"   # 全部图像（GIF格式，约3MB）
MITCMU_LIST_URL  = f"{MITCMU_BASE_URL}/list.html"    # 人脸关键点标注文本
MITCMU_TAR_FILE  = "MIT_CMU_images.tar"
MITCMU_LIST_FILE = "MIT_CMU_list.txt"


# ============================================================
# 公共工具函数
# ============================================================

def _reporthook(block_num: int, block_size: int, total_size: int):
    """urllib 下载进度回调，在同一行更新进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        pct     = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / 1024 / 1024
        mb_tot  = total_size / 1024 / 1024
        print(f"\r    进度: {mb_done:.1f} / {mb_tot:.1f} MB ({pct:.1f}%)",
              end="", flush=True)


def _download(urls, save_path: Path, label: str) -> bool:
    """
    下载文件，支持多个备选 URL，逐一尝试直到成功。
    文件已存在则直接跳过（返回 True），全部 URL 失败则返回 False。

    参数：
        urls      : 字符串或字符串列表，按优先级排列的下载地址
        save_path : 本地保存路径（Path 对象）
        label     : 日志标签（用于打印提示）
    """
    if isinstance(urls, str):
        urls = [urls]

    # 文件已存在且非空，直接跳过
    if save_path.exists() and save_path.stat().st_size > 0:
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"  [{label}] 文件已存在，跳过下载: {save_path.name} ({size_mb:.1f} MB)")
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for i, url in enumerate(urls):
        print(f"  [{label}] 尝试下载（地址 {i+1}/{len(urls)}）:")
        print(f"    {url}")
        try:
            urllib.request.urlretrieve(str(url), str(save_path), _reporthook)
            print()  # 换行（进度条最后一行）
            size_mb = save_path.stat().st_size / 1024 / 1024
            print(f"  [{label}] 下载成功，大小: {size_mb:.1f} MB → {save_path}")
            return True
        except Exception as e:
            print(f"\n  [{label}] 下载失败: {e}")
            # 删除不完整文件，防止下次误判为已存在
            if save_path.exists():
                save_path.unlink()

    return False


def _count_files(directory: Path, ext: str = ".png") -> int:
    """统计目录内指定扩展名的文件数量，目录不存在返回 0"""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() == ext)


def _variance_normalize(img: np.ndarray) -> np.ndarray:
    """
    方差归一化（论文 Section 5.4）。

    论文公式：σ² = m² - (1/N)Σx²，其中 m 为均值
    与统计学标准写法 σ² = E[x²] - (E[x])² 完全等价（代入 m = E[x] 即可验证）。
    归一化操作：normed = (x - mean) / std

    归一化只执行一次：
        - 正样本/负样本保存到磁盘前调用本函数
        - 训练时读取磁盘文件后，直接转 float32 使用，不再重复归一化
        - 检测时对每个子窗口独立调用本函数，与训练保持一致

    参数：
        img : uint8 或 float32 数组（任意形状）

    返回：
        float32 归一化数组，均值≈0，标准差≈1
        若方差极小（纯色图像），返回全零数组以避免除零
    """
    img  = img.astype(np.float32)
    mean = img.mean()
    std  = img.std()

    if std < 1e-6:
        # 纯色图像，标准差为零，无法归一化
        return np.zeros_like(img, dtype=np.float32)

    return (img - mean) / std


def _save_normalized_patch(normed: np.ndarray, save_path: Path):
    """
    将方差归一化后的 float32 图像保存为 uint8 PNG。

    归一化后值域约为 [-3, +3]，线性映射到 [0, 255]（min→0，max→255）。
    这保留了像素间的相对大小关系，Haar-like 特征（差值计算）不受影响。
    读取时直接转 float32 使用，无需再次归一化。

    参数：
        normed    : float32 归一化数组，形状 (24, 24)
        save_path : PNG 保存路径
    """
    lo, hi = normed.min(), normed.max()
    if hi - lo < 1e-6:
        # 全零图像（纯色图归一化结果），保存为全黑
        img_u8 = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    else:
        img_u8 = ((normed - lo) / (hi - lo) * 255).astype(np.uint8)
    cv2.imencode('.png', img_u8)[1].tofile(str(save_path))


# ============================================================
# 正样本：LFW funneled
# ============================================================

def _prepare_positives():
    """
    准备正样本（LFW funneled → 24×24 灰度 PNG）。

    检查：训练集和验证集目录内文件数均已达标 → 直接跳过。

    处理流程（对应论文 Section 5.1 & 5.4）：
        1. 下载 LFW funneled .tgz（约 174 MB），逐一尝试备选地址
           若全部失败，打印手动下载说明后退出
        2. 解压，得到 lfw_funneled/ 目录（人名子目录→jpg 图像）
        3. 遍历所有 jpg：读取 → 转灰度 → 缩放到 24×24 → 方差归一化
           注意：LFW 已由 Viola-Jones 居中并扩大 2.2 倍，无需额外中心裁剪
        4. 随机打乱后按 4000/500 划分，保存为 PNG

    手动下载说明（自动下载失败时）：
        下载地址: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
        将压缩包放到: data/downloads/lfw-funneled.tgz
        或将解压后的目录放到: data/downloads/lfw_funneled/
    """
    train_ok = _count_files(TRAIN_POS_DIR) >= TRAIN_POS_COUNT
    val_ok   = _count_files(VAL_POS_DIR)   >= VAL_POS_COUNT

    if train_ok and val_ok:
        print("[正样本] 文件已存在且数量充足，跳过处理")
        print(f"  训练集: {_count_files(TRAIN_POS_DIR)} 张 → {TRAIN_POS_DIR}")
        print(f"  验证集: {_count_files(VAL_POS_DIR)} 张 → {VAL_POS_DIR}")
        return

    print("[正样本] 开始准备 LFW funneled 数据集...")
    random.seed(SEED)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    archive  = DOWNLOAD_DIR / LFW_ARCHIVE
    lfw_root = DOWNLOAD_DIR / LFW_SUBDIR

    # 检查是否已手动解压
    if lfw_root.exists() and any(lfw_root.iterdir()):
        print(f"  [正样本] 检测到已解压目录，直接使用: {lfw_root}")
    else:
        # 尝试自动下载
        ok = _download(LFW_URLS, archive, "LFW funneled")
        if not ok:
            print("\n  [正样本] 所有下载地址均失败！")
            print("  请手动下载 LFW funneled 数据集：")
            print("    下载地址: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz")
            print(f"   将压缩包放到: {archive}")
            print("    或将解压后的 lfw_funneled/ 目录放到:")
            print(f"    {lfw_root}")
            sys.exit(1)

        # 解压
        print(f"  [正样本] 解压 LFW 数据集（约 174 MB，需要一些时间）...")
        with tarfile.open(str(archive), "r:gz") as tar:
            members = tar.getmembers()
            for m in tqdm(members, desc="    解压进度", unit="文件"):
                tar.extract(m, str(DOWNLOAD_DIR))
        print(f"  [正样本] 解压完成 → {lfw_root}")

    # 收集所有 jpg 图像路径
    print("  [正样本] 扫描 LFW 目录...")
    all_paths = []
    for person_dir in sorted(lfw_root.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() == ".jpg":
                all_paths.append(img_path)

    total_needed = TRAIN_POS_COUNT + VAL_POS_COUNT
    print(f"  [正样本] 发现 {len(all_paths)} 张图像，需要 {total_needed} 张")

    if len(all_paths) < total_needed:
        print(f"  [错误] 图像数量不足（{len(all_paths)} < {total_needed}），"
              f"请检查 LFW 数据集是否完整")
        sys.exit(1)

    # 随机打乱，依次处理并保存
    random.shuffle(all_paths)
    TRAIN_POS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_POS_DIR.mkdir(parents=True, exist_ok=True)

    train_cnt = val_cnt = skip_cnt = 0
    print(f"  [正样本] 处理图像：转灰度 → 缩放 24×24 → 方差归一化 → 保存 PNG")

    for img_path in tqdm(all_paths, desc="    处理进度", unit="张"):
        if train_cnt >= TRAIN_POS_COUNT and val_cnt >= VAL_POS_COUNT:
            break

        # 读取原图并转灰度
        try:
            img = cv2.imdecode(
                np.frombuffer(img_path.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
        except Exception:
            img = None
        if img is None:
            skip_cnt += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 缩放到 24×24（INTER_AREA 插值，缩小时效果最好）
        # LFW 已居中并留有边缘，无需额外裁剪
        resized = cv2.resize(gray, (OUTPUT_SIZE, OUTPUT_SIZE),
                             interpolation=cv2.INTER_AREA)

        # 方差归一化（论文 Section 5.4，此处是唯一一次归一化）
        normed = _variance_normalize(resized)

        # 分配到训练集或验证集
        if train_cnt < TRAIN_POS_COUNT:
            save_path = TRAIN_POS_DIR / f"face_{train_cnt:05d}.png"
            train_cnt += 1
        else:
            save_path = VAL_POS_DIR / f"face_{val_cnt:05d}.png"
            val_cnt += 1

        _save_normalized_patch(normed, save_path)

    print(f"  [正样本] 完成！")
    print(f"    训练集正样本: {train_cnt} 张 → {TRAIN_POS_DIR}")
    print(f"    验证集正样本: {val_cnt} 张 → {VAL_POS_DIR}")
    if skip_cnt:
        print(f"    读取失败跳过: {skip_cnt} 张")


# ============================================================
# 负样本：COCO val2017 非人脸图像
# ============================================================

def _load_non_person_image_paths(anno_path: Path, images_dir: Path) -> list:
    """
    读取 COCO instances_val2017.json，筛选不含 person 的图像路径。

    COCO person category_id = 1（标准值，代码中从标注文件动态读取）。
    凡标注中含 person 的图像全部排除，确保负样本不含人脸区域。

    参数：
        anno_path  : instances_val2017.json 文件路径
        images_dir : COCO val2017 图像目录

    返回：
        不含 person 的图像 Path 列表
    """
    print(f"  [负样本] 读取 COCO 标注: {anno_path.name}")
    with open(str(anno_path), "r") as f:
        coco = json.load(f)

    # 动态获取 person 的 category_id（标准 COCO 中为 1）
    person_ids = {cat["id"] for cat in coco["categories"]
                  if cat["name"] == "person"}
    print(f"    person category_id = {person_ids}")

    # 含 person 标注的图像 id 集合（快速查找）
    imgs_with_person = {ann["image_id"] for ann in coco["annotations"]
                        if ann["category_id"] in person_ids}

    # 筛选：不含 person 且图像文件实际存在
    result = []
    for img_info in coco["images"]:
        if img_info["id"] not in imgs_with_person:
            p = images_dir / img_info["file_name"]
            if p.exists():
                result.append(p)

    print(f"    总图像数: {len(coco['images'])}")
    print(f"    含 person: {len(imgs_with_person)} 张（已排除）")
    print(f"    可用非人脸图: {len(result)} 张")
    return result


def _extract_patches(img_path: Path, n: int) -> list:
    """
    从单张图像随机裁取最多 n 个 24×24 灰度 patch，并做方差归一化。

    质量过滤：方差低于 MIN_VAR_THRESHOLD 的 patch 被视为纯色区域（天空、墙壁等）
    丢弃，使负样本具有足够的纹理多样性，避免分类器学到过于简单的边界。

    参数：
        img_path : 图像文件路径
        n        : 最多裁取数量

    返回：
        float32 ndarray 列表，每个形状 (24, 24)，已方差归一化
    """
    try:
        img = cv2.imdecode(
            np.frombuffer(img_path.read_bytes(), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
    except Exception:
        img = None
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    if h < OUTPUT_SIZE or w < OUTPUT_SIZE:
        return []  # 图像过小，无法裁取

    patches      = []
    max_attempts = n * 5  # 最多尝试次数，补偿质量过滤的丢弃

    for _ in range(max_attempts):
        if len(patches) >= n:
            break

        # 随机选取左上角坐标
        y = random.randint(0, h - OUTPUT_SIZE)
        x = random.randint(0, w - OUTPUT_SIZE)
        patch = gray[y:y + OUTPUT_SIZE, x:x + OUTPUT_SIZE].astype(np.float32)

        # 质量过滤：丢弃纯色/低纹理区域
        if np.var(patch) < MIN_VAR_THRESHOLD:
            continue

        # 方差归一化（与正样本处理完全一致，仅做一次）
        patches.append(_variance_normalize(patch))

    return patches


def _prepare_negatives():
    """
    准备负样本（COCO val2017 非人脸 patch → 24×24 灰度 PNG）。

    检查：训练集和验证集目录内文件数均已达标 → 直接跳过。

    处理流程（对应论文 Section 5.1）：
        1. 下载 COCO val2017 图像（约 1 GB）和 annotations（约 240 MB）
        2. 解压后读取 instances_val2017.json，筛选不含 person 的图像
        3. 每张图随机裁取最多 PATCHES_PER_IMAGE 个 24×24 patch
        4. 方差过滤（MIN_VAR_THRESHOLD）+ 方差归一化
        5. 按 10000/2000 划分，保存为 PNG

    关于 Hard Negative Mining（论文 Section 4.1 & Table 2）：
        Hard Negative Mining 是第二阶段负样本获取，由 cascade_trainer.py 执行：
        用已训练的前 i 层级联扫描非人脸图，收集误检子窗口作为下一层负样本。
        本脚本只负责第一阶段初始负样本的准备。
    """
    train_ok = _count_files(TRAIN_NEG_DIR) >= TRAIN_NEG_COUNT
    val_ok   = _count_files(VAL_NEG_DIR)   >= VAL_NEG_COUNT

    if train_ok and val_ok:
        print("[负样本] 文件已存在且数量充足，跳过处理")
        print(f"  训练集: {_count_files(TRAIN_NEG_DIR)} 个 → {TRAIN_NEG_DIR}")
        print(f"  验证集: {_count_files(VAL_NEG_DIR)} 个 → {VAL_NEG_DIR}")
        return

    print("[负样本] 开始准备 COCO val2017 负样本...")
    random.seed(SEED)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 步骤 1：下载 COCO 图像（约 1 GB）
    img_zip = DOWNLOAD_DIR / COCO_IMG_ZIP
    if not _download(COCO_IMG_URL, img_zip, "COCO val2017 图像"):
        print("  [负样本] 下载失败，请手动下载：")
        print(f"    {COCO_IMG_URL}")
        print(f"    保存到: {img_zip}")
        sys.exit(1)

    # 步骤 2：下载 COCO 标注（约 240 MB）
    anno_zip = DOWNLOAD_DIR / COCO_ANNO_ZIP
    if not _download(COCO_ANNO_URL, anno_zip, "COCO annotations"):
        print("  [负样本] 下载失败，请手动下载：")
        print(f"    {COCO_ANNO_URL}")
        print(f"    保存到: {anno_zip}")
        sys.exit(1)

    # 步骤 3：解压图像
    images_dir = DOWNLOAD_DIR / COCO_IMG_SUBDIR
    if not images_dir.exists():
        print("  [负样本] 解压 COCO 图像（约 1 GB，需要几分钟）...")
        with zipfile.ZipFile(str(img_zip), "r") as zf:
            for m in tqdm(zf.namelist(), desc="    解压图像", unit="文件"):
                zf.extract(m, str(DOWNLOAD_DIR))
        print(f"  [负样本] 图像解压完成 → {images_dir}")
    else:
        print(f"  [负样本] 图像目录已存在，跳过解压: {images_dir.name}")

    # 步骤 4：解压标注
    anno_check = DOWNLOAD_DIR / "annotations"
    if not anno_check.exists():
        print("  [负样本] 解压 COCO 标注...")
        with zipfile.ZipFile(str(anno_zip), "r") as zf:
            for m in tqdm(zf.namelist(), desc="    解压标注", unit="文件"):
                zf.extract(m, str(DOWNLOAD_DIR))
        print("  [负样本] 标注解压完成")
    else:
        print("  [负样本] 标注目录已存在，跳过解压")

    # 步骤 5：筛选非人脸图像
    anno_path = DOWNLOAD_DIR / COCO_ANNO_FILE
    img_paths = _load_non_person_image_paths(anno_path, images_dir)
    random.shuffle(img_paths)

    print(f"  [负样本] 预计可提取约 {len(img_paths) * PATCHES_PER_IMAGE} 个 patch")
    print(f"  [负样本] 目标: 训练集 {TRAIN_NEG_COUNT} 个 / 验证集 {VAL_NEG_COUNT} 个")

    # 步骤 6：提取 patch 并保存
    TRAIN_NEG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_NEG_DIR.mkdir(parents=True, exist_ok=True)

    train_cnt = val_cnt = 0
    pbar = tqdm(img_paths, desc="    提取 patch", unit="张图像")

    for img_path in pbar:
        if train_cnt >= TRAIN_NEG_COUNT and val_cnt >= VAL_NEG_COUNT:
            break

        patches = _extract_patches(img_path, PATCHES_PER_IMAGE)

        for patch in patches:
            if train_cnt < TRAIN_NEG_COUNT:
                _save_normalized_patch(patch,
                                       TRAIN_NEG_DIR / f"neg_{train_cnt:05d}.png")
                train_cnt += 1
            elif val_cnt < VAL_NEG_COUNT:
                _save_normalized_patch(patch,
                                       VAL_NEG_DIR / f"neg_{val_cnt:05d}.png")
                val_cnt += 1
            else:
                break

        pbar.set_postfix({"训练": train_cnt, "验证": val_cnt})

    print(f"  [负样本] 完成！")
    print(f"    训练集负样本: {train_cnt} 个 → {TRAIN_NEG_DIR}")
    print(f"    验证集负样本: {val_cnt} 个 → {VAL_NEG_DIR}")

    if train_cnt < TRAIN_NEG_COUNT or val_cnt < VAL_NEG_COUNT:
        print(f"  [警告] 数量不足！可调大 PATCHES_PER_IMAGE（当前 {PATCHES_PER_IMAGE}）")


# ============================================================
# 测试集：MIT+CMU
# ============================================================

def _parse_keypoint_list(list_text: str) -> dict:
    """
    解析 MIT+CMU list.html 中的关键点标注文本，保存两眼坐标。

    原始格式（每行一张人脸）：
        filename  左眼x 左眼y  右眼x 右眼y  鼻x 鼻y  嘴左x 嘴左y  嘴中x 嘴中y  嘴右x 嘴右y
    示例：
        voyager.gif 122.0 43.0 137.0 47.0 129.0 54.0 123.0 59.0 128.0 60.0 133.0 62.0

    评估标准（Rowley 1998 论文原始准则）：
        MIT+CMU 数据集本身没有提供 bounding box，只有关键点坐标。
        官方评估标准是：两眼中点（eye_midpoint）落在检测框内则判定为 TP。
        这是 Viola-Jones 论文评估该数据集时也沿用的标准。
        因此本函数直接保存两眼坐标和眼睛中点，不估算 bounding box。

    参数：
        list_text : list.html 页面的全部文本内容

    返回：
        {文件名: [{"left_eye": [lx, ly], "right_eye": [rx, ry],
                   "eye_midpoint": [cx, cy]}, ...]}
        其中 eye_midpoint = ((lx+rx)/2, (ly+ry)/2)，是评估时的判定点
    """
    annotations = {}

    for line in list_text.splitlines():
        line = line.strip()
        # 跳过空行和标题行（不含数字字符）
        if not line or not any(c.isdigit() for c in line):
            continue

        parts = line.split()
        # 格式：filename + 12 个坐标数值（6 个关键点 × 2）
        if len(parts) < 13:
            continue

        filename = parts[0]
        try:
            lx = float(parts[1])   # 左眼 x
            ly = float(parts[2])   # 左眼 y
            rx = float(parts[3])   # 右眼 x
            ry = float(parts[4])   # 右眼 y
        except (ValueError, IndexError):
            continue

        # 两眼中点（评估时的 TP 判定点）
        cx = (lx + rx) / 2.0
        cy = (ly + ry) / 2.0

        if filename not in annotations:
            annotations[filename] = []
        annotations[filename].append({
            "left_eye":     [lx, ly],
            "right_eye":    [rx, ry],
            "eye_midpoint": [cx, cy],   # 检测框包含此点 → TP
        })

    return annotations


def _save_test_json(annotations: dict):
    """
    将测试集所有标注汇总保存为 JSON，供评估脚本读取。

    JSON 结构：
    {
        "total_images": 130,
        "total_faces":  507,
        "eval_criterion": "eye_midpoint_in_box",
        "annotations": {
            "voyager.jpg": {
                "image_path": "/绝对路径/voyager.jpg",
                "faces": [
                    {
                        "left_eye":     [122.0, 43.0],
                        "right_eye":    [137.0, 47.0],
                        "eye_midpoint": [129.5, 45.0]
                    },
                    ...
                ]
            },
            ...
        }
    }

    评估方式（eval_criterion = "eye_midpoint_in_box"）：
        对每个检测到的 bounding box (x, y, w, h)：
            若某张人脸的 eye_midpoint 满足
                x <= cx <= x+w  且  y <= cy <= y+h
            则判定该人脸被检测到（TP），否则为漏检（FN）。
        不被任何 ground truth 对应的检测框计为 FP。
        这是 Rowley et al. 1998 论文的原始评估标准，
        Viola-Jones 2004 论文评估 MIT+CMU 时同样沿用此标准。
    """
    total_faces = sum(len(v) for v in annotations.values())
    summary = {
        "total_images":   len(annotations),
        "total_faces":    total_faces,
        "eval_criterion": "eye_midpoint_in_box",   # 评估准则说明
        "annotations":    {
            fname: {
                "image_path": str(TEST_IMG_DIR / fname),
                "faces":      faces   # 每个元素含 left_eye/right_eye/eye_midpoint
            }
            for fname, faces in annotations.items()
        }
    }
    TEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    TEST_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"  [测试集] 汇总 JSON 已保存: {TEST_JSON}")
    print(f"    总图像数: {summary['total_images']}")
    print(f"    总人脸数: {summary['total_faces']}")
    print(f"    评估标准: eye_midpoint_in_box（两眼中点落在检测框内 → TP）")


def _prepare_testset():
    """
    准备 MIT+CMU 正面人脸测试集（自动下载图像和标注）。

    测试集说明：
        - 图像保留完整原图（灰度 JPG），不做 24×24 处理
        - 检测器对整张图做多尺度滑动窗口扫描
        - 输出检测框与 ground truth bounding box 做 IoU 匹配，统计 TP/FP

    下载内容：
        图像 tar 包：images.tar（包含 test/、test-low/、newtest/ 三个子集）
        标注文件：list.html（关键点坐标文本，对应 Test Set A+B+C 共 507 张脸）

    标注处理：
        原始标注为每张人脸的 6 个关键点坐标（两眼、鼻、三个嘴角）。
        脚本从两眼间距推算人脸 bounding box，详见 _parse_keypoint_list()。

    三种处理情况：
        A. images/ 有图像 且 all_faces.json 存在 → 直接跳过
        B. images/ 有图像 但无 JSON → 重新下载标注并生成 JSON
        C. 无图像 → 自动下载并处理
    """
    img_count = sum(_count_files(TEST_IMG_DIR, e)
                    for e in [".jpg", ".jpeg", ".png", ".gif"])

    if img_count > 0 and TEST_JSON.exists():
        print("[测试集] 文件已存在，跳过处理")
        print(f"  图像: {img_count} 张 → {TEST_IMG_DIR}")
        print(f"  标注: {TEST_JSON}")
        return

    print("[测试集] 准备 MIT+CMU 正面人脸测试集...")
    print("  图像保留完整原图，供检测器多尺度扫描后与 ground truth 比对")

    TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TEST_ANN_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 步骤 1：下载并解压图像 tar 包
    # ----------------------------------------------------------------
    tar_path = DOWNLOAD_DIR / MITCMU_TAR_FILE

    # 检查是否已手动放置图像，若有则跳过下载
    existing = list(TEST_IMG_DIR.glob("*.jpg")) + list(TEST_IMG_DIR.glob("*.gif"))
    if existing:
        print(f"  [测试集] 检测到已有 {len(existing)} 张图像，跳过下载")
    else:
        ok = _download(MITCMU_TAR_URL, tar_path, "MIT+CMU 图像 tar")
        if not ok:
            print("  [测试集] 下载失败！请手动下载：")
            print(f"    {MITCMU_TAR_URL}")
            print(f"    解压后将图像放入: {TEST_IMG_DIR}")
            print("  [提示] 测试集缺失不影响训练，可先跳过。")
            return

        # 解压 tar（包含 test/、test-low/、newtest/ 三个子目录）
        print("  [测试集] 解压图像 tar 包...")
        import tarfile as _tarfile
        with _tarfile.open(str(tar_path), "r") as tar:
            members = tar.getmembers()
            gif_members = [m for m in members
                           if m.name.lower().endswith(".gif")]
            print(f"    tar 包内共 {len(gif_members)} 张 GIF 图像")
            for m in tqdm(gif_members, desc="    解压", unit="张"):
                tar.extract(m, str(DOWNLOAD_DIR / "mitcmu_raw"))
        print("  [测试集] 解压完成")

        # 将解压出的 GIF 统一转为灰度 JPG 保存到 TEST_IMG_DIR
        raw_dir = DOWNLOAD_DIR / "mitcmu_raw"
        gif_files = list(raw_dir.rglob("*.gif")) + list(raw_dir.rglob("*.GIF"))
        print(f"  [测试集] 转换 {len(gif_files)} 张 GIF → 灰度 JPG...")

        converted = 0
        for gif_path in tqdm(gif_files, desc="    转换", unit="张"):
            # OpenCV 可读取 GIF 的第一帧
            try:
                img = cv2.imdecode(
                    np.frombuffer(gif_path.read_bytes(), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE
                )
            except Exception:
                img = None
            if img is None:
                # OpenCV 不支持部分 GIF，尝试用 Pillow
                try:
                    from PIL import Image as PILImage
                    pil_img = PILImage.open(str(gif_path)).convert("L")
                    img = np.array(pil_img, dtype=np.uint8)
                except Exception:
                    print(f"    [跳过] 无法读取: {gif_path.name}")
                    continue

            save_name = gif_path.stem + ".jpg"
            cv2.imencode('.jpg', img)[1].tofile(str(TEST_IMG_DIR / save_name))
            converted += 1

        print(f"  [测试集] 成功转换: {converted} 张图像 → {TEST_IMG_DIR}")

    # ----------------------------------------------------------------
    # 步骤 2：下载并解析关键点标注文件
    # ----------------------------------------------------------------
    list_path = DOWNLOAD_DIR / MITCMU_LIST_FILE

    # 下载 list.html（文本内容）
    ok = _download(MITCMU_LIST_URL, list_path, "MIT+CMU 标注 list.html")
    if not ok:
        print("  [测试集] 标注下载失败，将生成空标注 JSON（评估结果无意义）")
        # 生成空标注
        img_files = list(TEST_IMG_DIR.glob("*.jpg"))
        annotations = {f.name: [] for f in img_files}
    else:
        # 解析关键点文本，转换为 bounding box
        list_text = list_path.read_text(encoding="utf-8", errors="ignore")
        print(f"  [测试集] 解析关键点标注...")
        kp_annotations = _parse_keypoint_list(list_text)

        print(f"    共解析到 {len(kp_annotations)} 张图的标注，"
              f"合计 {sum(len(v) for v in kp_annotations.values())} 张人脸")

        # 将标注文件名（如 voyager.gif）映射到实际保存的 jpg 文件名
        annotations = {}
        img_files   = list(TEST_IMG_DIR.glob("*.jpg"))
        jpg_stems   = {f.stem: f.name for f in img_files}  # stem→jpg文件名

        for gif_name, faces in kp_annotations.items():
            stem = Path(gif_name).stem
            if stem in jpg_stems:
                annotations[jpg_stems[stem]] = faces
            else:
                # 标注有，但图像未找到（可能子集不同）
                pass

        # 补充没有标注的图像（标注为空列表）
        for jpg_file in img_files:
            if jpg_file.name not in annotations:
                annotations[jpg_file.name] = []

        matched = sum(1 for v in annotations.values() if len(v) > 0)
        print(f"    与图像匹配成功: {matched} 张有标注 / "
              f"{len(annotations)} 张图像")

    # ----------------------------------------------------------------
    # 步骤 3：保存汇总 JSON
    # ----------------------------------------------------------------
    _save_test_json(annotations)


# ============================================================
# 公开数据读取接口（供训练脚本调用）
# ============================================================

def _load_images_from_dir(directory: Path, max_count: int = None) -> np.ndarray:
    """
    从目录读取所有 PNG 图像，返回 float32 numpy 数组。

    关于归一化：
        图像在保存时已做方差归一化（线性映射到 uint8 [0,255]）。
        读取时直接转 float32，不再重复归一化。
        Haar-like 特征计算基于像素差值，线性变换不改变特征判别能力。

    参数：
        directory : 图像目录（如 TRAIN_POS_DIR）
        max_count : 最多读取数量（None 表示全部）

    返回：
        float32 ndarray，形状 (N, 24, 24)，值域 [0.0, 255.0]
        目录不存在或为空时，返回 shape=(0, 24, 24) 的空数组
    """
    if not directory.exists():
        print(f"  [警告] 目录不存在: {directory}")
        return np.empty((0, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

    files = sorted(directory.glob("*.png"))
    if max_count is not None:
        files = files[:max_count]

    if not files:
        print(f"  [警告] 目录为空: {directory}")
        return np.empty((0, OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

    images = []
    for f in files:
        try:
            img = cv2.imdecode(
                np.frombuffer(f.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
        except Exception:
            img = None
        if img is None:
            continue
        # 直接转 float32，不重复归一化
        images.append(img.astype(np.float32))

    arr = np.array(images, dtype=np.float32)
    print(f"  读取 {len(arr):5d} 张图像 ← {directory.relative_to(DATA_DIR)}")
    return arr


def prepare_all_data():
    """
    一键准备全部数据（正样本 + 负样本 + 测试集）。
    已存在且数量足够的数据自动跳过，缺失则下载处理。
    可直接运行 python train/data_loader.py 调用。
    """
    print("=" * 60)
    print("  Viola-Jones 数据准备流程")
    print(f"  数据根目录: {DATA_DIR.resolve()}")
    print("=" * 60)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- 步骤 1/3：正样本（LFW funneled，约 174 MB）---")
    _prepare_positives()

    print("\n--- 步骤 2/3：负样本（COCO val2017，约 1.2 GB）---")
    _prepare_negatives()

    print("\n--- 步骤 3/3：测试集（MIT+CMU，自动下载）---")
    _prepare_testset()

    # 最终统计
    print("\n" + "=" * 60)
    print("  数据准备完成，目录统计：")
    stats = [
        (TRAIN_POS_DIR, "训练集正样本", TRAIN_POS_COUNT),
        (TRAIN_NEG_DIR, "训练集负样本", TRAIN_NEG_COUNT),
        (VAL_POS_DIR,   "验证集正样本", VAL_POS_COUNT),
        (VAL_NEG_DIR,   "验证集负样本", VAL_NEG_COUNT),
    ]
    for d, label, target in stats:
        cnt    = _count_files(d)
        status = "✓" if cnt >= target else f"✗ 不足（需 {target}）"
        print(f"  {label:12s}: {cnt:6d} 个  [{status}]")

    img_cnt = sum(_count_files(TEST_IMG_DIR, e)
                  for e in [".jpg", ".png", ".pgm", ".gif"])
    print(f"  {'测试集图像':12s}: {img_cnt:6d} 张  "
          f"[{'✓' if img_cnt > 0 else '需手动下载'}]")
    print(f"  汇总 JSON    : "
          f"{'已存在 ✓' if TEST_JSON.exists() else '待生成（先放测试图像）'}")
    print("=" * 60)


def load_train_data(max_pos: int = None, max_neg: int = None) -> tuple:
    """
    加载训练集正负样本，返回 float32 numpy 数组。
    文件不足时自动触发数据准备。

    参数：
        max_pos : 最多读取正样本数（None = 全部 4000 张）
        max_neg : 最多读取负样本数（None = 全部 10000 张）

    返回：
        (X_pos, X_neg)
            X_pos : float32 ndarray，(N_pos, 24, 24)，值域 [0, 255]
            X_neg : float32 ndarray，(N_neg, 24, 24)，值域 [0, 255]
            图像已在保存时做过方差归一化，读取后可直接用于特征计算
    """
    if (_count_files(TRAIN_POS_DIR) < TRAIN_POS_COUNT or
            _count_files(TRAIN_NEG_DIR) < TRAIN_NEG_COUNT):
        print("[load_train_data] 训练数据不完整，自动准备...")
        _prepare_positives()
        _prepare_negatives()

    print("[load_train_data] 加载训练集...")
    X_pos = _load_images_from_dir(TRAIN_POS_DIR, max_pos)
    X_neg = _load_images_from_dir(TRAIN_NEG_DIR, max_neg)
    print(f"  正样本 shape: {X_pos.shape}")
    print(f"  负样本 shape: {X_neg.shape}")
    return X_pos, X_neg


def load_val_data(max_pos: int = None, max_neg: int = None) -> tuple:
    """
    加载验证集正负样本，返回 float32 numpy 数组。
    文件不足时自动触发数据准备。

    参数：
        max_pos : 最多读取正样本数（None = 全部 500 张）
        max_neg : 最多读取负样本数（None = 全部 2000 张）

    返回：
        (X_pos, X_neg)
            X_pos : float32 ndarray，(N_pos, 24, 24)
            X_neg : float32 ndarray，(N_neg, 24, 24)
    """
    if (_count_files(VAL_POS_DIR) < VAL_POS_COUNT or
            _count_files(VAL_NEG_DIR) < VAL_NEG_COUNT):
        print("[load_val_data] 验证数据不完整，自动准备...")
        _prepare_positives()
        _prepare_negatives()

    print("[load_val_data] 加载验证集...")
    X_pos = _load_images_from_dir(VAL_POS_DIR, max_pos)
    X_neg = _load_images_from_dir(VAL_NEG_DIR, max_neg)
    print(f"  正样本 shape: {X_pos.shape}")
    print(f"  负样本 shape: {X_neg.shape}")
    return X_pos, X_neg


def load_test_data() -> dict:
    """
    加载测试集标注信息，供评估脚本使用。
    all_faces.json 不存在时自动触发整理流程。

    返回：
        dict，结构：
        {
            "total_images":   130,
            "total_faces":    507,
            "eval_criterion": "eye_midpoint_in_box",
            "annotations": {
                "voyager.jpg": {
                    "image_path": "/绝对路径/voyager.jpg",
                    "faces": [
                        {
                            "left_eye":     [122.0, 43.0],
                            "right_eye":    [137.0, 47.0],
                            "eye_midpoint": [129.5, 45.0]
                        },
                        ...
                    ]
                },
                ...
            }
        }

    评估用法示例（在 evaluate.py 中）：
        for fname, info in data["annotations"].items():
            img = cv2.imread(info["image_path"])
            detections = detector.detect(img)          # [(x,y,w,h), ...]
            for face in info["faces"]:
                cx, cy = face["eye_midpoint"]
                # 检查任意一个检测框包含该中点
                detected = any(
                    x <= cx <= x+w and y <= cy <= y+h
                    for (x, y, w, h) in detections
                )

    测试集未就绪时，返回 total_images=0 的空结构（不报错）。
    """
    if not TEST_JSON.exists():
        print("[load_test_data] 标注 JSON 不存在，尝试整理测试集...")
        _prepare_testset()

    if not TEST_JSON.exists():
        print("[load_test_data] 测试集未就绪，返回空结构")
        return {"total_images": 0, "total_faces": 0, "annotations": {}}

    print("[load_test_data] 加载测试集标注...")
    with open(str(TEST_JSON), "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  共 {data['total_images']} 张图，{data['total_faces']} 个标注人脸")
    return data


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Viola-Jones 数据准备脚本（train/data_loader.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python train/data_loader.py                  # 准备全部数据
  python train/data_loader.py --only positive  # 只准备正样本（LFW）
  python train/data_loader.py --only negative  # 只准备负样本（COCO）
  python train/data_loader.py --only test      # 只整理测试集

MIT+CMU 测试集下载地址（自动下载，也可手动）：
  图像: https://www.cs.cmu.edu/afs/cs/project/vision/vasc/idb/images/face/frontal_images/images.tar
  标注: https://www.cs.cmu.edu/afs/cs/project/vision/vasc/idb/images/face/frontal_images/list.html
  手动下载后图像放入 data/test/images/，标注文件放入 data/downloads/MIT_CMU_list.txt
        """
    )
    parser.add_argument(
        "--only",
        choices=["positive", "negative", "test"],
        help="只准备指定部分（默认准备全部）"
    )
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.only == "positive":
        print("--- 仅准备正样本（LFW funneled）---")
        _prepare_positives()
    elif args.only == "negative":
        print("--- 仅准备负样本（COCO val2017）---")
        _prepare_negatives()
    elif args.only == "test":
        print("--- 仅整理测试集（MIT+CMU）---")
        _prepare_testset()
    else:
        prepare_all_data()