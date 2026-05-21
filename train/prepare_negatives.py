"""
prepare_negatives.py
==========================
负样本大图准备模块（Viola-Jones 算法复现）—— 基于 Intel Image Classification 数据集

本文件位于 train/ 目录下，作为 prepare_negatives.py 的补充脚本，
专门处理 Intel Image Classification 数据集中的无人脸场景图像。

处理逻辑：
    1. 从 Intel 数据集的 seg_train / seg_test 两个分割中，
       读取指定类别（forest / glacier / mountain / sea / buildings）的图像
    2. 用 OpenCV Haar 级联检测器做严格人脸过滤（最严格参数：minNeighbors=2）
    3. 通过过滤的图像原图复制到目标目录
    4. 训练集目标 9000 张，验证集目标 1000 张，合计 10000 张
    5. 数量不足时打印警告，不报错退出

使用的类别（共 5 类，均无明显人脸风险）：
    - forest    / 森林   （2271 张 train + 测试集部分）
    - glacier   / 冰川   （2404 张 train + 测试集部分）
    - mountain  / 山峰   （2512 张 train + 测试集部分）
    - sea       / 海洋   （2274 张 train + 测试集部分）
    - buildings / 建筑   （2191 张 train + 测试集部分，偶有远景路人，检测器会过滤）
    排除的类别：street（街道，含行人风险高）

关于 street 类别不纳入的原因：
    street 类别街道场景中行人比例较高，即使 Haar 检测器可以过滤掉正面人脸，
    但侧脸、背影等仍可能进入负样本库，对训练产生轻微干扰。
    其余 5 类合计约 11,000+ 张，数量已充足，无需冒险纳入 street。

关于人体但无明显人脸的情况：
    Viola-Jones Haar 检测器专门检测正面清晰人脸，对以下情况基本不触发：
    - 远景小人（分辨率不足 minSize 阈值）
    - 侧脸或背影
    - 遮挡严重的人脸
    因此 buildings 类别中偶尔出现的远景路人不会影响过滤结果，直接保留。

保存目录（与 data_loader.py / prepare_negatives.py 路径常量完全一致）：
    data/
    ├── downloads/
    │   └── intel/                  ← 手动解压后的 Intel 数据集根目录
    │       ├── seg_train/          ← 训练集（按类别子目录）
    │       │   ├── forest/
    │       │   ├── glacier/
    │       │   ├── mountain/
    │       │   ├── sea/
    │       │   ├── buildings/
    │       │   └── street/         ← 不使用
    │       └── seg_test/           ← 测试集（按类别子目录，结构同上）
    │           ├── forest/
    │           ├── glacier/
    │           ├── mountain/
    │           ├── sea/
    │           └── buildings/
    ├── train/
    │   └── negative/    9000 张原始大图（续接已有编号）
    └── val/
        └── negative/    1000 张原始大图（续接已有编号）

Intel Image Classification 数据集下载：
    Kaggle 下载地址（需登录 Kaggle，约 350 MB）：
        https://www.kaggle.com/datasets/puneet6060/intel-image-classification
    下载后解压，确保目录结构为：
        data/downloads/intel/seg_train/forest/xxx.jpg
        data/downloads/intel/seg_test/glacier/xxx.jpg
        ...

关于文件命名续接：
    本脚本检测 train/negative/ 和 val/negative/ 目录中现有的最大序号，
    从该序号 +1 开始命名新文件（neg_XXXXX.jpg），
    因此可与 prepare_negatives.py 的输出无缝合并，不会覆盖已有文件。

关于人脸检测参数（最严格模式）：
    scaleFactor=1.05   每次缩放步长更小，检测更细致（比默认 1.1 更严格）
    minNeighbors=2     只需 2 个相邻矩形即判定为人脸（比默认 3 更敏感）
    minSize=(20, 20)   最小检测尺寸 20px（比默认 30px 更小，能检测更远的人脸）
    效果：宁可误杀，不放过。过滤率会高于 prepare_negatives.py，但安全性更高。

使用方式：
    # 直接运行（使用默认路径 data/downloads/intel/）
    python train/prepare_negatives_intel.py

    # 指定 Intel 数据集根目录
    python train/prepare_negatives_intel.py --intel-dir /path/to/intel

    # 自定义目标数量（从现有文件续接）
    python train/prepare_negatives_intel.py --train-count 9000 --val-count 1000

参考论文：Viola & Jones, "Robust Real-Time Face Detection", IJCV 2004
参考数据集：Intel Image Classification Challenge, Analytics Vidhya / Kaggle, 2018
"""

import sys
import shutil
import random
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ============================================================
# 路径配置（与 data_loader.py / prepare_negatives.py 完全一致）
# ============================================================

# train/prepare_negatives_intel.py 所在目录
TRAIN_DIR = Path(__file__).parent.resolve()

# data/ 目录与 train/ 同级
DATA_DIR = TRAIN_DIR.parent / "data"

# 负样本保存目录（与 data_loader.py 中 TRAIN_NEG_DIR / VAL_NEG_DIR 完全一致）
TRAIN_NEG_DIR = DATA_DIR / "train" / "negative"
VAL_NEG_DIR   = DATA_DIR / "val"   / "negative"

# 原始数据集目录
DOWNLOAD_DIR  = DATA_DIR / "downloads"

# Intel 数据集默认解压路径
INTEL_DIR = DOWNLOAD_DIR

# ============================================================
# 数量配置
# ============================================================

TRAIN_NEG_COUNT = 9000    # 训练集负样本大图总目标（含已有）
VAL_NEG_COUNT   = 1000    # 验证集负样本大图总目标（含已有）

SEED = 42   # 随机种子

# ============================================================
# Intel 数据集配置
# ============================================================

# 使用的分割目录名（seg_train + seg_test 均读取，最大化利用数据量）
INTEL_SPLITS = ["seg_train/seg_train", "seg_test/seg_test"]

# 使用的类别（排除 street）
INTEL_CATEGORIES = ["forest", "glacier", "mountain", "sea", "buildings"]

# ============================================================
# 人脸检测器配置（最严格模式）
# ============================================================

HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 最严格参数：宁可多误杀，保证负样本库干净
FACE_DETECT_SCALE_FACTOR  = 1.05   # 步长更小，检测更细致
FACE_DETECT_MIN_NEIGHBORS = 2      # 更敏感，只需 2 个相邻矩形
FACE_DETECT_MIN_SIZE      = (20, 20)  # 最小检测尺寸更小，能发现远处人脸


# ============================================================
# 工具函数
# ============================================================

def _count_files(directory: Path, exts: tuple = (".jpg", ".jpeg", ".png")) -> int:
    """统计目录内指定扩展名的文件数量，目录不存在返回 0。"""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def _get_max_index(directory: Path) -> int:
    """
    获取目录中 neg_XXXXX.jpg 格式文件的最大序号。
    目录为空或无匹配文件时返回 -1（下一个序号从 0 开始）。

    用于续接 prepare_negatives.py 已生成的文件，避免覆盖。
    """
    max_idx = -1
    if not directory.exists():
        return max_idx
    for f in directory.iterdir():
        if f.stem.startswith("neg_") and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
            try:
                idx = int(f.stem[4:])   # "neg_00123" → 123
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass
    return max_idx


def _build_face_detector() -> cv2.CascadeClassifier:
    """
    加载 OpenCV 内置 Haar 级联人脸检测器。

    Windows 上 OpenCV C++ 层无法处理含中文字符的路径，
    解决方案：将 xml 文件复制到系统临时目录（纯 ASCII 路径）后再加载。

    本脚本使用最严格的检测参数：
        scaleFactor=1.05, minNeighbors=2, minSize=(20,20)
    比 prepare_negatives.py 的默认参数更敏感，过滤率更高。

    返回：
        cv2.CascadeClassifier 对象
    """
    src_path = Path(HAAR_CASCADE_PATH)
    if not src_path.exists():
        print(f"[错误] 找不到 OpenCV Haar 级联文件: {HAAR_CASCADE_PATH}")
        print("  请确认 OpenCV 安装正确。安装命令: pip install opencv-python")
        sys.exit(1)

    # 判断路径是否含非 ASCII 字符（Windows 中文路径问题）
    load_path = str(src_path)
    try:
        load_path.encode("ascii")
        needs_copy = False
    except UnicodeEncodeError:
        needs_copy = True

    if needs_copy:
        tmp_dir = Path(tempfile.gettempdir())
        tmp_xml = tmp_dir / "haarcascade_frontalface_default.xml"
        if not tmp_xml.exists():
            shutil.copy2(str(src_path), str(tmp_xml))
            print(f"  [人脸检测器] 路径含非 ASCII 字符，已复制到临时目录: {tmp_xml}")
        else:
            print(f"  [人脸检测器] 使用已缓存的临时文件: {tmp_xml}")
        load_path = str(tmp_xml)
    else:
        print(f"  [人脸检测器] 直接加载: {load_path}")

    detector = cv2.CascadeClassifier(load_path)
    if detector.empty():
        print(f"[错误] Haar 级联文件加载失败: {load_path}")
        print("  请尝试重新安装 OpenCV: pip install --force-reinstall opencv-python")
        sys.exit(1)

    print(f"  [人脸检测器] 加载成功（最严格模式）")
    print(f"    scaleFactor={FACE_DETECT_SCALE_FACTOR}  "
          f"（步长更小，检测更细致）")
    print(f"    minNeighbors={FACE_DETECT_MIN_NEIGHBORS}  "
          f"（更敏感，只需 {FACE_DETECT_MIN_NEIGHBORS} 个相邻矩形）")
    print(f"    minSize={FACE_DETECT_MIN_SIZE}  "
          f"（最小检测尺寸更小，能发现远处人脸）")
    return detector


def _has_face(img_path: Path, detector: cv2.CascadeClassifier) -> bool:
    """
    判断图像中是否包含人脸（最严格检测）。

    使用更严格的参数（scaleFactor=1.05, minNeighbors=2, minSize=(20,20)），
    宁可误判非人脸为人脸（假正例），也不放过真正含人脸的图像。

    对于"有人体但脸不明显"的情况：
        - 远景小人（< 20px）：不触发，保留
        - 侧脸 / 背影：Haar 只检测正面，不触发，保留
        - 遮挡严重：通常不触发，保留
        - 清晰正面人脸：必定触发，丢弃

    参数：
        img_path : 图像文件路径
        detector : 已加载的 CascadeClassifier

    返回：
        True  → 检测到人脸，丢弃
        False → 无人脸，保留
    """
    try:
        raw = img_path.read_bytes()
        img = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )
    except Exception as e:
        print(f"  [警告] 读取失败，跳过: {img_path.name} — {e}")
        return True  # 保守处理：读取失败一律丢弃

    if img is None:
        print(f"  [警告] 解码失败，跳过: {img_path.name}")
        return True

    # 图像过小（宽或高 < 50px）→ 无意义，直接丢弃
    if img.shape[0] < 50 or img.shape[1] < 50:
        return True

    # 直方图均衡化：增强低对比度图像中人脸的可见性，提高检测召回率
    img = cv2.equalizeHist(img)

    faces = detector.detectMultiScale(
        img,
        scaleFactor=FACE_DETECT_SCALE_FACTOR,
        minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
        minSize=FACE_DETECT_MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    return len(faces) > 0


def _collect_intel_images(intel_dir: Path) -> list:
    """
    从 Intel 数据集的 seg_train / seg_test 分割中，
    收集 forest / glacier / mountain / sea / buildings 五类图像路径。

    Intel 数据集目录结构：
        intel/
        ├── seg_train/
        │   ├── forest/
        │   │   ├── 200.jpg
        │   │   └── ...
        │   ├── glacier/
        │   ├── mountain/
        │   ├── sea/
        │   ├── buildings/
        │   └── street/    ← 跳过
        └── seg_test/
            ├── forest/
            ├── glacier/
            ├── mountain/
            ├── sea/
            └── buildings/

    返回：
        list of Path，所有找到的图像路径（随机打乱后返回）
    """
    valid_exts = {".jpg", ".jpeg", ".png"}
    all_images = []
    category_counts = {}

    print(f"  [扫描] Intel 数据集根目录: {intel_dir}")
    print(f"  [扫描] 读取分割: {INTEL_SPLITS}")
    print(f"  [扫描] 使用类别: {INTEL_CATEGORIES}")

    for split in INTEL_SPLITS:
        split_dir = intel_dir / split
        if not split_dir.exists():
            print(f"  [警告] 分割目录不存在，跳过: {split_dir}")
            continue

        for category in INTEL_CATEGORIES:
            cat_dir = split_dir / category
            if not cat_dir.exists():
                # seg_test 中部分类别可能不存在，静默跳过
                continue

            imgs = [
                f for f in cat_dir.iterdir()
                if f.is_file() and f.suffix.lower() in valid_exts
            ]
            all_images.extend(imgs)

            key = f"{split}/{category}"
            category_counts[key] = len(imgs)
            print(f"    {key:30s}: {len(imgs):5d} 张")

    # 按类别打印汇总
    print(f"\n  [扫描] 各类别汇总（seg_train + seg_test）：")
    for cat in INTEL_CATEGORIES:
        total = sum(
            v for k, v in category_counts.items()
            if k.endswith(f"/{cat}")
        )
        print(f"    {cat:12s}: {total:5d} 张")

    print(f"\n  [扫描] 合计: {len(all_images)} 张图像")

    if not all_images:
        print(f"  [错误] 未找到任何图像，请检查目录结构：")
        print(f"    {intel_dir}/seg_train/forest/xxx.jpg")
        print(f"    {intel_dir}/seg_test/glacier/xxx.jpg")
        return []

    # 随机打乱，保证训练/验证集来自不同类别
    random.seed(SEED)
    random.shuffle(all_images)
    print(f"  [扫描] 随机打乱完成（seed={SEED}）")

    return all_images


# ============================================================
# 主流程
# ============================================================

def prepare_negatives_intel(
    intel_dir: Path = None,
    train_count: int = TRAIN_NEG_COUNT,
    val_count:   int = VAL_NEG_COUNT,
):
    """
    从 Intel Image Classification 数据集筛选非人脸图像，
    追加复制到训练集和验证集目录（续接已有编号，不覆盖旧文件）。

    已达目标数量 → 直接跳过，不重复处理。
    数量不足时打印警告，不报错退出。

    参数：
        intel_dir   : Intel 数据集根目录（None 时使用默认路径）
        train_count : 训练集负样本总目标数量（含已有，默认 9000）
        val_count   : 验证集负样本总目标数量（含已有，默认 1000）
    """
    total_target = train_count + val_count

    # ---- 检查当前已有数量 ----
    train_existing = _count_files(TRAIN_NEG_DIR)
    val_existing   = _count_files(VAL_NEG_DIR)

    print(f"[Intel负样本] 当前已有: 训练 {train_existing} 张, 验证 {val_existing} 张")
    print(f"[Intel负样本] 目标总量: 训练 {train_count} 张, 验证 {val_count} 张")

    if train_existing >= train_count and val_existing >= val_count:
        print("[Intel负样本] 数量已充足，跳过处理")
        return

    train_needed = max(0, train_count - train_existing)
    val_needed   = max(0, val_count   - val_existing)
    print(f"[Intel负样本] 还需补充: 训练 {train_needed} 张, 验证 {val_needed} 张")

    # ---- 确定 Intel 数据集路径 ----
    images_root = intel_dir or INTEL_DIR

    if not images_root.exists():
        print(f"\n[Intel负样本] [错误] Intel 数据集目录不存在: {images_root}")
        print()
        print("  请从 Kaggle 下载 Intel Image Classification 数据集（约 350 MB）：")
        print("    https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
        print()
        print(f"  下载解压后，确保目录结构为：")
        print(f"    {images_root}/seg_train/forest/xxx.jpg")
        print(f"    {images_root}/seg_test/glacier/xxx.jpg")
        print(f"    ...")
        sys.exit(1)

    print(f"\n[Intel负样本] 开始准备 Intel 负样本大图")
    print(f"  Intel 数据集目录: {images_root}")

    # ---- 加载人脸检测器（最严格模式）----
    print("\n[Intel负样本] 加载 OpenCV 人脸检测器（最严格模式）...")
    detector = _build_face_detector()

    # ---- 收集图像路径 ----
    print("\n[Intel负样本] 扫描 Intel 图像文件...")
    all_images = _collect_intel_images(images_root)

    if not all_images:
        print("[Intel负样本] [错误] 未找到任何图像，请检查目录结构后重试。")
        sys.exit(1)

    # ---- 创建输出目录 ----
    TRAIN_NEG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_NEG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 获取续接起始序号（不覆盖旧文件）----
    train_next_idx = _get_max_index(TRAIN_NEG_DIR) + 1
    val_next_idx   = _get_max_index(VAL_NEG_DIR)   + 1
    print(f"\n[Intel负样本] 文件续接起始序号: 训练集 neg_{train_next_idx:05d}, "
          f"验证集 neg_{val_next_idx:05d}")

    # ---- 当前计数（用于判断是否已满）----
    train_cnt = train_existing
    val_cnt   = val_existing

    # ---- 本次新增计数 ----
    train_added = 0
    val_added   = 0

    skipped_face   = 0
    skipped_exists = 0

    print(f"\n[Intel负样本] 开始过滤与复制")
    print(f"  过滤策略: 最严格 Haar 检测（scaleFactor=1.05, minNeighbors=2, minSize=20px）")
    print(f"  直方图均衡化: 开启（增强低对比度图像的人脸可见性）")
    print(f"  文件命名: 续接现有最大序号，避免覆盖旧文件\n")

    for img_path in tqdm(all_images, desc="  [Intel负样本] 处理进度", unit="张"):

        # 已达目标，提前结束
        if train_cnt >= train_count and val_cnt >= val_count:
            tqdm.write(f"\n  [完成] 已达目标数量，提前结束遍历")
            break

        # 最严格人脸检测过滤
        if _has_face(img_path, detector):
            skipped_face += 1
            continue

        # 决定放入训练集还是验证集（优先填满训练集）
        if train_cnt < train_count:
            dest_dir    = TRAIN_NEG_DIR
            file_idx    = train_next_idx
            train_next_idx += 1
            train_cnt   += 1
            train_added += 1
        elif val_cnt < val_count:
            dest_dir   = VAL_NEG_DIR
            file_idx   = val_next_idx
            val_next_idx += 1
            val_cnt   += 1
            val_added += 1
        else:
            break

        dest_path = dest_dir / f"neg_{file_idx:05d}.jpg"

        # 防御：目标文件意外存在（理论上不会，因为从最大序号+1开始）
        if dest_path.exists():
            skipped_exists += 1
            continue

        # 复制原图（保留原始分辨率、色彩，不做任何处理）
        try:
            shutil.copy2(str(img_path), str(dest_path))
        except Exception as e:
            tqdm.write(f"\n  [警告] 复制失败: {img_path.name} → {dest_path.name}: {e}")
            # 回退计数
            if dest_dir == TRAIN_NEG_DIR:
                train_cnt   -= 1
                train_added -= 1
                train_next_idx -= 1
            else:
                val_cnt   -= 1
                val_added -= 1
                val_next_idx -= 1
            continue

        # 每 500 张打印一次进度
        total_added = train_added + val_added
        if total_added % 500 == 0 and total_added > 0:
            tqdm.write(
                f"  [进度] 本次新增 {total_added} 张 "
                f"（训练: {train_cnt}/{train_count}, "
                f"验证: {val_cnt}/{val_count}, "
                f"跳过含人脸: {skipped_face}）"
            )

    # ---- 最终统计 ----
    print(f"\n[Intel负样本] 处理完成！")
    print(f"  本次新增: 训练集 {train_added} 张, 验证集 {val_added} 张")
    print(f"  当前合计: 训练集 {train_cnt:6d} 张 → {TRAIN_NEG_DIR}")
    print(f"            验证集 {val_cnt:6d} 张 → {VAL_NEG_DIR}")
    print(f"  人脸被过滤: {skipped_face:6d} 张（过滤率: "
          f"{skipped_face / max(1, skipped_face + train_added + val_added) * 100:.1f}%）")
    if skipped_exists:
        print(f"  文件已存在跳过: {skipped_exists} 张")

    # ---- 数量不足提示 ----
    if train_cnt < train_count:
        shortage = train_count - train_cnt
        print(f"\n  [警告] 训练集负样本仍不足目标（差 {shortage} 张）")
        print("  Intel 数据集图像已用完，可通过以下方式补充：")
        print("    - LHQ 自然风景数据集 (Kaggle: hshukla/landscapes)")
        print("    - 手动复制图像到目录，文件命名续接当前最大序号")
        print(f"      {TRAIN_NEG_DIR}")

    if val_cnt < val_count:
        shortage = val_count - val_cnt
        print(f"\n  [警告] 验证集负样本仍不足目标（差 {shortage} 张）")
        print(f"      {VAL_NEG_DIR}")

    # ---- 样本核查 ----
    _print_sample_info()


def _print_sample_info():
    """打印训练集和验证集各前 3 张图的基本信息，供人工核查。"""
    print(f"\n[Intel负样本] 样本核查（训练集最新 3 张）：")
    files = sorted(TRAIN_NEG_DIR.glob("*.jpg"))[-3:]
    for f in files:
        try:
            img = cv2.imdecode(
                np.frombuffer(f.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is not None:
                h, w = img.shape[:2]
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name}: {w}×{h} px, {size_kb:.1f} KB")
        except Exception:
            pass

    print(f"\n[Intel负样本] 样本核查（验证集最新 3 张）：")
    files = sorted(VAL_NEG_DIR.glob("*.jpg"))[-3:]
    for f in files:
        try:
            img = cv2.imdecode(
                np.frombuffer(f.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is not None:
                h, w = img.shape[:2]
                size_kb = f.stat().st_size / 1024
                print(f"  {f.name}: {w}×{h} px, {size_kb:.1f} KB")
        except Exception:
            pass


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Viola-Jones 负样本准备脚本（Intel Image Classification → train/negative & val/negative）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认路径（Intel 数据集放在 data/downloads/intel/ 下）
  python train/prepare_negatives_intel.py

  # 手动指定 Intel 数据集根目录
  python train/prepare_negatives_intel.py --intel-dir /path/to/intel

  # 自定义总目标数量
  python train/prepare_negatives_intel.py --train-count 9000 --val-count 1000

Intel Image Classification 下载地址（需登录 Kaggle，约 350 MB）：
  https://www.kaggle.com/datasets/puneet6060/intel-image-classification

解压后目录结构须为：
  data/downloads/intel/seg_train/forest/xxx.jpg
  data/downloads/intel/seg_train/glacier/xxx.jpg
  data/downloads/intel/seg_test/mountain/xxx.jpg
  ...（不含 street 类别，该类别会被自动跳过）

文件续接说明：
  本脚本自动检测 train/negative/ 和 val/negative/ 中已有的最大序号，
  从该序号 +1 开始命名新文件，不会覆盖 prepare_negatives.py 的输出。

人脸检测说明（最严格模式）：
  scaleFactor=1.05, minNeighbors=2, minSize=(20,20) + 直方图均衡化
  远景路人、侧脸、背影通常不会触发检测，正面清晰人脸必定被过滤。

输出目录（续接已有文件）：
  data/train/negative/neg_XXXXX.jpg  （续接已有编号）
  data/val/negative/neg_XXXXX.jpg    （续接已有编号）
        """
    )
    parser.add_argument(
        "--intel-dir", type=str, default=None,
        help="Intel 数据集根目录（默认: data/downloads/intel/）"
    )
    parser.add_argument(
        "--train-count", type=int, default=TRAIN_NEG_COUNT,
        help=f"训练集负样本总目标（含已有，默认 {TRAIN_NEG_COUNT}）"
    )
    parser.add_argument(
        "--val-count", type=int, default=VAL_NEG_COUNT,
        help=f"验证集负样本总目标（含已有，默认 {VAL_NEG_COUNT}）"
    )
    args = parser.parse_args()

    intel_dir = Path(args.intel_dir) if args.intel_dir else None

    print("=" * 65)
    print("  Viola-Jones 负样本大图准备（Intel Image Classification）")
    print(f"  数据根目录  : {DATA_DIR.resolve()}")
    print(f"  Intel 目录  : {intel_dir or INTEL_DIR}")
    print(f"  使用类别    : {', '.join(INTEL_CATEGORIES)}")
    print(f"  总目标数量  : 训练 {args.train_count} + 验证 {args.val_count} 张")
    print(f"  检测模式    : 最严格（scaleFactor=1.05, minNeighbors=2, minSize=20px）")
    print("=" * 65)

    prepare_negatives_intel(
        intel_dir=intel_dir,
        train_count=args.train_count,
        val_count=args.val_count,
    )

    print("\n" + "=" * 65)
    print("  Intel 负样本准备完成，目录统计：")
    train_cnt = _count_files(TRAIN_NEG_DIR)
    val_cnt   = _count_files(VAL_NEG_DIR)
    t_ok = "✓" if train_cnt >= args.train_count else f"✗ 不足（需 {args.train_count}）"
    v_ok = "✓" if val_cnt   >= args.val_count   else f"✗ 不足（需 {args.val_count}）"
    print(f"  训练集负样本: {train_cnt:6d} 张  [{t_ok}]  → {TRAIN_NEG_DIR}")
    print(f"  验证集负样本: {val_cnt:6d} 张  [{v_ok}]  → {VAL_NEG_DIR}")
    print("=" * 65)