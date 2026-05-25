"""
prepare_positives.py
====================
正样本准备模块（Viola-Jones 算法复现）—— 基于 WIDER FACE 数据集

本文件位于 train/ 目录下，**仅负责正样本处理**，负样本和测试集由独立脚本处理。

数据来源：WIDER FACE 训练集
    - 官方主页: http://shuoyang1213.me/WIDERFACE/
    - 训练图像: WIDER_train.zip（约 1.5 GB）
    - 标注文件: wider_face_split.zip（含 wider_face_train_bbx_gt.txt）
    - Hugging Face 镜像（推荐，国内友好）:
        https://huggingface.co/datasets/CUHK-CSE/wider_face

数据保存目录（与 train/ 同级的 data/ 文件夹，格式与 data_loader.py 一致）：
    data/
    ├── downloads/
    │   ├── WIDER_train/          解压后的 WIDER FACE 训练图像
    │   │   └── images/
    │   │       ├── 0--Parade/
    │   │       ├── 1--Handshaking/
    │   │       └── ...（共 61 个事件类别子目录）
    │   └── wider_face_split/
    │       └── wider_face_train_bbx_gt.txt   标注文件
    ├── train/
    │   └── positive/   4000 张 24×24 灰度 PNG（方差归一化后保存）
    └── val/
        └── positive/    500 张 24×24 灰度 PNG

标注文件格式（wider_face_train_bbx_gt.txt）：
    # 每组由三部分组成：
    # 1. 图像相对路径（相对于 WIDER_train/images/）
    0--Parade/0_Parade_marchingband_1_849.jpg
    # 2. 该图中的人脸数量
    1
    # 3. 每行一个人脸：x y w h blur expression illumination occlusion pose invalid
    449 330 122 149 0 0 0 0 0 0
    #   各字段含义：
    #     x, y : 边界框左上角坐标
    #     w, h : 边界框宽度和高度
    #     blur       : 0=清晰, 1=普通模糊, 2=严重模糊
    #     expression : 0=正常, 1=夸张
    #     illumination: 0=正常, 1=夸张光照
    #     occlusion  : 0=无遮挡, 1=部分遮挡, 2=严重遮挡
    #     pose       : 0=典型正面(typical), 1=非典型(atypical)
    #     invalid    : 0=有效, 1=无效（分辨率过低等）

过滤条件（对应论文 Section 5.1 对正样本的要求）：
    - pose == 0       : 仅保留典型正面人脸（atypical 定义为偏转/俯仰 > 30° 或 yaw > 90°）
    - occlusion == 0  : 无遮挡
    - blur == 0       : 清晰图像
    - invalid == 0    : 有效标注
    - w >= MIN_FACE_PX 且 h >= MIN_FACE_PX : 人脸框足够大，resize 后不失真

bbox 外扩逻辑（对应论文 Section 5.1）：
    论文原文："bounding box enlarged by 50%"
    WIDER FACE 的 bbox 已紧密包含额头、下巴和脸颊，与论文裁剪规范高度一致。
    外扩 50% 后：以原 bbox 中心为圆心，新边长 = 原边长 × 1.5，
    超出图像边界的部分 clip 到 [0, 图像尺寸]。

关于方差归一化（论文 Section 5.4）：
    论文公式：σ² = m² - (1/N)Σx²（m 为均值），与标准写法 σ² = E[x²] - (E[x])² 等价。
    归一化操作：normed = (x - mean) / std
    本脚本在保存 PNG 前执行一次归一化（float32 → 线性映射 → uint8），
    训练时读取后直接以 float32 使用，不再重复归一化。
    检测时对每个子窗口独立执行相同操作。

使用方式：
    # 直接运行（准备正样本）
    python train/prepare_positives.py

    # 指定参数
    python train/prepare_positives.py --train-count 4000 --val-count 500

    # 如已手动下载 WIDER FACE，指定解压目录
    python train/prepare_positives.py \\
        --wider-images /path/to/WIDER_train/images \\
        --wider-anno   /path/to/wider_face_split/wider_face_train_bbx_gt.txt

    # 作为模块调用（供其他训练脚本使用）
    from prepare_positives import prepare_positives, load_positive_data

参考论文：Viola & Jones, "Robust Real-Time Face Detection", IJCV 2004
参考数据集：Yang et al., "WIDER FACE: A Face Detection Benchmark", CVPR 2016
"""

import os
import sys
import random
import zipfile
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from tqdm import tqdm


# ============================================================
# 路径配置（与 data_loader.py 保持一致）
# ============================================================

# train/prepare_positives.py 所在目录
TRAIN_DIR = Path(__file__).parent.resolve()

# data/ 目录与 train/ 同级
DATA_DIR = TRAIN_DIR.parent / "data"

# 正样本保存目录（与 data_loader.py 中 TRAIN_POS_DIR / VAL_POS_DIR 完全一致）
TRAIN_POS_DIR = DATA_DIR / "train" / "positive"
VAL_POS_DIR   = DATA_DIR / "val"   / "positive"

# 原始数据集缓存目录
DOWNLOAD_DIR  = DATA_DIR / "downloads"

# WIDER FACE 解压后的默认路径
WIDER_IMAGES_DIR = DOWNLOAD_DIR / "WIDER_train" / "images"
WIDER_ANNO_FILE  = DOWNLOAD_DIR / "wider_face_split" / "wider_face_train_bbx_gt.txt"


# ============================================================
# 数量与过滤配置
# ============================================================

TRAIN_POS_COUNT = 4000    # 训练集正样本目标数量（论文约 4916 张）
VAL_POS_COUNT   = 500     # 验证集正样本目标数量
OUTPUT_SIZE     = 24      # 目标图像尺寸（论文规定 24×24）
EXPAND_RATIO    = 1.5     # bbox 外扩比例（论文 "enlarged by 50%"）
MIN_FACE_PX     = 40      # 原始 bbox 最小边长（像素），过小的人脸 resize 后质量差
SEED            = 42      # 随机种子，保证可复现


# ============================================================
# 下载地址说明（WIDER FACE 官方）
# ============================================================
#
# WIDER_train.zip（约 1.5 GB）—— 请手动下载后放入 DOWNLOAD_DIR 并解压
#   Google Drive : https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view
#   腾讯微云     : https://share.weiyun.com/5WjCBWV
#   Hugging Face : https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_train.zip
#
# wider_face_split.zip（约 2 MB）—— 标注文件，可自动下载，也可手动下载
#   官方直链     : http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip
#
# 解压后目录结构须为：
#   data/downloads/WIDER_train/images/0--Parade/xxx.jpg   ← 训练图像
#   data/downloads/wider_face_split/wider_face_train_bbx_gt.txt  ← 标注文件

WIDER_ANNO_URL  = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
WIDER_ANNO_ZIP  = "wider_face_split.zip"
WIDER_TRAIN_ZIP = "WIDER_train.zip"   # 仅用于错误提示中显示文件名


# ============================================================
# 公共工具函数（与 data_loader.py 接口一致）
# ============================================================

def _reporthook(block_num: int, block_size: int, total_size: int):
    """urllib 下载进度回调，在同一行更新进度"""
    downloaded = block_num * block_size
    if total_size > 0:
        pct     = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / 1024 / 1024
        mb_tot  = total_size  / 1024 / 1024
        print(f"\r    进度: {mb_done:.1f} / {mb_tot:.1f} MB ({pct:.1f}%)",
              end="", flush=True)


def _download(url: str, save_path: Path, label: str) -> bool:
    """
    下载单个文件，已存在则跳过。
    返回 True 表示成功（或已存在），False 表示失败。
    """
    if save_path.exists() and save_path.stat().st_size > 0:
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"  [{label}] 文件已存在，跳过下载: {save_path.name} ({size_mb:.1f} MB)")
        return True

    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [{label}] 开始下载: {url}")
    try:
        urllib.request.urlretrieve(str(url), str(save_path), _reporthook)
        print()  # 换行
        size_mb = save_path.stat().st_size / 1024 / 1024
        print(f"  [{label}] 下载成功，大小: {size_mb:.1f} MB → {save_path}")
        return True
    except Exception as e:
        print(f"\n  [{label}] 下载失败: {e}")
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

    公式：σ² = E[x²] - (E[x])²，归一化：normed = (x - mean) / std
    若方差极小（纯色图像），返回全零数组以避免除零。

    参数：
        img : uint8 或 float32 数组，形状 (H, W)
    返回：
        float32 归一化数组，均值 ≈ 0，标准差 ≈ 1
    """
    img  = img.astype(np.float32)
    mean = img.mean()
    std  = img.std()
    if std < 1e-6:
        # 纯色图像，标准差为零，返回全零
        return np.zeros_like(img, dtype=np.float32)
    return (img - mean) / std


def _save_normalized_patch(normed: np.ndarray, save_path: Path):
    """
    将方差归一化后的 float32 图像保存为 uint8 PNG。

    归一化后值域约 [-3, +3]，线性映射到 [0, 255]（min→0，max→255）。
    Haar-like 特征基于差值计算，线性变换不影响判别能力。

    参数：
        normed    : float32 归一化数组，形状 (24, 24)
        save_path : PNG 保存路径（父目录须已存在）
    """
    lo, hi = normed.min(), normed.max()
    if hi - lo < 1e-6:
        img_u8 = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    else:
        img_u8 = ((normed - lo) / (hi - lo) * 255).astype(np.uint8)
    cv2.imencode('.png', img_u8)[1].tofile(str(save_path))


# ============================================================
# WIDER FACE 标注解析
# ============================================================

def _parse_wider_annotations(anno_file: Path) -> list:
    """
    解析 wider_face_train_bbx_gt.txt，返回所有标注记录列表。

    标注文件格式（每组三部分）：
        图像相对路径（如 0--Parade/0_Parade_xxx.jpg）
        人脸数量 N
        N 行，每行：x y w h blur expression illumination occlusion pose invalid

    返回：
        list of dict，每个 dict 对应一个人脸标注：
        {
            "img_rel_path": "0--Parade/0_Parade_xxx.jpg",  # 相对 WIDER_train/images/ 的路径
            "x": int, "y": int, "w": int, "h": int,       # 原始 bbox（左上角 + 宽高）
            "blur": int,        # 0=清晰, 1=普通, 2=严重
            "expression": int,  # 0=正常, 1=夸张
            "illumination": int,# 0=正常, 1=夸张
            "occlusion": int,   # 0=无, 1=部分, 2=严重
            "pose": int,        # 0=typical正面, 1=atypical非典型
            "invalid": int,     # 0=有效, 1=无效
        }
    """
    print(f"  [解析标注] 读取: {anno_file}")
    records = []
    skipped_zero = 0  # 人脸数=0 的图像（WIDER FACE 中少量存在）

    with open(str(anno_file), "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        # ---- 第 1 行：图像相对路径 ----
        img_rel_path = lines[i].strip()
        i += 1
        if i >= len(lines):
            break

        # ---- 第 2 行：人脸数量 ----
        try:
            num_faces = int(lines[i].strip())
        except ValueError:
            # 格式异常，跳过
            print(f"  [警告] 无法解析人脸数量，行内容：'{lines[i]}'，跳过")
            i += 1
            continue
        i += 1

        # WIDER FACE 中存在 num_faces=0 的条目，后面仍有一行占位符 "0 0 0 0 0 0 0 0 0 0"
        if num_faces == 0:
            skipped_zero += 1
            i += 1  # 跳过占位符行
            continue

        # ---- 第 3 ~ N+2 行：逐个人脸标注 ----
        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            i += 1

            # 正常标注应有 10 个字段
            if len(parts) < 10:
                print(f"  [警告] 标注字段不足（{len(parts)} < 10），行内容：'{' '.join(parts)}'，跳过")
                continue

            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                blur        = int(parts[4])
                expression  = int(parts[5])
                illumination= int(parts[6])
                occlusion   = int(parts[7])
                pose        = int(parts[8])
                invalid     = int(parts[9])
            except ValueError:
                print(f"  [警告] 字段转换失败，跳过：'{' '.join(parts)}'")
                continue

            records.append({
                "img_rel_path": img_rel_path,
                "x": x, "y": y, "w": w, "h": h,
                "blur": blur,
                "expression": expression,
                "illumination": illumination,
                "occlusion": occlusion,
                "pose": pose,
                "invalid": invalid,
            })

    print(f"  [解析标注] 共解析 {len(records)} 条人脸标注")
    if skipped_zero:
        print(f"  [解析标注] 跳过 {skipped_zero} 张 num_faces=0 的图像")
    return records


def _filter_records(records: list) -> list:
    """
    过滤标注记录，仅保留符合论文正样本要求的人脸。

    过滤条件：
        pose == 0       : 典型正面人脸（atypical 定义为偏转/俯仰 > 30° 或 yaw > 90°）
        occlusion == 0  : 无遮挡（partial/heavy 遮挡的脸 Haar 特征会受干扰）
        blur == 0       : 清晰（模糊图像影响特征学习）
        invalid == 0    : 有效标注
        w >= MIN_FACE_PX 且 h >= MIN_FACE_PX : 人脸框足够大，避免放大插值失真

    参数：
        records : _parse_wider_annotations 返回的全部标注列表
    返回：
        过滤后的标注列表
    """
    before = len(records)
    filtered = []
    cnt_pose = cnt_occ = cnt_blur = cnt_invalid = cnt_size = 0

    for r in records:
        if r["pose"] != 0:
            cnt_pose += 1
            continue
        if r["occlusion"] != 0:
            cnt_occ += 1
            continue
        if r["blur"] != 0:
            cnt_blur += 1
            continue
        if r["invalid"] != 0:
            cnt_invalid += 1
            continue
        if r["w"] < MIN_FACE_PX or r["h"] < MIN_FACE_PX:
            cnt_size += 1
            continue
        filtered.append(r)

    after = len(filtered)
    print(f"  [过滤] 过滤前: {before} 条 → 过滤后: {after} 条")
    print(f"    非正面人脸 (pose!=0):      {cnt_pose:6d} 条")
    print(f"    有遮挡 (occlusion!=0):     {cnt_occ:6d} 条")
    print(f"    模糊   (blur!=0):          {cnt_blur:6d} 条")
    print(f"    无效   (invalid!=0):       {cnt_invalid:6d} 条")
    print(f"    尺寸过小 (<{MIN_FACE_PX}px):     {cnt_size:6d} 条")
    return filtered


# ============================================================
# 正样本提取核心函数
# ============================================================

def _crop_face_patch(record: dict, images_root: Path) -> Optional[np.ndarray]:
    """
    根据一条 WIDER FACE 标注，裁剪并处理为 24×24 灰度正样本。

    处理步骤：
        1. 读取原图（BGR → 灰度）
        2. 计算外扩后的裁剪区域（bbox 扩大 EXPAND_RATIO 倍，clip 到图像边界）
        3. 裁剪人脸区域
        4. 缩放到 OUTPUT_SIZE × OUTPUT_SIZE（INTER_AREA，缩小时最佳）
        5. 方差归一化

    参数：
        record      : 单条标注 dict（来自 _filter_records）
        images_root : WIDER_train/images/ 目录路径

    返回：
        float32 ndarray，形状 (24, 24)，已方差归一化
        读取失败或裁剪区域为空时返回 None
    """
    img_path = images_root / record["img_rel_path"]

    # 读取原图
    try:
        raw = img_path.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [警告] 读取失败: {img_path.name} — {e}")
        return None

    if img is None:
        print(f"  [警告] 解码失败（可能不是有效图像）: {img_path.name}")
        return None

    img_h, img_w = img.shape[:2]
    x, y, w, h = record["x"], record["y"], record["w"], record["h"]

    # ---- bbox 外扩 50%（对应论文 "bounding box enlarged by 50%"）----
    # 以 bbox 中心为基准，向四周等比例扩大
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * EXPAND_RATIO
    new_h = h * EXPAND_RATIO

    x1 = int(round(cx - new_w / 2.0))
    y1 = int(round(cy - new_h / 2.0))
    x2 = int(round(cx + new_w / 2.0))
    y2 = int(round(cy + new_h / 2.0))

    # clip 到图像边界，避免越界
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # 检查裁剪区域有效性（外扩后可能边长为 0）
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    # 裁剪 + 转灰度
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 缩放到 24×24（INTER_AREA 插值，缩小时效果最好）
    resized = cv2.resize(gray, (OUTPUT_SIZE, OUTPUT_SIZE),
                         interpolation=cv2.INTER_AREA)

    # 方差归一化（论文 Section 5.4，此处是唯一一次归一化）
    normed = _variance_normalize(resized)
    return normed


# ============================================================
# 主流程
# ============================================================

def prepare_positives(
    wider_images_dir: Path = None,
    wider_anno_file:  Path = None,
    train_count: int = TRAIN_POS_COUNT,
    val_count:   int = VAL_POS_COUNT,
):
    """
    从 WIDER FACE 数据集准备正样本，保存为 24×24 灰度 PNG。

    文件已存在且数量达标 → 直接跳过，不重复处理。

    参数：
        wider_images_dir : WIDER_train/images/ 的路径（None 时使用默认 DOWNLOAD_DIR 下的路径）
        wider_anno_file  : wider_face_train_bbx_gt.txt 的路径（None 时使用默认路径）
        train_count      : 训练集正样本目标数量（默认 4000）
        val_count        : 验证集正样本目标数量（默认 500）

    输出目录结构（与 data_loader.py 完全一致）：
        data/train/positive/face_00000.png ~ face_03999.png
        data/val/positive/face_00000.png   ~ face_00499.png
    """
    # ---- 检查是否已完成 ----
    train_ok = _count_files(TRAIN_POS_DIR) >= train_count
    val_ok   = _count_files(VAL_POS_DIR)   >= val_count

    if train_ok and val_ok:
        print("[正样本] 文件已存在且数量充足，跳过处理")
        print(f"  训练集正样本: {_count_files(TRAIN_POS_DIR)} 张 → {TRAIN_POS_DIR}")
        print(f"  验证集正样本: {_count_files(VAL_POS_DIR)} 张 → {VAL_POS_DIR}")
        return

    total_needed = train_count + val_count
    print(f"[正样本] 开始准备 WIDER FACE 正样本（目标: 训练 {train_count} + 验证 {val_count} 张）")
    random.seed(SEED)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 确定路径 ----
    images_root = wider_images_dir or WIDER_IMAGES_DIR
    anno_file   = wider_anno_file  or WIDER_ANNO_FILE

    # ---- 检查/下载标注文件 ----
    if not anno_file.exists():
        print(f"\n[正样本] 标注文件不存在: {anno_file}")
        print("  尝试自动下载标注文件（较小，约 2 MB）...")
        anno_zip = DOWNLOAD_DIR / WIDER_ANNO_ZIP
        ok = _download(WIDER_ANNO_URL, anno_zip, "WIDER 标注")
        if ok:
            print(f"  解压标注文件 → {DOWNLOAD_DIR}")
            with zipfile.ZipFile(str(anno_zip), "r") as zf:
                zf.extractall(str(DOWNLOAD_DIR))
            print(f"  标注文件解压完成: {anno_file}")
        if not anno_file.exists():
            print("\n  [错误] 标注文件仍不存在，请手动下载后放置到以下路径：")
            print(f"    {anno_file}")
            print("  下载地址（标注 zip）：")
            print(f"    {WIDER_ANNO_URL}")
            print("  或从 Hugging Face 获取整个数据集：")
            print("    https://huggingface.co/datasets/CUHK-CSE/wider_face")
            sys.exit(1)

    # ---- 检查图像目录（仅支持手动下载，不自动下载图像）----
    if not images_root.exists() or not any(images_root.iterdir()):
        print(f"\n[正样本] [错误] 图像目录不存在或为空: {images_root}")
        print()
        print("  请手动下载 WIDER FACE 训练集（WIDER_train.zip，约 1.5 GB）：")
        print("    Google Drive : https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view")
        print("    腾讯微云     : https://share.weiyun.com/5WjCBWV")
        print("    Hugging Face : https://huggingface.co/datasets/wider_face/blob/main/data/WIDER_train.zip")
        print()
        print(f"  下载后将 WIDER_train.zip 解压到:")
        print(f"    {DOWNLOAD_DIR}/")
        print("  解压后目录结构须为：")
        print(f"    {WIDER_IMAGES_DIR}/0--Parade/xxx.jpg")
        print(f"    {WIDER_IMAGES_DIR}/1--Handshaking/xxx.jpg")
        print("    ...")
        print()
        print("  解压命令（Linux/Mac）:")
        print(f"    unzip {DOWNLOAD_DIR}/{WIDER_TRAIN_ZIP} -d {DOWNLOAD_DIR}/")
        sys.exit(1)

    # ---- 解析标注 ----
    print(f"\n[正样本] 解析标注文件: {anno_file.name}")
    all_records = _parse_wider_annotations(anno_file)

    # ---- 过滤 ----
    print(f"\n[正样本] 按过滤条件筛选（正面/无遮挡/清晰/有效/尺寸≥{MIN_FACE_PX}px）...")
    filtered = _filter_records(all_records)

    if len(filtered) < total_needed:
        print(f"\n  [警告] 过滤后可用标注 {len(filtered)} 条，少于目标 {total_needed} 条")
        print("  将放宽 MIN_FACE_PX 限制或调小 train_count/val_count")
        # 不直接退出，尽量生成，后续循环会提前结束

    # ---- 随机打乱，保证训练/验证集多样性 ----
    random.shuffle(filtered)
    print(f"\n[正样本] 过滤后可用: {len(filtered)} 条，开始裁剪处理...")

    # ---- 创建输出目录 ----
    TRAIN_POS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_POS_DIR.mkdir(parents=True, exist_ok=True)

    train_cnt = val_cnt = skip_cnt = 0

    print(f"  处理流程：读取原图 → 外扩{int((EXPAND_RATIO-1)*100)}%裁剪 → 转灰度 → "
          f"缩放{OUTPUT_SIZE}×{OUTPUT_SIZE} → 方差归一化 → 保存 PNG")
    print(f"  目标：训练集 {train_count} 张，验证集 {val_count} 张\n")

    for record in tqdm(filtered, desc="  [正样本] 处理进度", unit="张"):
        # 已达到目标数量时停止
        if train_cnt >= train_count and val_cnt >= val_count:
            break

        # 裁剪处理
        normed = _crop_face_patch(record, images_root)
        if normed is None:
            skip_cnt += 1
            continue

        # 分配到训练集或验证集（优先填满训练集）
        if train_cnt < train_count:
            save_path = TRAIN_POS_DIR / f"face_{train_cnt:05d}.png"
            train_cnt += 1
        elif val_cnt < val_count:
            save_path = VAL_POS_DIR / f"face_{val_cnt:05d}.png"
            val_cnt += 1
        else:
            break

        _save_normalized_patch(normed, save_path)

        # 每处理 500 张打印一次进度（tqdm 之外的补充信息）
        total_done = train_cnt + val_cnt
        if total_done % 500 == 0:
            tqdm.write(
                f"  [进度] 已完成 {total_done}/{total_needed} 张 "
                f"（训练: {train_cnt}, 验证: {val_cnt}, 跳过: {skip_cnt}）"
            )

    # ---- 最终汇报 ----
    print(f"\n[正样本] 处理完成！")
    print(f"  训练集正样本: {train_cnt:5d} 张 → {TRAIN_POS_DIR}")
    print(f"  验证集正样本: {val_cnt:5d} 张 → {VAL_POS_DIR}")
    if skip_cnt:
        print(f"  读取/裁剪失败，跳过: {skip_cnt} 张")
    if train_cnt < train_count:
        print(f"  [警告] 训练集正样本不足（目标 {train_count}，实际 {train_cnt}）")
        print(f"  建议放宽过滤条件（如允许 blur<=1）或补充其他数据集")
    if val_cnt < val_count:
        print(f"  [警告] 验证集正样本不足（目标 {val_count}，实际 {val_cnt}）")

    # ---- 打印几条样本信息供检查 ----
    _print_sample_info()


def _print_sample_info():
    """打印前几张训练集正样本的统计信息，供人工核查"""
    files = sorted(TRAIN_POS_DIR.glob("*.png"))[:5]
    if not files:
        return
    print(f"\n[正样本] 前 {len(files)} 张训练正样本统计（用于核查）：")
    for f in files:
        try:
            img = cv2.imdecode(
                np.frombuffer(f.read_bytes(), dtype=np.uint8),
                cv2.IMREAD_GRAYSCALE
            )
            if img is not None:
                arr = img.astype(np.float32)
                print(f"  {f.name}: shape={img.shape}, "
                      f"min={arr.min():.0f}, max={arr.max():.0f}, "
                      f"mean={arr.mean():.1f}, std={arr.std():.1f}")
        except Exception:
            pass


# ============================================================
# 数据读取接口（供训练脚本调用，与 data_loader.py 接口一致）
# ============================================================

def load_positive_data(split: str = "train", max_count: int = None) -> np.ndarray:
    """
    加载已准备好的正样本，返回 float32 numpy 数组。

    文件不足时自动触发 prepare_positives()。

    参数：
        split     : "train" 或 "val"
        max_count : 最多读取数量（None = 全部）

    返回：
        float32 ndarray，形状 (N, 24, 24)，值域 [0.0, 255.0]
        图像已在保存时做过方差归一化，读取后可直接用于特征计算
    """
    if split == "train":
        directory = TRAIN_POS_DIR
        target    = TRAIN_POS_COUNT
    elif split == "val":
        directory = VAL_POS_DIR
        target    = VAL_POS_COUNT
    else:
        raise ValueError(f"split 应为 'train' 或 'val'，收到: {split!r}")

    # 文件不足时自动准备
    if _count_files(directory) < target:
        print(f"[load_positive_data] {split} 正样本不完整，自动准备...")
        prepare_positives()

    # 读取所有 PNG
    files = sorted(directory.glob("*.png"))
    if max_count is not None:
        files = files[:max_count]

    if not files:
        print(f"  [警告] 正样本目录为空: {directory}")
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
        # 保存时已归一化，读取后直接转 float32，不再重复归一化
        images.append(img.astype(np.float32))

    arr = np.array(images, dtype=np.float32)
    print(f"  [load_positive_data] 加载 {split} 正样本: {len(arr)} 张，"
          f"shape={arr.shape} ← {directory.relative_to(DATA_DIR)}")
    return arr


# ============================================================
# 直接运行入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Viola-Jones 正样本准备脚本（WIDER FACE → 24×24 灰度 PNG）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认路径（WIDER FACE 数据集放在 data/downloads/ 下）
  python train/prepare_positives.py

  # 自定义训练/验证集数量
  python train/prepare_positives.py --train-count 4000 --val-count 500

  # 手动指定 WIDER FACE 路径（已手动解压时使用）
  python train/prepare_positives.py \\
      --wider-images /path/to/WIDER_train/images \\
      --wider-anno   /path/to/wider_face_split/wider_face_train_bbx_gt.txt

手动下载 WIDER FACE 数据集：
  官方主页: http://shuoyang1213.me/WIDERFACE/
  训练图像: WIDER_train.zip（约 1.5 GB）
  标注文件: wider_face_split.zip（约 2 MB）
  解压后目录结构：
    data/downloads/WIDER_train/images/0--Parade/xxx.jpg
    data/downloads/wider_face_split/wider_face_train_bbx_gt.txt

输出目录（与 data_loader.py 保持一致）：
  data/train/positive/face_00000.png ~ face_03999.png  （4000 张）
  data/val/positive/face_00000.png   ~ face_00499.png  （500 张）
        """
    )
    parser.add_argument(
        "--train-count", type=int, default=TRAIN_POS_COUNT,
        help=f"训练集正样本数量（默认 {TRAIN_POS_COUNT}）"
    )
    parser.add_argument(
        "--val-count", type=int, default=VAL_POS_COUNT,
        help=f"验证集正样本数量（默认 {VAL_POS_COUNT}）"
    )
    parser.add_argument(
        "--wider-images", type=str, default=None,
        help="WIDER_train/images/ 目录路径（默认使用 data/downloads/WIDER_train/images/）"
    )
    parser.add_argument(
        "--wider-anno", type=str, default=None,
        help="wider_face_train_bbx_gt.txt 路径（默认使用 data/downloads/wider_face_split/wider_face_train_bbx_gt.txt）"
    )
    args = parser.parse_args()

    # 将字符串路径转为 Path 对象
    wider_images = Path(args.wider_images) if args.wider_images else None
    wider_anno   = Path(args.wider_anno)   if args.wider_anno   else None

    print("=" * 60)
    print("  Viola-Jones 正样本准备（WIDER FACE）")
    print(f"  数据根目录: {DATA_DIR.resolve()}")
    print(f"  目标数量  : 训练 {args.train_count} + 验证 {args.val_count} 张")
    print("=" * 60)

    prepare_positives(
        wider_images_dir=wider_images,
        wider_anno_file =wider_anno,
        train_count     =args.train_count,
        val_count       =args.val_count,
    )

    print("\n" + "=" * 60)
    print("  正样本准备完成，目录统计：")
    train_cnt = _count_files(TRAIN_POS_DIR)
    val_cnt   = _count_files(VAL_POS_DIR)
    t_status  = "✓" if train_cnt >= args.train_count else f"✗ 不足（需 {args.train_count}）"
    v_status  = "✓" if val_cnt   >= args.val_count   else f"✗ 不足（需 {args.val_count}）"
    print(f"  训练集正样本: {train_cnt:6d} 张  [{t_status}]  → {TRAIN_POS_DIR}")
    print(f"  验证集正样本: {val_cnt:6d} 张  [{v_status}]  → {VAL_POS_DIR}")
    print("=" * 60)