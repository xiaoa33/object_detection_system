"""
train_cascade.py
================
级联分类器训练主入口脚本

数据处理流程：

  训练集正样本：
    从 pos_dir 读取已处理好的 24×24 人脸 patch（由 data_loader 完成裁剪/灰度/归一化）
    → 直接用 build_batch + compute_all_features 计算特征矩阵
    → 全程不变，写入缓存

  训练集负样本（第一层）：
    从 train_neg_dir 下的彩色大图随机裁取 max_neg 个 24×24 子窗口
    → 整张转灰度 → 方差归一化 → 计算特征矩阵
    → 写入缓存（第一层固定，不会变）
    后续层的负样本由 cascade_trainer 内部的 HNM 动态生成，不经过此处

  验证集正样本：
    从 val_pos_dir 读取，处理方式同训练正样本
    → 全程不变，写入缓存

  验证集负样本：
    不在此处处理，不写入缓存。
    每轮训练时由 cascade_trainer 从 val_neg_dir 大图随机采样 val_neg_per_round 个。
    每轮独立随机，避免固定采样偏差。

  缓存内容：训练正特征矩阵 + 第一层训练负特征矩阵 + 验证正特征矩阵
            （验证负不缓存，每轮重采样）

  正样本说明（改动）：
    旧版 _load_positive_patches 在 train_cascade.py 中做了转灰度、裁剪、归一化。
    新版正样本由 data_loader.load_positive_samples() 返回已处理好的 numpy 数组，
    train_cascade.py 只负责构建积分图和计算特征，不再重复做图像预处理。
"""

import os
import argparse
import pickle
import numpy as np

from prepare_positives import load_positive_data   # 返回已处理好的正样本数组列表
from integral_image import build_batch
from haar_features import enumerate_features, compute_all_features
from cascade_trainer import (
    CascadeTrainer,
    _collect_patches_from_dir,
    _patches_to_feature_matrix,
)


# ─────────────────────────────────────────────────────────────
#  命令行参数
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Viola-Jones 级联分类器训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据规模
    parser.add_argument("--max_pos",          type=int,   default=4000,
                        help="训练正样本数量上限")
    parser.add_argument("--max_neg",          type=int,   default=10000,
                        help="第一层训练负样本数量（从大图随机裁取）")
    parser.add_argument("--max_val_pos",      type=int,   default=500,
                        help="验证集正样本数量上限")
    parser.add_argument("--val_neg_per_round",type=int,   default=1000,
                        help="每轮评估时从验证大图随机采样的负样本数（每轮重采样）")

    # 级联超参数
    parser.add_argument("--target_fpr",            type=float, default=1e-5,
                        help="目标整体 FPR")
    parser.add_argument("--layer_max_fpr",          type=float, default=0.50,
                        help="单层最大 FPR")
    parser.add_argument("--layer_min_dr",           type=float, default=0.99,
                        help="单层最低 DR")
    parser.add_argument("--max_features_per_stage", type=int,   default=200,
                        help="单层特征数上限")
    parser.add_argument("--hnm_step",               type=int,   default=4,
                        help="HNM 滑动步长（150×150 大图推荐 4）")

    # 路径
    parser.add_argument("--pos_dir",        type=str, default="../data/train/positive",
                        help="训练正样本目录（已处理好的24×24 patch）")
    parser.add_argument("--train_neg_dir",  type=str, default="../data/train/negative",
                        help="训练负样本大图目录（彩色图，第一层随机裁取及后续 HNM 共用）")
    parser.add_argument("--val_pos_dir",    type=str, default="../data/val/positive",
                        help="验证正样本目录")
    parser.add_argument("--val_neg_dir",    type=str, default="../data/val/negative",
                        help="验证负样本大图目录（彩色图，每轮重新随机采样）")
    parser.add_argument("--model_out",      type=str, default="../models/cascade_model.pkl")
    parser.add_argument("--features_cache", type=str, default="../models/features_cache.npz",
                        help="特征缓存路径（训练正负 + 验证正，不含验证负）")
    parser.add_argument("--checkpoint",     type=str, default="../models/cascade_checkpoint.pkl")

    # 控制开关
    parser.add_argument("--force_recompute", action="store_true",
                        help="忽略缓存，强制重新计算特征矩阵")
    parser.add_argument("--restart",         action="store_true",
                        help="忽略断点，从头训练")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # ── Step 1：枚举 Haar 特征描述符 ─────────────────────────
    print("\n[Pipeline] Step 1. 枚举 Haar-like 特征描述符（24×24 窗口）...")
    features_desc = enumerate_features(win_size=24)
    print(f"  -> 共 {len(features_desc)} 个 Haar 特征描述符。")

    # ── Step 2：准备特征矩阵 ──────────────────────────────────
    # 缓存内容：训练正特征矩阵 + 第一层训练负特征矩阵 + 验证正特征矩阵
    # 验证负样本不缓存（每轮 cascade_trainer 内部重新采样）
    cache_ok = os.path.exists(args.features_cache) and not args.force_recompute

    if cache_ok:
        print(f"\n[Pipeline] Step 2. 加载特征缓存: {args.features_cache}")
        cache     = np.load(args.features_cache)
        X_pos_tr  = cache['X_pos_tr'].astype(np.float32)
        y_pos_tr  = cache['y_pos_tr'].astype(np.int32)
        X_neg_tr  = cache['X_neg_tr'].astype(np.float32)
        y_neg_tr  = cache['y_neg_tr'].astype(np.int32)
        X_val_pos = cache['X_val_pos'].astype(np.float32)
        y_val_pos = cache['y_val_pos'].astype(np.int32)
        del cache
        print(f"  -> 训练正: {len(X_pos_tr)}，训练负(第一层): {len(X_neg_tr)}，"
              f"验证正: {len(X_val_pos)}，特征维度: {X_pos_tr.shape[1]}")

    else:
        print(f"\n[Pipeline] Step 2. 未找到缓存，重新计算特征矩阵...")

        # ── 2a：读取训练正样本 ───────────────────────────────
        # 正样本由 data_loader.load_positive_samples() 返回已处理好的数组列表
        # （data_loader 内部已完成：读取 → 转灰度 → 裁剪至24×24 → 方差归一化）
        # train_cascade.py 不再重复做图像预处理，只负责构建积分图和计算特征
        print(f"\n  [2a] 读取训练正样本（max={args.max_pos}）...")
        print(f"       来源: {args.pos_dir}")
        print(f"       预处理: 由 data_loader.load_positive_samples() 完成（转灰度/裁剪/归一化）")
        imgs_pos_tr = load_positive_data(
            "train"
        )
        print(f"  -> 训练正样本: {len(imgs_pos_tr)} 张（已处理好的 float32 数组）")

        # ── 2b：从训练大图随机裁取第一层负样本 ──────────────
        # 第一层负样本从彩色大图中随机裁取，不做级联过滤（此时还没有级联）
        # 处理流程（由 _collect_patches_from_dir 完成）：
        #   读彩色大图 → 整张转灰度（一次转换，效率最高） → 随机裁24×24 → 方差归一化
        # 这批负样本固定不变，可以缓存；后续层的负样本由 HNM 在训练过程中动态生成
        print(f"\n  [2b] 从训练大图随机裁取第一层负样本（目标 {args.max_neg} 个）...")
        print(f"       来源: {args.train_neg_dir}（彩色大图）")
        print(f"       处理: 读彩色大图 → 整张转灰度 → 随机裁24×24 → 方差归一化")
        imgs_neg_tr = _collect_patches_from_dir(
            image_dir=args.train_neg_dir,
            n_samples=args.max_neg,
            win_size=24,
            mode='random',      # 第一层随机采样，不做级联过滤
        )
        print(f"  -> 第一层训练负样本: {len(imgs_neg_tr)} 个归一化 patch")

        # ── 2c：读取验证正样本 ───────────────────────────────
        # 处理方式与训练正样本相同，由 data_loader 完成预处理
        print(f"\n  [2c] 读取验证正样本（max={args.max_val_pos}）...")
        print(f"       来源: {args.val_pos_dir}")
        imgs_pos_val = load_positive_data(
            "val"
        )
        print(f"  -> 验证正样本: {len(imgs_pos_val)} 张")

        # 验证负样本不在此处处理（注释说明原因）
        # 每轮训练时由 cascade_trainer 的 _sample_val_neg_features() 从大图重新随机采样
        print(f"\n  [2c 说明] 验证负样本不在此处处理。")
        print(f"            每轮训练时由 cascade_trainer 从 {args.val_neg_dir} 重新随机采样。")

        # ── 2d：批量构建积分图 ────────────────────────────────
        print(f"\n  [2d] 批量构建积分图...")
        print(f"    训练正样本（{len(imgs_pos_tr)} 张）...")
        iimgs_pos_tr  = build_batch(imgs_pos_tr)
        print(f"    训练负样本（{len(imgs_neg_tr)} 张）...")
        iimgs_neg_tr  = build_batch(imgs_neg_tr)
        print(f"    验证正样本（{len(imgs_pos_val)} 张）...")
        iimgs_pos_val = build_batch(imgs_pos_val)
        print(f"  -> 积分图构建完成。")

        # ── 2e：批量计算 Haar 特征矩阵 ────────────────────────
        print(f"\n  [2e] 计算 Haar 特征矩阵（{len(features_desc)} 个特征）...")
        print(f"    训练正样本特征...")
        X_pos_tr = compute_all_features(
            iimgs_pos_tr, features_desc, scale=1.0
        ).astype(np.float32)
        y_pos_tr = np.ones(len(X_pos_tr), dtype=np.int32)

        print(f"    训练负样本特征（第一层）...")
        X_neg_tr = compute_all_features(
            iimgs_neg_tr, features_desc, scale=1.0
        ).astype(np.float32)
        y_neg_tr = np.zeros(len(X_neg_tr), dtype=np.int32)

        print(f"    验证正样本特征...")
        X_val_pos = compute_all_features(
            iimgs_pos_val, features_desc, scale=1.0
        ).astype(np.float32)
        y_val_pos = np.ones(len(X_val_pos), dtype=np.int32)

        print(f"  -> 训练正: {X_pos_tr.shape}，训练负: {X_neg_tr.shape}，"
              f"验证正: {X_val_pos.shape}")

        # ── 2f：保存缓存 ──────────────────────────────────────
        # 缓存：训练正 + 第一层训练负 + 验证正（三者均固定不变）
        # 不缓存验证负（每轮重采样，无需缓存）
        print(f"\n  [2f] 保存特征缓存至: {args.features_cache}")
        np.savez_compressed(
            args.features_cache,
            X_pos_tr=X_pos_tr,   y_pos_tr=y_pos_tr,
            X_neg_tr=X_neg_tr,   y_neg_tr=y_neg_tr,
            X_val_pos=X_val_pos, y_val_pos=y_val_pos,
        )
        print(f"  -> 缓存保存完成（含：训练正/负(第一层)/验证正；不含验证负）")

    # ── Step 3：加载断点 ──────────────────────────────────────
    resume_state = None
    if os.path.exists(args.checkpoint) and not args.restart:
        print(f"\n[Pipeline] Step 3. 加载断点: {args.checkpoint}")
        with open(args.checkpoint, 'rb') as f:
            resume_state = pickle.load(f)
        print(f"  -> 已训练 {resume_state['layer_idx']} 层，"
              f"FPR={resume_state['overall_fpr']:.4e}")
    else:
        reason = "--restart 指定" if args.restart else "无断点文件"
        print(f"\n[Pipeline] Step 3. 从头训练（{reason}）。")

    # ── Step 4：初始化 CascadeTrainer ────────────────────────
    print(f"\n[Pipeline] Step 4. 初始化 CascadeTrainer...")
    print(f"  单层最大 FPR    : {args.layer_max_fpr}")
    print(f"  单层最低 DR     : {args.layer_min_dr}")
    print(f"  目标整体 FPR    : {args.target_fpr:.2e}")
    print(f"  单层特征上限    : {args.max_features_per_stage}")
    print(f"  每轮验证负样本  : {args.val_neg_per_round} 个（每轮重采样）")
    print(f"  HNM 步长        : {args.hnm_step}")
    print(f"  训练负样本大图  : {args.train_neg_dir}")
    print(f"  验证负样本大图  : {args.val_neg_dir}")

    trainer = CascadeTrainer(
        X_pos_train         = X_pos_tr,
        y_pos_train         = y_pos_tr,
        X_neg_train         = X_neg_tr,          # 第一层负样本（后续层由 HNM 生成）
        y_neg_train         = y_neg_tr,
        X_val_pos           = X_val_pos,         # 验证正样本特征矩阵（全程固定）
        y_val_pos           = y_val_pos,
        val_neg_image_dir   = args.val_neg_dir,  # 验证负样本大图目录（每轮重采样）
        train_neg_image_dir = args.train_neg_dir,# 训练负样本大图目录（HNM 扫描）
        features_desc       = features_desc,
        target_fpr          = args.target_fpr,
        layer_max_fpr       = args.layer_max_fpr,
        layer_min_dr        = args.layer_min_dr,
        max_features_per_stage = args.max_features_per_stage,
        val_neg_per_round   = args.val_neg_per_round,
        hnm_step            = args.hnm_step,
        checkpoint_path     = args.checkpoint,
        resume_state        = resume_state,
    )

    # ── Step 5：执行训练 ──────────────────────────────────────
    print(f"\n[Pipeline] Step 5. 开始级联训练...")
    stages = trainer.train()
    print(f"\n[Pipeline] 训练结束，共 {len(stages)} 层。")

    # ── Step 6：模型序列化 ────────────────────────────────────
    # 将特征索引（整数）转为描述符字典，方便检测模块无需 features_desc 独立加载
    print(f"\n[Pipeline] Step 6. 模型后处理：特征索引 → 描述符...")
    for stage in stages:
        for wc in stage.weak_classifiers:
            if isinstance(wc.feature_idx, (int, np.integer)):
                wc.feature_idx = features_desc[wc.feature_idx]

    with open(args.model_out, 'wb') as f:
        pickle.dump(stages, f)
    print(f"[Pipeline] ✓ 模型已保存: {args.model_out}")

    # ── 摘要 ─────────────────────────────────────────────────
    total_weak = sum(len(s.weak_classifiers) for s in stages)
    print(f"\n{'='*60}")
    print(f"[Summary] 训练完成")
    print(f"  总层数       : {len(stages)}")
    print(f"  各层特征数   : {[len(s.weak_classifiers) for s in stages]}")
    print(f"  总弱分类器数 : {total_weak}")
    print(f"  各层阈值     : {[round(s.threshold, 3) for s in stages]}")
    print(f"  模型路径     : {args.model_out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()