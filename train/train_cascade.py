"""
train_cascade.py
================
级联分类器训练主入口脚本（支持自适应特征枚举）
"""

import os
import argparse
import pickle
import numpy as np

from train.prepare_positives import load_positive_data   # 返回已处理好的正样本数组列表
from train.integral_image import build_batch
from train.haar_features import enumerate_features_adaptive, compute_all_features
from train.cascade_trainer import (
    CascadeTrainer,
    _collect_patches_from_dir,
    _variance_normalize_patch,
)


# ─────────────────────────────────────────────────────────────
#  命令行参数
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Viola-Jones 级联分类器训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 数据规模（测试版默认值；生产训练可调整为 max_pos=4000, max_neg=10000, target_fpr=1e-5）
    parser.add_argument("--max_pos",          type=int,   default=1200,
                        help="训练正样本数量上限")
    parser.add_argument("--max_neg",          type=int,   default=3000,
                        help="第一层训练负样本数量")
    parser.add_argument("--max_val_pos",      type=int,   default=400,
                        help="验证集正样本数量上限")
    parser.add_argument("--val_neg_per_round",type=int,   default=800,
                        help="固定验证负样本采集数量")

    # 级联超参数
    parser.add_argument("--target_fpr",            type=float, default=0.01,
                        help="目标整体 FPR（测试版设大一些方便快速收敛）")
    parser.add_argument("--layer_max_fpr",          type=float, default=0.60,
                        help="单层最大 FPR")
    parser.add_argument("--layer_min_dr",           type=float, default=0.99,
                        help="单层最低 DR")
    parser.add_argument("--max_features_per_stage", type=int,   default=400,
                        help="单层特征数上限")
    parser.add_argument("--hnm_step",               type=int,   default=4,
                        help="HNM 滑动步长")

    # 路径配置
    parser.add_argument("--pos_dir",        type=str, default="data/train/positive",
                        help="训练正样本目录")
    parser.add_argument("--train_neg_dir",  type=str, default="data/train/negative",
                        help="训练负样本大图目录")
    parser.add_argument("--val_pos_dir",    type=str, default="data/val/positive",
                        help="验证正样本目录")
    parser.add_argument("--val_neg_dir",    type=str, default="data/val/negative",
                        help="验证负样本大图目录")
    parser.add_argument("--model_out",      type=str, default="models/cascade_model.pkl",
                        help="级联分类器模型输出路径")
    parser.add_argument("--features_cache", type=str, default="models/features_cache.npz",
                        help="特征缓存路径")
    parser.add_argument("--checkpoint",     type=str, default="models/cascade_checkpoint.pkl",
                        help="训练断点保存路径")

    # 控制开关
    parser.add_argument("--force_recompute", action="store_true",
                        help="忽略缓存，强制重新计算特征矩阵")
    parser.add_argument("--restart",         action="store_true",
                        help="忽略断点，从头训练")
    parser.add_argument("--interactive",     action="store_true",
                        help="开启交互模式（FPR不达标时手动选择策略）")

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
#  主流程
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # ── Step 1：枚举 Haar 特征描述符 ─────────────────────────
    print("\n[Pipeline] Step 1. 动态自适应枚举 Haar 特征描述符...")

    # 传入加载的训练集正样本数量 (args.max_pos)，算法自动决定步长参数
    features_desc = enumerate_features_adaptive(n_pos_samples=args.max_pos, win_size=24)
    print(f"  -> 共 {len(features_desc)} 个 Haar 特征描述符。")

    # ── Step 2：准备特征矩阵 ──────────────────────────────────
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

        # 向后兼容：旧版缓存可能不含验证负样本
        if 'X_val_neg' in cache and 'y_val_neg' in cache:
            X_val_neg = cache['X_val_neg'].astype(np.float32)
            y_val_neg = cache['y_val_neg'].astype(np.int32)
        else:
            print("  -> 旧版缓存不含验证负样本特征，需重新计算（请加 --force_recompute）")
            cache.close()
            cache_ok = False

        if cache_ok:
            del cache
            print(f"  -> 训练正: {len(X_pos_tr)}，训练负(第一层): {len(X_neg_tr)}，"
                  f"验证正: {len(X_val_pos)}，验证负: {len(X_val_neg)}，特征维度: {X_pos_tr.shape[1]}")

    if not cache_ok:
        print(f"\n[Pipeline] Step 2. 未找到完整缓存，重新计算特征矩阵...")

        # ── 2a：读取训练正样本 ───────────────────────────────
        print(f"\n  [2a] 读取训练正样本（max={args.max_pos}）...")
        print(f"       来源: {args.pos_dir}")
        imgs_pos_tr = load_positive_data("train", max_count=args.max_pos)

        # 将正样本统一归一化
        normalized_pos_tr = []
        for img in imgs_pos_tr:
            normed = _variance_normalize_patch(img)
            if normed is not None:
                normalized_pos_tr.append(normed)
        imgs_pos_tr = np.array(normalized_pos_tr, dtype=np.float32)

        print(f"  -> 训练正样本: {len(imgs_pos_tr)} 张（方差归一化 float32 数组）")

        # ── 2b：从训练大图随机裁取第一层负样本 ──────────────
        print(f"\n  [2b] 从训练大图随机裁取第一层负样本（目标 {args.max_neg} 个）...")
        print(f"       来源: {args.train_neg_dir}")
        imgs_neg_tr = _collect_patches_from_dir(
            image_dir=args.train_neg_dir,
            n_samples=args.max_neg,
            win_size=24,
            mode='random',
            one_per_image=False,
        )
        imgs_neg_tr = np.array(imgs_neg_tr, dtype=np.float32)
        print(f"  -> 第一层训练负样本: {len(imgs_neg_tr)} 个归一化 patch")

        # ── 2c：读取验证正样本 ───────────────────────────────
        print(f"\n  [2c] 读取验证正样本（max={args.max_val_pos}）...")
        print(f"       来源: {args.val_pos_dir}")
        imgs_pos_val = load_positive_data("val", max_count=args.max_val_pos)

        # 将验证正样本统一归一化
        normalized_pos_val = []
        for img in imgs_pos_val:
            normed = _variance_normalize_patch(img)
            if normed is not None:
                normalized_pos_val.append(normed)
        imgs_pos_val = np.array(normalized_pos_val, dtype=np.float32)

        print(f"  -> 验证正样本: {len(imgs_pos_val)} 张")

        # ── 2d：从验证大图随机裁取验证负样本 ──────────────────
        print(f"\n  [2d] 从验证大图随机裁取验证负样本（目标 {args.val_neg_per_round} 个）...")
        print(f"       来源: {args.val_neg_dir}")
        imgs_neg_val = _collect_patches_from_dir(
            image_dir=args.val_neg_dir,
            n_samples=args.val_neg_per_round,
            win_size=24,
            mode='random',
            one_per_image=True,  # 均匀覆盖验证大图
        )
        imgs_neg_val = np.array(imgs_neg_val, dtype=np.float32)
        print(f"  -> 验证负样本: {len(imgs_neg_val)} 个归一化 patch")


        # ── 2e：批量构建积分图 ────────────────────────────────
        print(f"\n  [2e] 批量构建积分图...")
        print(f"    训练正样本（{len(imgs_pos_tr)} 张）...")
        iimgs_pos_tr  = build_batch(imgs_pos_tr)
        print(f"    训练负样本（{len(imgs_neg_tr)} 张）...")
        iimgs_neg_tr  = build_batch(imgs_neg_tr)
        print(f"    验证正样本（{len(imgs_pos_val)} 张）...")
        iimgs_pos_val = build_batch(imgs_pos_val)
        print(f"    验证负样本（{len(imgs_neg_val)} 张）...")
        iimgs_neg_val = build_batch(imgs_neg_val)
        print(f"  -> 积分图构建完成。")

        # ── 2f：批量计算 Haar 特征矩阵 ────────────────────────
        print(f"\n  [2f] 计算 Haar 特征矩阵（{len(features_desc)} 个特征）...")
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

        print(f"    验证负样本特征...")
        X_val_neg = compute_all_features(
            iimgs_neg_val, features_desc, scale=1.0
        ).astype(np.float32)
        y_val_neg = np.zeros(len(X_val_neg), dtype=np.int32)

        print(f"  -> 训练正: {X_pos_tr.shape}，训练负: {X_neg_tr.shape}，"
              f"验证正: {X_val_pos.shape}，验证负: {X_val_neg.shape}")

        # ── 2g：保存缓存 ──────────────────────────────────────
        print(f"\n  [2g] 保存特征缓存至: {args.features_cache}")
        np.savez_compressed(
            args.features_cache,
            X_pos_tr=X_pos_tr,   y_pos_tr=y_pos_tr,
            X_neg_tr=X_neg_tr,   y_neg_tr=y_neg_tr,
            X_val_pos=X_val_pos, y_val_pos=y_val_pos,
            X_val_neg=X_val_neg, y_val_neg=y_val_neg,
        )
        print(f"  -> 缓存保存完成（已含固定的验证负样本特征）")

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
    print(f"  验证负样本数    : {len(X_val_neg)} 个（全程固定）")
    print(f"  HNM 步长        : {args.hnm_step}")
    print(f"  训练负样本大图  : {args.train_neg_dir}")

    trainer = CascadeTrainer(
        X_pos_train         = X_pos_tr,
        y_pos_train         = y_pos_tr,
        X_neg_train         = X_neg_tr,          # 第一层负样本
        y_neg_train         = y_neg_tr,
        X_val_pos           = X_val_pos,         # 验证正样本
        y_val_pos           = y_val_pos,
        X_val_neg           = X_val_neg,         # 验证负样本
        y_val_neg           = y_val_neg,
        train_neg_image_dir = args.train_neg_dir,# 训练集大图目录（HNM 扫描用）
        features_desc       = features_desc,
        target_fpr          = args.target_fpr,
        layer_max_fpr       = args.layer_max_fpr,
        layer_min_dr        = args.layer_min_dr,
        max_features_per_stage = args.max_features_per_stage,
        hnm_step            = args.hnm_step,
        checkpoint_path     = args.checkpoint,
        resume_state        = resume_state,
        non_interactive     = not args.interactive,  # 默认非交互；--interactive 开启手动选择
    )

    # ── Step 5：执行训练 ──────────────────────────────────────
    print(f"\n[Pipeline] Step 5. 开始级联训练...")
    stages = trainer.train()
    print(f"\n[Pipeline] 训练结束，共 {len(stages)} 层。")

    # ── Step 6：模型序列化 ────────────────────────────────────
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
