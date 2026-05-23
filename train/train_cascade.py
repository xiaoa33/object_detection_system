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
    → 写入缓存（第一层固定不变）
    后续层的负样本由 cascade_trainer 内部的 HNM 动态生成，不经过此处

  验证集正样本：
    从 val_pos_dir 读取，处理方式同训练正样本
    → 全程不变，写入缓存

  验证集负样本：【改动】一次性从 val_neg_dir 下每张大图中随机裁取 1 个子窗口
    正好有 1000 张大图，每张取 1 个，共 1000 个，均匀覆盖所有大图
    → 整张转灰度 → 方差归一化 → 计算特征矩阵
    → 写入缓存，全程固定不变（与验证正样本一起构成稳定的评估标准）
    旧版：不缓存，每轮由 cascade_trainer 重新采样（导致评估标准不一致）
    新版：一次性准备好，写入缓存，全程使用同一批验证负样本

  缓存内容：
    训练正特征矩阵 + 第一层训练负特征矩阵 + 验证正特征矩阵 + 验证负特征矩阵
    （四者均固定不变，下次运行直接加载）

  关于负样本大图打乱顺序：
    - 第一层随机裁取：_collect_patches_from_dir 内部 np.random.shuffle(files)，
      每次运行随机打乱，不会固定从同一张图开始。
    - HNM 扫描（cascade_trainer 内部）：同样调用 _collect_patches_from_dir，
      每次 HNM 也重新打乱，每层扫描的起始图像都不同。
"""

import os
import argparse
import pickle
import numpy as np

from train.prepare_positives import load_positive_data   # 返回已处理好的正样本数组列表
from train.integral_image import build_batch
from train.haar_features import enumerate_features, compute_all_features
from train.cascade_trainer import (
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
    parser.add_argument("--max_pos",     type=int, default=4000,
                        help="训练正样本数量上限")
    parser.add_argument("--max_neg",     type=int, default=10000,
                        help="第一层训练负样本数量（从大图随机裁取）")
    parser.add_argument("--max_val_pos", type=int, default=500,
                        help="验证集正样本数量上限")
    # 【改动】val_neg_count 替代原来的 val_neg_per_round
    # 验证负样本固定为 1000 个（每张大图取1个），不再每轮重采样
    parser.add_argument("--val_neg_count", type=int, default=1000,
                        help="验证集负样本数量（每张大图取1个子窗口，共val_neg_count个，全程固定）")

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
    parser.add_argument("--pos_dir",       type=str, default="../data/train/positive",
                        help="训练正样本目录（已处理好的24×24 patch）")
    parser.add_argument("--train_neg_dir", type=str, default="../data/train/negative",
                        help="训练负样本大图目录（彩色图，第一层随机裁取及后续 HNM 共用）")
    parser.add_argument("--val_pos_dir",   type=str, default="../data/val/positive",
                        help="验证正样本目录")
    parser.add_argument("--val_neg_dir",   type=str, default="../data/val/negative",
                        help="验证负样本大图目录（彩色图，每张取1个子窗口，共1000个，全程固定）")
    parser.add_argument("--model_out",     type=str, default="../models/cascade_model.pkl")
    parser.add_argument("--features_cache",type=str, default="../models/features_cache.npz",
                        help="特征缓存路径（训练正/负 + 验证正/负，四者均固定）")
    parser.add_argument("--checkpoint",    type=str, default="../models/cascade_checkpoint.pkl")

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
    # 【改动】缓存内容增加验证负样本特征矩阵（X_val_neg / y_val_neg）
    # 旧版：验证负不缓存，每轮重采样（cascade_trainer 内部完成）
    # 新版：验证负也写入缓存，全程固定，与训练正/负/验证正一起存储
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
        # 【改动】加载验证负样本特征矩阵（旧版缓存中没有这两个键，需重新计算）
        if 'X_val_neg' in cache and 'y_val_neg' in cache:
            X_val_neg = cache['X_val_neg'].astype(np.float32)
            y_val_neg = cache['y_val_neg'].astype(np.int32)
            print(f"  -> 训练正: {len(X_pos_tr)}，训练负(第一层): {len(X_neg_tr)}，"
                  f"验证正: {len(X_val_pos)}，验证负: {len(X_val_neg)}，"
                  f"特征维度: {X_pos_tr.shape[1]}")
        else:
            print(f"  -> 旧版缓存不含验证负样本特征，需重新计算（请加 --force_recompute）")
            cache.close()
            cache_ok = False  # 强制重新计算
        del cache

    if not cache_ok:
        print(f"\n[Pipeline] Step 2. 未找到/不含完整缓存，重新计算特征矩阵...")

        # ── 2a：读取训练正样本 ───────────────────────────────
        # 正样本由 load_positive_data() 返回已处理好的 float32 数组列表
        # （内部已完成：读取 → 转灰度 → 裁剪至24×24 → 方差归一化）
        print(f"\n  [2a] 读取训练正样本（max={args.max_pos}）...")
        imgs_pos_tr = load_positive_data("train")
        print(f"  -> 训练正样本: {len(imgs_pos_tr)} 张（已处理好的 float32 数组）")

        # ── 2b：从训练大图随机裁取第一层负样本 ──────────────
        # 处理流程：读彩色大图 → 整张转灰度 → 随机裁24×24 → 方差归一化
        # 注意：_collect_patches_from_dir 内部会打乱图像顺序，不会固定从同一张开始
        print(f"\n  [2b] 从训练大图随机裁取第一层负样本（目标 {args.max_neg} 个）...")
        print(f"       目录: {args.train_neg_dir}（彩色大图）")
        print(f"       流程: 读彩色图 → 整张转灰度 → 随机裁24×24 → 方差归一化")
        imgs_neg_tr = _collect_patches_from_dir(
            image_dir=args.train_neg_dir,
            n_samples=args.max_neg,
            win_size=24,
            mode='random',      # 第一层随机采样，不做级联过滤
        )
        print(f"  -> 第一层训练负样本: {len(imgs_neg_tr)} 个归一化 patch")

        # ── 2c：读取验证正样本 ───────────────────────────────
        print(f"\n  [2c] 读取验证正样本（max={args.max_val_pos}）...")
        imgs_pos_val = load_positive_data("val")
        print(f"  -> 验证正样本: {len(imgs_pos_val)} 张")

        # ── 2d：从验证大图随机裁取验证负样本 ─────────────────
        # 【改动】验证负样本在此一次性准备好，写入缓存，全程固定不变。
        # 旧版：不在此处处理，cascade_trainer 每轮重采样（导致评估标准不一致）。
        # 新版策略：正好有 val_neg_count（默认1000）张大图，每张取1个子窗口。
        #   - one_per_image=True 确保每张大图恰好贡献1个样本，覆盖均匀
        #   - 不同大图的场景、光照、内容各异，多样性有保证
        #   - 处理流程同训练负样本：读彩色图 → 整张转灰度 → 随机裁24×24 → 方差归一化
        print(f"\n  [2d] 从验证大图裁取验证负样本（每张图取1个，共{args.val_neg_count}个）...")
        print(f"       目录: {args.val_neg_dir}（彩色大图）")
        print(f"       策略: 每张大图随机取1个子窗口，均匀覆盖所有大图，全程固定")
        imgs_neg_val = _collect_patches_from_dir(
            image_dir=args.val_neg_dir,
            n_samples=args.val_neg_count,
            win_size=24,
            mode='random',
            one_per_image=True,   # 每张图只取1个，均匀覆盖1000张大图
        )
        print(f"  -> 验证负样本: {len(imgs_neg_val)} 个归一化 patch（全程固定）")

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
        # 【改动】缓存现在包含验证负样本（X_val_neg / y_val_neg）
        # 旧版缓存只有训练正/负/验证正三部分，没有验证负
        print(f"\n  [2g] 保存特征缓存至: {args.features_cache}")
        np.savez_compressed(
            args.features_cache,
            X_pos_tr=X_pos_tr,   y_pos_tr=y_pos_tr,
            X_neg_tr=X_neg_tr,   y_neg_tr=y_neg_tr,
            X_val_pos=X_val_pos, y_val_pos=y_val_pos,
            X_val_neg=X_val_neg, y_val_neg=y_val_neg,   # 新增：验证负样本
        )
        print(f"  -> 缓存保存完成（训练正/负(第一层)/验证正/验证负，四者均固定）")

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
    print(f"  HNM 步长        : {args.hnm_step}")
    print(f"  训练负样本大图  : {args.train_neg_dir}")
    print(f"  验证集负样本    : {len(X_val_neg)} 个，全程固定（已写入缓存）")

    # 【改动】CascadeTrainer 不再需要 val_neg_image_dir / val_neg_per_round 参数
    # 取而代之的是直接传入固定的 X_val_neg / y_val_neg 特征矩阵
    trainer = CascadeTrainer(
        X_pos_train         = X_pos_tr,
        y_pos_train         = y_pos_tr,
        X_neg_train         = X_neg_tr,          # 第一层负样本（后续层由 HNM 生成）
        y_neg_train         = y_neg_tr,
        X_val_pos           = X_val_pos,         # 验证正样本特征矩阵（全程固定）
        y_val_pos           = y_val_pos,
        X_val_neg           = X_val_neg,         # 【改动】验证负样本特征矩阵（全程固定）
        y_val_neg           = y_val_neg,         # 【改动】旧版通过 val_neg_image_dir 每轮采样
        train_neg_image_dir = args.train_neg_dir,# 训练负样本大图目录（HNM 扫描）
        features_desc       = features_desc,
        target_fpr          = args.target_fpr,
        layer_max_fpr       = args.layer_max_fpr,
        layer_min_dr        = args.layer_min_dr,
        max_features_per_stage = args.max_features_per_stage,
        hnm_step            = args.hnm_step,
        checkpoint_path     = args.checkpoint,
        resume_state        = resume_state,
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