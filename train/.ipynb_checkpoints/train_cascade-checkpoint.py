"""
train_cascade.py
================
级联分类器训练主入口脚本
"""

import os
import argparse
import pickle
import numpy as np

from train.data_loader import load_train_data, load_val_data
from train.integral_image import build_batch
from train.haar_features import enumerate_features, compute_all_features
from train.cascade_trainer import CascadeTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Viola-Jones 级联分类器训练脚本")
    # 【修改】提升默认数量级，修改默认目标FPR
    parser.add_argument("--max_pos", type=int, default=2000, help="训练集初始正样本数")
    parser.add_argument("--max_neg", type=int, default=5000, help="训练集初始负样本数")
    parser.add_argument("--target_fpr", type=float, default=1e-5, help="目标整体假正率")
    parser.add_argument("--model_out", type=str, default="../models/cascade_model.pkl", help="最终模型输出路径")
    
    parser.add_argument("--features_cache", type=str, default="../models/features_cache.npz", help="特征矩阵缓存路径")
    parser.add_argument("--checkpoint", type=str, default="../models/cascade_checkpoint.pkl", help="训练断点存档路径")
    parser.add_argument("--force_recompute", action="store_true", help="强制重新计算特征矩阵(忽略缓存)")
    parser.add_argument("--restart", action="store_true", help="强制从第0层重新训练(忽略断点)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    print("\n[Pipeline] 1 & 2. 准备枚举 Haar 特征...")
    features_desc = enumerate_features(win_size=24)

    # =========================================================
    # [新增] 优先从缓存加载耗时的特征矩阵，而不是每次计算
    # =========================================================
# 在计算特征时，强制转换为 float32 节省一半内存
    if os.path.exists(args.features_cache) and not args.force_recompute:
        print(f"\n[Pipeline] 3 & 4. 发现特征缓存文件，极速加载中: {args.features_cache}")
        # 【优化】使用 joblib 或 np.load 处理大文件更优，但最简单的是保持不变
        cache = np.load(args.features_cache)
        X_pos_tr = cache['X_pos_tr'].astype(np.float32) # 【新增】
        y_pos_tr = cache['y_pos_tr'].astype(np.int8)    # 标签用 int8 足矣
        X_neg_tr = cache['X_neg_tr'].astype(np.float32)
        y_neg_tr = cache['y_neg_tr'].astype(np.int8)
        X_val = cache['X_val'].astype(np.float32)
        y_val = cache['y_val'].astype(np.int8)
        
        # 释放 cache 对象的内存
        del cache 
        print(f"  -> 加载完毕！特征维度: {X_pos_tr.shape[1]}")
    else:
        print("\n[Pipeline] 1. 加载图像数据...")
        imgs_pos_tr, imgs_neg_tr = load_train_data(max_pos=args.max_pos, max_neg=args.max_neg)
        
        # 【新增】正样本数据增强：水平翻转
        print("  -> 执行正样本数据增强 (水平翻转)...")
        # 假设 imgs_pos_tr 是 shape (N, 24, 24) 的 numpy 数组 或 list
        if isinstance(imgs_pos_tr, list):
            imgs_pos_tr_flipped = [np.fliplr(img) for img in imgs_pos_tr]
            imgs_pos_tr.extend(imgs_pos_tr_flipped)
        else:
            imgs_pos_tr_flipped = np.flip(imgs_pos_tr, axis=2) # 针对 (N, H, W)
            imgs_pos_tr = np.concatenate((imgs_pos_tr, imgs_pos_tr_flipped), axis=0)
        print(f"  -> 正样本数量扩充至: {len(imgs_pos_tr)}")

        imgs_pos_val, imgs_neg_val = load_val_data(max_pos=100, max_neg=200)

        print("\n[Pipeline] 3. 构建积分图...")
        iimgs_pos_tr = build_batch(imgs_pos_tr)
        iimgs_neg_tr = build_batch(imgs_neg_tr)
        iimgs_pos_val = build_batch(imgs_pos_val)
        iimgs_neg_val = build_batch(imgs_neg_val)

        print("\n[Pipeline] 4. 批量计算特征矩阵 (极为耗时)...")
        # 【优化】确保计算出来的直接是 float32
        X_pos_tr = compute_all_features(iimgs_pos_tr, features_desc, scale=1.0).astype(np.float32)
        X_neg_tr = compute_all_features(iimgs_neg_tr, features_desc, scale=1.0).astype(np.float32)
        y_pos_tr = np.ones(len(X_pos_tr), dtype=np.int8)
        y_neg_tr = np.zeros(len(X_neg_tr), dtype=np.int8)

        X_pos_val = compute_all_features(iimgs_pos_val, features_desc, scale=1.0)
        X_neg_val = compute_all_features(iimgs_neg_val, features_desc, scale=1.0)
        X_val = np.vstack([X_pos_val, X_neg_val])
        y_val = np.hstack([np.ones(len(X_pos_val)), np.zeros(len(X_neg_val))]).astype(np.int32)
        
        # [新增] 将算好的特征保存到磁盘
        print(f"\n[Pipeline] 保存特征缓存至: {args.features_cache}")
        np.savez_compressed(
            args.features_cache, 
            X_pos_tr=X_pos_tr, y_pos_tr=y_pos_tr,
            X_neg_tr=X_neg_tr, y_neg_tr=y_neg_tr,
            X_val=X_val, y_val=y_val
        )

    # =========================================================
    # [新增] 尝试读取上一层的训练断点
    # =========================================================
    resume_state = None
    if os.path.exists(args.checkpoint) and not args.restart:
        with open(args.checkpoint, 'rb') as f:
            resume_state = pickle.load(f)

    # 5. 启动级联训练
    print("\n[Pipeline] 5. 初始化 Cascade Trainer...")
    trainer = CascadeTrainer(
        X_pos_train=X_pos_tr, y_pos_train=y_pos_tr,
        X_neg_train=X_neg_tr, y_neg_train=y_neg_tr,
        X_val=X_val, y_val=y_val,
        target_fpr=args.target_fpr,
        layer_max_fpr=0.50,  
        layer_min_dr=0.995,
        checkpoint_path=args.checkpoint, # 传入保存路径
        resume_state=resume_state        # 传入断点状态
    )
    
    stages = trainer.train()

    # 6. 模型后处理与序列化
    print("\n[Pipeline] 6. 最终模型后处理与序列化保存...")
    for stage in stages:
        for wc in stage.weak_classifiers:
            # 劫持：把索引变成真正的特征元数据 (仅针对没有映射过的 int 索引)
            if isinstance(wc.feature_idx, (int, np.integer)):
                wc.feature_idx = features_desc[wc.feature_idx]
                
    with open(args.model_out, "wb") as f:
        pickle.dump(stages, f)
        
    print(f"[Pipeline] 训练流水线圆满结束！模型已保存至: {args.model_out}")

if __name__ == "__main__":
    main()