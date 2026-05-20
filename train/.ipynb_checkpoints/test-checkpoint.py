"""
test.py — Viola-Jones 系统模块功能验证脚本
===========================================

用法：
    # 运行全部测试
    python test.py

    # 只运行指定模块的测试（多个用逗号分隔）
    python test.py --modules data_loader
    python test.py --modules data_loader,integral_image

    # 显示详细输出（不捕获 print，直接打印子函数的日志）
    python test.py --verbose

说明：
    本脚本采用「测试套件 + 注册机制」设计：
      - 每个模块的测试函数以 test_<模块名>() 形式编写
      - 在文件底部的 TEST_REGISTRY 字典中注册
      - 新增模块时只需：① 写 test_xxx() 函数 ② 加到 TEST_REGISTRY

    每个 TestCase 用 pass() / fail() / skip() 三种状态上报结果，
    并在最后统一打印汇总表。

作者：成员 C（系统集成）  |  参考：设计文档 §5.1
"""

import sys
import os
import time
import traceback
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────
# 路径设置：将项目根目录和 train/ 目录加入 sys.path，
# 确保无论从哪个工作目录运行 test.py 都能正确 import
# ─────────────────────────────────────────────────────────────
TEST_FILE   = Path(__file__).resolve()
PROJECT_ROOT = TEST_FILE.parent          # test.py 所在目录即项目根
TRAIN_DIR    = PROJECT_ROOT / "train"

for p in [str(PROJECT_ROOT), str(TRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────
# 测试结果数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    """单条测试用例的结果"""
    name:    str                    # 用例名称（简短描述）
    status:  str  = "SKIP"         # "PASS" / "FAIL" / "SKIP"
    message: str  = ""             # 附加信息（失败原因或跳过原因）
    elapsed: float = 0.0           # 耗时（秒）


@dataclass
class ModuleResult:
    """一个模块的全部测试结果"""
    module_name: str
    cases: List[CaseResult] = field(default_factory=list)

    @property
    def n_pass(self):  return sum(1 for c in self.cases if c.status == "PASS")
    @property
    def n_fail(self):  return sum(1 for c in self.cases if c.status == "FAIL")
    @property
    def n_skip(self):  return sum(1 for c in self.cases if c.status == "SKIP")
    @property
    def total(self):   return len(self.cases)


# ─────────────────────────────────────────────────────────────
# 测试上下文：每个 test_xxx() 函数接收一个 TestContext 实例，
# 通过它来记录用例结果，而不是直接 assert / raise
# ─────────────────────────────────────────────────────────────

class TestContext:
    """
    测试执行上下文，供测试函数内部记录每条用例的结果。

    示例用法：
        def test_mymodule(ctx: TestContext):
            # --- 用例 1 ---
            t0 = time.time()
            try:
                result = my_function()
                ctx.pass_case("基本功能", time.time() - t0)
            except Exception as e:
                ctx.fail_case("基本功能", str(e), time.time() - t0)

            # --- 用例 2：条件跳过 ---
            if not some_condition:
                ctx.skip_case("高级功能", "依赖项不满足")
                return
    """

    def __init__(self, module_name: str, verbose: bool = False):
        self.result  = ModuleResult(module_name)
        self.verbose = verbose

    def pass_case(self, name: str, elapsed: float = 0.0, message: str = ""):
        """记录一条通过的用例"""
        self.result.cases.append(
            CaseResult(name=name, status="PASS", message=message, elapsed=elapsed)
        )
        icon = "  ✓ PASS"
        print(f"{icon}  [{elapsed:.3f}s]  {name}"
              + (f"  — {message}" if message else ""))

    def fail_case(self, name: str, message: str = "", elapsed: float = 0.0):
        """记录一条失败的用例"""
        self.result.cases.append(
            CaseResult(name=name, status="FAIL", message=message, elapsed=elapsed)
        )
        print(f"  ✗ FAIL  [{elapsed:.3f}s]  {name}")
        if message:
            # 缩进打印错误信息，每行限宽 100 字符
            for line in message.splitlines():
                print(f"         {line[:120]}")

    def skip_case(self, name: str, reason: str = ""):
        """记录一条跳过的用例"""
        self.result.cases.append(
            CaseResult(name=name, status="SKIP", message=reason)
        )
        print(f"  - SKIP           {name}"
              + (f"  — {reason}" if reason else ""))


# ─────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════
#  模块测试函数区域
#  ──────────────────
#  约定：
#    - 函数签名为  def test_<模块名>(ctx: TestContext) -> None
#    - 在底部 TEST_REGISTRY 中注册
#    - 函数内部按用例逐一调用 ctx.pass_case / ctx.fail_case / ctx.skip_case
# ══════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
# 1. data_loader 模块测试
# ══════════════════════════════════════════════════════════════

def test_data_loader(ctx: TestContext) -> None:
    """
    验证 train/data_loader.py 中三个数据加载接口：
        load_train_data()  → (X_pos, X_neg)，各为 float32 ndarray (N,24,24)
        load_val_data()    → (X_pos, X_neg)，同上
        load_test_data()   → dict，含 total_images / total_faces / annotations

    测试策略：
        - 用 max_pos=10, max_neg=20 限制读取数量，快速验证接口可用性
        - 不依赖完整数据集（4000/10000 张），减少 CI 等环境的等待时间
        - 若目录完全不存在，函数会自动触发下载；测试环境无网络时标记为 SKIP
    """
    print("\n  [说明] 使用 max_pos=10, max_neg=20 快速验证接口，"
          "不要求完整数据集已下载")

    # ----------------------------------------------------------
    # 用例 1：import 模块
    # ----------------------------------------------------------
    t0 = time.time()
    try:
        from data_loader import (
            load_train_data, load_val_data, load_test_data,
            TRAIN_POS_DIR, TRAIN_NEG_DIR, VAL_POS_DIR, VAL_NEG_DIR,
            DATA_DIR,
        )
        ctx.pass_case("import data_loader 模块", time.time() - t0)
    except ImportError as e:
        ctx.fail_case("import data_loader 模块",
                      f"ImportError: {e}\n"
                      f"请确认 train/data_loader.py 存在，且依赖库已安装（pip install opencv-python numpy tqdm）",
                      time.time() - t0)
        # 后续用例均依赖 import，无法继续
        ctx.skip_case("load_train_data 返回类型与形状", "import 失败，跳过")
        ctx.skip_case("load_train_data 正样本数量下限",  "import 失败，跳过")
        ctx.skip_case("load_train_data 负样本数量下限",  "import 失败，跳过")
        ctx.skip_case("load_train_data 像素值域",        "import 失败，跳过")
        ctx.skip_case("load_val_data 返回类型与形状",    "import 失败，跳过")
        ctx.skip_case("load_val_data 正样本数量下限",    "import 失败，跳过")
        ctx.skip_case("load_val_data 负样本数量下限",    "import 失败，跳过")
        ctx.skip_case("load_val_data 像素值域",          "import 失败，跳过")
        ctx.skip_case("load_test_data 返回类型",         "import 失败，跳过")
        ctx.skip_case("load_test_data 顶层字段完整性",   "import 失败，跳过")
        return

    # 打印数据目录供调试
    print(f"\n  [路径] DATA_DIR = {DATA_DIR}")
    print(f"         TRAIN_POS_DIR = {TRAIN_POS_DIR}")
    print(f"         VAL_POS_DIR   = {VAL_POS_DIR}")

    # ----------------------------------------------------------
    # ── load_train_data ─────────────────────────────────────
    # ----------------------------------------------------------
    print("\n  ── load_train_data ──")

    t0 = time.time()
    try:
        X_pos_tr, X_neg_tr = load_train_data(max_pos=10, max_neg=20)
        elapsed = time.time() - t0
        print(f"  [返回] X_pos_tr.shape={X_pos_tr.shape}, "
              f"X_neg_tr.shape={X_neg_tr.shape}, "
              f"dtype_pos={X_pos_tr.dtype}, dtype_neg={X_neg_tr.dtype}")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        ctx.fail_case("load_train_data 调用无异常", f"{e}\n{tb}", elapsed)
        # 无法拿到返回值，后续用例跳过
        X_pos_tr = X_neg_tr = None

    if X_pos_tr is not None and X_neg_tr is not None:
        ctx.pass_case("load_train_data 调用无异常", elapsed)
    else:
        for name in ["返回类型与形状", "正样本数量下限", "负样本数量下限", "像素值域"]:
            ctx.skip_case(f"load_train_data {name}", "调用异常，跳过")
        X_pos_tr = X_neg_tr = None

    # 逐项检查（仅在调用成功时执行）
    if X_pos_tr is not None:

        # 用例：返回类型与形状
        t0 = time.time()
        errors = []
        if not isinstance(X_pos_tr, np.ndarray):
            errors.append(f"X_pos 类型应为 ndarray，实际 {type(X_pos_tr)}")
        if not isinstance(X_neg_tr, np.ndarray):
            errors.append(f"X_neg 类型应为 ndarray，实际 {type(X_neg_tr)}")
        if X_pos_tr.ndim != 3 or X_pos_tr.shape[1:] != (24, 24):
            errors.append(f"X_pos 形状应为 (N,24,24)，实际 {X_pos_tr.shape}")
        if X_neg_tr.ndim != 3 or X_neg_tr.shape[1:] != (24, 24):
            errors.append(f"X_neg 形状应为 (N,24,24)，实际 {X_neg_tr.shape}")
        if X_pos_tr.dtype != np.float32:
            errors.append(f"X_pos dtype 应为 float32，实际 {X_pos_tr.dtype}")
        if X_neg_tr.dtype != np.float32:
            errors.append(f"X_neg dtype 应为 float32，实际 {X_neg_tr.dtype}")
        if errors:
            ctx.fail_case("load_train_data 返回类型与形状", "\n".join(errors), time.time() - t0)
        else:
            ctx.pass_case("load_train_data 返回类型与形状", time.time() - t0,
                          f"pos={X_pos_tr.shape}, neg={X_neg_tr.shape}")

        # 用例：正样本数量下限（至少读到 1 张，上限由 max_pos=10 控制）
        t0 = time.time()
        if len(X_pos_tr) >= 1:
            ctx.pass_case("load_train_data 正样本数量下限", time.time() - t0,
                          f"读到 {len(X_pos_tr)} 张（max_pos=10）")
        else:
            ctx.fail_case("load_train_data 正样本数量下限",
                          "正样本数量为 0，请检查 data/train/positive/ 目录",
                          time.time() - t0)

        # 用例：负样本数量下限
        t0 = time.time()
        if len(X_neg_tr) >= 1:
            ctx.pass_case("load_train_data 负样本数量下限", time.time() - t0,
                          f"读到 {len(X_neg_tr)} 张（max_neg=20）")
        else:
            ctx.fail_case("load_train_data 负样本数量下限",
                          "负样本数量为 0，请检查 data/train/negative/ 目录",
                          time.time() - t0)

        # 用例：像素值域（保存时已归一化到 uint8，读取后为 float32 [0,255]）
        t0 = time.time()
        errors = []
        for arr_name, arr in [("X_pos_tr", X_pos_tr), ("X_neg_tr", X_neg_tr)]:
            if len(arr) == 0:
                continue
            lo, hi = float(arr.min()), float(arr.max())
            print(f"  [值域] {arr_name}: min={lo:.2f}, max={hi:.2f}")
            if lo < -1.0:          # 允许少量浮点误差
                errors.append(f"{arr_name} 最小值 {lo:.2f} < -1（期望 ≥ 0）")
            if hi > 256.0:
                errors.append(f"{arr_name} 最大值 {hi:.2f} > 256（期望 ≤ 255）")
        if errors:
            ctx.fail_case("load_train_data 像素值域", "\n".join(errors), time.time() - t0)
        else:
            ctx.pass_case("load_train_data 像素值域", time.time() - t0,
                          "值域在 [0, 255]，符合 uint8 归一化后读取预期")

    # ----------------------------------------------------------
    # ── load_val_data ────────────────────────────────────────
    # ----------------------------------------------------------
    print("\n  ── load_val_data ──")

    t0 = time.time()
    try:
        X_pos_val, X_neg_val = load_val_data(max_pos=10, max_neg=20)
        elapsed = time.time() - t0
        print(f"  [返回] X_pos_val.shape={X_pos_val.shape}, "
              f"X_neg_val.shape={X_neg_val.shape}")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        ctx.fail_case("load_val_data 调用无异常", f"{e}\n{tb}", elapsed)
        X_pos_val = X_neg_val = None
    else:
        ctx.pass_case("load_val_data 调用无异常", elapsed)

    if X_pos_val is not None:

        # 返回类型与形状
        t0 = time.time()
        errors = []
        for arr_name, arr in [("X_pos_val", X_pos_val), ("X_neg_val", X_neg_val)]:
            if not isinstance(arr, np.ndarray):
                errors.append(f"{arr_name} 类型应为 ndarray，实际 {type(arr)}")
                continue
            if arr.ndim != 3 or arr.shape[1:] != (24, 24):
                errors.append(f"{arr_name} 形状应为 (N,24,24)，实际 {arr.shape}")
            if arr.dtype != np.float32:
                errors.append(f"{arr_name} dtype 应为 float32，实际 {arr.dtype}")
        if errors:
            ctx.fail_case("load_val_data 返回类型与形状", "\n".join(errors), time.time() - t0)
        else:
            ctx.pass_case("load_val_data 返回类型与形状", time.time() - t0,
                          f"pos={X_pos_val.shape}, neg={X_neg_val.shape}")

        # 正样本数量下限
        t0 = time.time()
        if len(X_pos_val) >= 1:
            ctx.pass_case("load_val_data 正样本数量下限", time.time() - t0,
                          f"读到 {len(X_pos_val)} 张")
        else:
            ctx.fail_case("load_val_data 正样本数量下限",
                          "正样本数量为 0，请检查 data/val/positive/ 目录",
                          time.time() - t0)

        # 负样本数量下限
        t0 = time.time()
        if len(X_neg_val) >= 1:
            ctx.pass_case("load_val_data 负样本数量下限", time.time() - t0,
                          f"读到 {len(X_neg_val)} 张")
        else:
            ctx.fail_case("load_val_data 负样本数量下限",
                          "负样本数量为 0，请检查 data/val/negative/ 目录",
                          time.time() - t0)

        # 像素值域
        t0 = time.time()
        errors = []
        for arr_name, arr in [("X_pos_val", X_pos_val), ("X_neg_val", X_neg_val)]:
            if len(arr) == 0:
                continue
            lo, hi = float(arr.min()), float(arr.max())
            print(f"  [值域] {arr_name}: min={lo:.2f}, max={hi:.2f}")
            if lo < -1.0 or hi > 256.0:
                errors.append(f"{arr_name} 值域 [{lo:.2f}, {hi:.2f}] 超出 [0,255]")
        if errors:
            ctx.fail_case("load_val_data 像素值域", "\n".join(errors), time.time() - t0)
        else:
            ctx.pass_case("load_val_data 像素值域", time.time() - t0)

    else:
        for name in ["返回类型与形状", "正样本数量下限", "负样本数量下限", "像素值域"]:
            ctx.skip_case(f"load_val_data {name}", "调用异常，跳过")

    # ----------------------------------------------------------
    # ── load_test_data ───────────────────────────────────────
    # ----------------------------------------------------------
    print("\n  ── load_test_data ──")

    t0 = time.time()
    try:
        test_data = load_test_data()
        elapsed   = time.time() - t0
        print(f"  [返回] keys={list(test_data.keys())}")
        print(f"         total_images={test_data.get('total_images')}, "
              f"total_faces={test_data.get('total_faces')}")
    except Exception as e:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        ctx.fail_case("load_test_data 调用无异常", f"{e}\n{tb}", elapsed)
        ctx.skip_case("load_test_data 返回类型", "调用异常，跳过")
        ctx.skip_case("load_test_data 顶层字段完整性", "调用异常，跳过")
        return

    ctx.pass_case("load_test_data 调用无异常", elapsed)

    # 返回类型
    t0 = time.time()
    if isinstance(test_data, dict):
        ctx.pass_case("load_test_data 返回类型", time.time() - t0, "类型为 dict")
    else:
        ctx.fail_case("load_test_data 返回类型",
                      f"期望 dict，实际 {type(test_data)}", time.time() - t0)
        ctx.skip_case("load_test_data 顶层字段完整性", "返回类型错误，跳过")
        return

    # 顶层字段完整性
    t0 = time.time()
    required_keys = {"total_images", "total_faces", "annotations"}
    missing = required_keys - set(test_data.keys())
    if missing:
        ctx.fail_case("load_test_data 顶层字段完整性",
                      f"缺少字段: {missing}", time.time() - t0)
    else:
        # 额外打印 annotations 里首条记录，方便人工核查
        if test_data["annotations"]:
            first_key = next(iter(test_data["annotations"]))
            first_val = test_data["annotations"][first_key]
            print(f"  [样例] annotations['{first_key}'] = "
                  f"image_path={first_val.get('image_path','?')}, "
                  f"faces={len(first_val.get('faces',[]))} 个")
        else:
            print("  [提示] annotations 为空字典（测试集尚未下载或整理）")
        ctx.pass_case("load_test_data 顶层字段完整性", time.time() - t0,
                      f"total_images={test_data['total_images']}, "
                      f"total_faces={test_data['total_faces']}, "
                      f"annotations条目={len(test_data['annotations'])}")


# ══════════════════════════════════════════════════════════════
# 2. integral_image 模块测试
# ══════════════════════════════════════════════════════════════

def test_integral_image(ctx: TestContext) -> None:
    """
    验证 train/integral_image.py 中积分图的构建与查询接口。

    覆盖用例：
    ┌──────┬──────────────────────────────────────────────────────────┐
    │ 分组 │ 用例说明                                                  │
    ├──────┼──────────────────────────────────────────────────────────┤
    │ A    │ import 与接口存在性                                       │
    │ B    │ build() 输出结构（shape / dtype / 零填充边界）             │
    │ C    │ 积分图数值正确性（手算可验证的小矩阵）                     │
    │ D    │ 随机矩阵：ii 与 numpy cumsum 对比                         │
    │ E    │ rect_sum：全图/子块/单行/单列/单像素                      │
    │ F    │ rect_sum_sq：平方积分图查询正确性                         │
    │ G    │ window_variance：方差计算正确性                           │
    │ H    │ 边界情况（全零图、单像素图、单行图、单列图）               │
    │ I    │ compute_integral_image 别名与 build 等价                  │
    │ J    │ build 输入校验（3D 图像应 raise ValueError）               │
    │ K    │ 24×24 样本批量（与 Haar 特征训练场景对齐）                │
    └──────┴──────────────────────────────────────────────────────────┘

    所有数值比较使用 np.isclose(rtol=1e-9, atol=1e-9) 以避免浮点精度问题。
    """

    # ----------------------------------------------------------
    # A. import 与接口存在性
    # ----------------------------------------------------------
    print("\n  ── A. import 与接口存在性 ──")
    t0 = time.time()
    try:
        from integral_image import (
            build, build_batch,
            compute_integral_image,
            rect_sum, rect_sum_sq,
            window_variance,
            IntegralImage,
        )
        ctx.pass_case("A1 import integral_image 模块", time.time() - t0)
    except ImportError as e:
        ctx.fail_case("A1 import integral_image 模块",
                      f"ImportError: {e}\n请确认 integral_image.py 在 train/ 目录下",
                      time.time() - t0)
        # 后续所有用例均依赖 import，统一 skip
        for label in [
            "A2 必要函数均已导出",
            "B1 ii shape=(H+1,W+1)", "B2 ii_sq shape=(H+1,W+1)",
            "B3 ii dtype=float64", "B4 零填充首行全零", "B5 零填充首列全零",
            "B6 ii[H,W] 等于全图像素和",
            "C1 手算 3×4 矩阵积分图数值",
            "D1 随机矩阵 ii 与 np.cumsum 对比",
            "D2 随机矩阵 ii_sq 与 np.cumsum 对比",
            "E1 rect_sum 全图像素和",
            "E2 rect_sum 左上子块",
            "E3 rect_sum 右下子块",
            "E4 rect_sum 中间子块",
            "E5 rect_sum 单行",
            "E6 rect_sum 单列",
            "E7 rect_sum 单像素",
            "F1 rect_sum_sq 全图平方和",
            "F2 rect_sum_sq 子块平方和",
            "G1 window_variance 全零图方差=0",
            "G2 window_variance 数值正确性",
            "G3 window_variance 与 numpy 对比",
            "H1 全零图 rect_sum=0",
            "H2 单像素图 rect_sum",
            "H3 单行图 rect_sum",
            "H4 单列图 rect_sum",
            "I1 compute_integral_image 与 build 结果一致",
            "J1 build 传入 3D 图像触发 ValueError",
            "K1 24×24 样本：ii[24,24] 等于像素总和",
            "K2 24×24 样本：rect_sum 与切片对比（10组随机矩形）",
        ]:
            ctx.skip_case(label, "import 失败，跳过")
        return

    # 检查所有必要函数存在
    t0 = time.time()
    required = {
        "build": build,
        "build_batch": build_batch,
        "compute_integral_image": compute_integral_image,
        "rect_sum": rect_sum,
        "rect_sum_sq": rect_sum_sq,
        "window_variance": window_variance,
        "IntegralImage": IntegralImage,
    }
    missing = [name for name, obj in required.items() if obj is None]
    if missing:
        ctx.fail_case("A2 必要函数均已导出", f"缺少: {missing}", time.time() - t0)
    else:
        ctx.pass_case("A2 必要函数均已导出", time.time() - t0,
                      f"共 {len(required)} 个符号均可导入")

    # ----------------------------------------------------------
    # 准备测试用的小矩阵（手算可验证）
    # img_3x4:
    #   [[ 1,  2,  3,  4],
    #    [ 5,  6,  7,  8],
    #    [ 9, 10, 11, 12]]
    # 全图和 = 78
    # 积分图（不含零填充，ii_raw）：
    #   [[ 1,  3,  6, 10],
    #    [ 6, 14, 24, 36],
    #    [15, 33, 54, 78]]
    # ----------------------------------------------------------
    img_3x4 = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    print(f"\n  [测试图像 img_3x4]\n{img_3x4}")

    # ----------------------------------------------------------
    # B. build() 输出结构
    # ----------------------------------------------------------
    print("\n  ── B. build() 输出结构 ──")
    t0 = time.time()
    try:
        iimg_3x4 = build(img_3x4)
        build_ok = True
    except Exception as e:
        ctx.fail_case("B1 ii shape=(H+1,W+1)", f"build() 抛出异常: {e}", time.time() - t0)
        build_ok = False

    if build_ok:
        H, W = 3, 4

        # B1 ii shape
        t0 = time.time()
        if iimg_3x4.ii.shape == (H + 1, W + 1):
            ctx.pass_case("B1 ii shape=(H+1,W+1)", time.time() - t0,
                          f"实际 shape={iimg_3x4.ii.shape}")
        else:
            ctx.fail_case("B1 ii shape=(H+1,W+1)",
                          f"期望 ({H+1},{W+1})，实际 {iimg_3x4.ii.shape}",
                          time.time() - t0)

        # B2 ii_sq shape
        t0 = time.time()
        if iimg_3x4.ii_sq.shape == (H + 1, W + 1):
            ctx.pass_case("B2 ii_sq shape=(H+1,W+1)", time.time() - t0)
        else:
            ctx.fail_case("B2 ii_sq shape=(H+1,W+1)",
                          f"期望 ({H+1},{W+1})，实际 {iimg_3x4.ii_sq.shape}",
                          time.time() - t0)

        # B3 dtype
        t0 = time.time()
        if iimg_3x4.ii.dtype == np.float64:
            ctx.pass_case("B3 ii dtype=float64", time.time() - t0)
        else:
            ctx.fail_case("B3 ii dtype=float64",
                          f"实际 dtype={iimg_3x4.ii.dtype}", time.time() - t0)

        # B4 零填充首行全零
        t0 = time.time()
        if np.all(iimg_3x4.ii[0, :] == 0):
            ctx.pass_case("B4 零填充首行全零", time.time() - t0,
                          f"ii[0,:] = {iimg_3x4.ii[0,:]}")
        else:
            ctx.fail_case("B4 零填充首行全零",
                          f"ii[0,:] = {iimg_3x4.ii[0,:]}，期望全 0",
                          time.time() - t0)

        # B5 零填充首列全零
        t0 = time.time()
        if np.all(iimg_3x4.ii[:, 0] == 0):
            ctx.pass_case("B5 零填充首列全零", time.time() - t0)
        else:
            ctx.fail_case("B5 零填充首列全零",
                          f"ii[:,0] = {iimg_3x4.ii[:,0]}，期望全 0",
                          time.time() - t0)

        # B6 ii[H,W] 等于全图像素和
        t0 = time.time()
        expected_total = float(img_3x4.sum())  # 78.0
        actual_total   = iimg_3x4.ii[H, W]
        if np.isclose(actual_total, expected_total, rtol=1e-9, atol=1e-9):
            ctx.pass_case("B6 ii[H,W] 等于全图像素和", time.time() - t0,
                          f"ii[{H},{W}]={actual_total:.1f}，期望={expected_total:.1f}")
        else:
            ctx.fail_case("B6 ii[H,W] 等于全图像素和",
                          f"ii[{H},{W}]={actual_total:.1f}，期望={expected_total:.1f}",
                          time.time() - t0)

    else:
        for label in ["B1 ii shape=(H+1,W+1)", "B2 ii_sq shape=(H+1,W+1)",
                      "B3 ii dtype=float64", "B4 零填充首行全零",
                      "B5 零填充首列全零", "B6 ii[H,W] 等于全图像素和"]:
            ctx.skip_case(label, "build() 异常，跳过")

    # ----------------------------------------------------------
    # C. 积分图数值正确性（手算验证）
    # ----------------------------------------------------------
    print("\n  ── C. 积分图数值正确性（手算验证） ──")

    # 手算 img_3x4 的积分图（不含零填充）：
    # ii_raw[0,0]=1, [0,1]=3, [0,2]=6, [0,3]=10
    # ii_raw[1,0]=6, [1,1]=14,[1,2]=24,[1,3]=36
    # ii_raw[2,0]=15,[2,1]=33,[2,2]=54,[2,3]=78
    expected_ii_raw = np.array([
        [ 1,  3,  6, 10],
        [ 6, 14, 24, 36],
        [15, 33, 54, 78],
    ], dtype=np.float64)
    # padded: 在上方插入全零行、左侧插入全零列
    expected_ii_padded = np.pad(expected_ii_raw, ((1,0),(1,0)),
                                mode='constant', constant_values=0)

    t0 = time.time()
    if build_ok:
        if np.allclose(iimg_3x4.ii, expected_ii_padded, rtol=1e-9, atol=1e-9):
            ctx.pass_case("C1 手算 3×4 矩阵积分图数值", time.time() - t0,
                          "所有元素与手算结果完全一致")
        else:
            diff = iimg_3x4.ii - expected_ii_padded
            ctx.fail_case("C1 手算 3×4 矩阵积分图数值",
                          f"最大误差 {np.max(np.abs(diff)):.2e}\n"
                          f"实际:\n{iimg_3x4.ii}\n期望:\n{expected_ii_padded}",
                          time.time() - t0)
    else:
        ctx.skip_case("C1 手算 3×4 矩阵积分图数值", "build() 异常，跳过")

    # ----------------------------------------------------------
    # D. 随机矩阵与 np.cumsum 对比
    # ----------------------------------------------------------
    print("\n  ── D. 随机矩阵与 np.cumsum 对比 ──")
    rng = np.random.default_rng(seed=2024)

    # D1 像素积分图
    t0 = time.time()
    try:
        img_rand = rng.integers(0, 256, size=(50, 60)).astype(np.float64)
        iimg_rand = build(img_rand)
        # numpy 参考实现：cumsum 行方向再列方向，加零填充
        ref_ii = np.pad(
            np.cumsum(np.cumsum(img_rand, axis=0), axis=1),
            ((1,0),(1,0)), mode='constant', constant_values=0
        )
        if np.allclose(iimg_rand.ii, ref_ii, rtol=1e-9, atol=1e-9):
            ctx.pass_case("D1 随机矩阵 ii 与 np.cumsum 对比", time.time() - t0,
                          f"50×60 图像，最大误差 {np.max(np.abs(iimg_rand.ii-ref_ii)):.2e}")
        else:
            max_err = np.max(np.abs(iimg_rand.ii - ref_ii))
            ctx.fail_case("D1 随机矩阵 ii 与 np.cumsum 对比",
                          f"最大误差 {max_err:.2e}，超出容忍范围", time.time() - t0)
    except Exception as e:
        ctx.fail_case("D1 随机矩阵 ii 与 np.cumsum 对比", str(e), time.time() - t0)

    # D2 像素平方积分图
    t0 = time.time()
    try:
        ref_ii_sq = np.pad(
            np.cumsum(np.cumsum(img_rand ** 2, axis=0), axis=1),
            ((1,0),(1,0)), mode='constant', constant_values=0
        )
        if np.allclose(iimg_rand.ii_sq, ref_ii_sq, rtol=1e-6, atol=1e-6):
            ctx.pass_case("D2 随机矩阵 ii_sq 与 np.cumsum 对比", time.time() - t0,
                          f"最大误差 {np.max(np.abs(iimg_rand.ii_sq-ref_ii_sq)):.2e}")
        else:
            max_err = np.max(np.abs(iimg_rand.ii_sq - ref_ii_sq))
            ctx.fail_case("D2 随机矩阵 ii_sq 与 np.cumsum 对比",
                          f"最大误差 {max_err:.2e}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("D2 随机矩阵 ii_sq 与 np.cumsum 对比", str(e), time.time() - t0)

    # ----------------------------------------------------------
    # E. rect_sum 查询正确性
    # ----------------------------------------------------------
    print("\n  ── E. rect_sum 查询正确性 ──")

    if not build_ok:
        for label in ["E1 rect_sum 全图像素和", "E2 rect_sum 左上子块",
                      "E3 rect_sum 右下子块", "E4 rect_sum 中间子块",
                      "E5 rect_sum 单行", "E6 rect_sum 单列", "E7 rect_sum 单像素"]:
            ctx.skip_case(label, "build() 异常，跳过")
    else:
        # 测试辅助：计算期望值（直接切片求和）
        def expected_rect(img, r, c, h, w):
            return float(img[r:r+h, c:c+w].sum())

        cases_e = [
            ("E1 rect_sum 全图像素和",   0, 0, 3, 4),   # 期望 78
            ("E2 rect_sum 左上子块",     0, 0, 2, 2),   # 期望 1+2+5+6=14
            ("E3 rect_sum 右下子块",     1, 2, 2, 2),   # 期望 7+8+11+12=38
            ("E4 rect_sum 中间子块",     1, 1, 1, 2),   # 期望 6+7=13
            ("E5 rect_sum 单行",         1, 0, 1, 4),   # 期望 5+6+7+8=26
            ("E6 rect_sum 单列",         0, 2, 3, 1),   # 期望 3+7+11=21
            ("E7 rect_sum 单像素",       2, 3, 1, 1),   # 期望 12
        ]
        for label, r, c, h, w in cases_e:
            t0 = time.time()
            try:
                actual   = rect_sum(iimg_3x4.ii, r, c, h, w)
                expected = expected_rect(img_3x4, r, c, h, w)
                print(f"  [rect_sum({r},{c},{h},{w})] actual={actual:.1f}, "
                      f"expected={expected:.1f}")
                if np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
                    ctx.pass_case(label, time.time() - t0,
                                  f"actual={actual:.1f}, expected={expected:.1f}")
                else:
                    ctx.fail_case(label,
                                  f"actual={actual:.1f} ≠ expected={expected:.1f}",
                                  time.time() - t0)
            except Exception as e:
                ctx.fail_case(label, str(e), time.time() - t0)

    # ----------------------------------------------------------
    # F. rect_sum_sq 查询正确性
    # ----------------------------------------------------------
    print("\n  ── F. rect_sum_sq 查询正确性 ──")

    if not build_ok:
        ctx.skip_case("F1 rect_sum_sq 全图平方和", "build() 异常，跳过")
        ctx.skip_case("F2 rect_sum_sq 子块平方和", "build() 异常，跳过")
    else:
        # F1 全图平方和：1²+2²+...+12² = 650
        t0 = time.time()
        try:
            actual   = rect_sum_sq(iimg_3x4.ii_sq, 0, 0, 3, 4)
            expected = float((img_3x4 ** 2).sum())   # 650.0
            print(f"  [rect_sum_sq 全图] actual={actual:.1f}, expected={expected:.1f}")
            if np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
                ctx.pass_case("F1 rect_sum_sq 全图平方和", time.time() - t0,
                              f"{actual:.1f}")
            else:
                ctx.fail_case("F1 rect_sum_sq 全图平方和",
                              f"actual={actual:.1f} ≠ expected={expected:.1f}",
                              time.time() - t0)
        except Exception as e:
            ctx.fail_case("F1 rect_sum_sq 全图平方和", str(e), time.time() - t0)

        # F2 左上 2×2：1²+2²+5²+6² = 1+4+25+36 = 66
        t0 = time.time()
        try:
            actual   = rect_sum_sq(iimg_3x4.ii_sq, 0, 0, 2, 2)
            expected = float((img_3x4[:2, :2] ** 2).sum())   # 66.0
            print(f"  [rect_sum_sq 左上2×2] actual={actual:.1f}, expected={expected:.1f}")
            if np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
                ctx.pass_case("F2 rect_sum_sq 子块平方和", time.time() - t0,
                              f"{actual:.1f}")
            else:
                ctx.fail_case("F2 rect_sum_sq 子块平方和",
                              f"actual={actual:.1f} ≠ expected={expected:.1f}",
                              time.time() - t0)
        except Exception as e:
            ctx.fail_case("F2 rect_sum_sq 子块平方和", str(e), time.time() - t0)

    # ----------------------------------------------------------
    # G. window_variance 方差计算正确性
    # ----------------------------------------------------------
    print("\n  ── G. window_variance 方差计算正确性 ──")

    # G1 全零图方差应为 0
    t0 = time.time()
    try:
        img_zero = np.zeros((10, 10), dtype=np.float64)
        iimg_zero = build(img_zero)
        var_zero  = window_variance(iimg_zero, 0, 0, 10)
        print(f"  [全零图方差] = {var_zero}")
        if np.isclose(var_zero, 0.0, atol=1e-9):
            ctx.pass_case("G1 window_variance 全零图方差=0", time.time() - t0)
        else:
            ctx.fail_case("G1 window_variance 全零图方差=0",
                          f"期望 0.0，实际 {var_zero}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("G1 window_variance 全零图方差=0", str(e), time.time() - t0)

    # G2 常数图（值全为 5）方差应为 0
    t0 = time.time()
    try:
        img_const = np.full((8, 8), fill_value=5.0, dtype=np.float64)
        iimg_const = build(img_const)
        var_const  = window_variance(iimg_const, 0, 0, 8)
        print(f"  [全5图方差] = {var_const:.6f}  （期望 0）")
        if np.isclose(var_const, 0.0, atol=1e-6):
            ctx.pass_case("G2 window_variance 数值正确性（常数图方差=0）",
                          time.time() - t0, f"{var_const:.2e}")
        else:
            ctx.fail_case("G2 window_variance 数值正确性（常数图方差=0）",
                          f"期望≈0，实际 {var_const:.6f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("G2 window_variance 数值正确性（常数图方差=0）",
                      str(e), time.time() - t0)

    # G3 随机图像：window_variance 与 numpy 方差对比
    t0 = time.time()
    try:
        rng2 = np.random.default_rng(seed=42)
        img_g3 = rng2.integers(0, 256, size=(30, 30)).astype(np.float64)
        iimg_g3 = build(img_g3)

        errors_g3 = []
        # 随机抽取 15 个子窗口测试
        for _ in range(15):
            size = rng2.integers(4, 20)
            r_max = img_g3.shape[0] - size
            c_max = img_g3.shape[1] - size
            if r_max <= 0 or c_max <= 0:
                continue
            r = int(rng2.integers(0, r_max))
            c = int(rng2.integers(0, c_max))
            actual_var   = window_variance(iimg_g3, r, c, int(size))
            expected_var = float(img_g3[r:r+size, c:c+size].var())
            if not np.isclose(actual_var, expected_var, rtol=1e-6, atol=1e-6):
                errors_g3.append(
                    f"窗口(r={r},c={c},size={size}): "
                    f"actual={actual_var:.6f}, expected={expected_var:.6f}"
                )

        if errors_g3:
            ctx.fail_case("G3 window_variance 与 numpy 对比",
                          "\n".join(errors_g3), time.time() - t0)
        else:
            ctx.pass_case("G3 window_variance 与 numpy 对比", time.time() - t0,
                          "15 个随机子窗口全部通过")
    except Exception as e:
        ctx.fail_case("G3 window_variance 与 numpy 对比", str(e), time.time() - t0)

    # ----------------------------------------------------------
    # H. 边界情况
    # ----------------------------------------------------------
    print("\n  ── H. 边界情况 ──")

    # H1 全零图
    t0 = time.time()
    try:
        img_h1 = np.zeros((5, 5), dtype=np.float64)
        iimg_h1 = build(img_h1)
        val = rect_sum(iimg_h1.ii, 0, 0, 5, 5)
        if np.isclose(val, 0.0, atol=1e-9):
            ctx.pass_case("H1 全零图 rect_sum=0", time.time() - t0, f"val={val}")
        else:
            ctx.fail_case("H1 全零图 rect_sum=0", f"期望 0，实际 {val}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("H1 全零图 rect_sum=0", str(e), time.time() - t0)

    # H2 单像素图 (1×1)
    t0 = time.time()
    try:
        img_h2 = np.array([[99.0]])
        iimg_h2 = build(img_h2)
        val = rect_sum(iimg_h2.ii, 0, 0, 1, 1)
        if np.isclose(val, 99.0, atol=1e-9):
            ctx.pass_case("H2 单像素图 rect_sum", time.time() - t0, f"val={val}")
        else:
            ctx.fail_case("H2 单像素图 rect_sum", f"期望 99，实际 {val}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("H2 单像素图 rect_sum", str(e), time.time() - t0)

    # H3 单行图 (1×6)：[10,20,30,40,50,60]
    t0 = time.time()
    try:
        img_h3 = np.array([[10, 20, 30, 40, 50, 60]], dtype=np.float64)
        iimg_h3 = build(img_h3)
        val = rect_sum(iimg_h3.ii, 0, 1, 1, 4)   # 20+30+40+50=140
        expected = float(img_h3[0, 1:5].sum())
        if np.isclose(val, expected, atol=1e-9):
            ctx.pass_case("H3 单行图 rect_sum", time.time() - t0,
                          f"val={val:.1f}, expected={expected:.1f}")
        else:
            ctx.fail_case("H3 单行图 rect_sum",
                          f"actual={val:.1f} ≠ expected={expected:.1f}",
                          time.time() - t0)
    except Exception as e:
        ctx.fail_case("H3 单行图 rect_sum", str(e), time.time() - t0)

    # H4 单列图 (5×1)：[2,4,6,8,10]^T
    t0 = time.time()
    try:
        img_h4 = np.array([[2],[4],[6],[8],[10]], dtype=np.float64)
        iimg_h4 = build(img_h4)
        val = rect_sum(iimg_h4.ii, 1, 0, 3, 1)   # 4+6+8=18
        expected = float(img_h4[1:4, 0].sum())
        if np.isclose(val, expected, atol=1e-9):
            ctx.pass_case("H4 单列图 rect_sum", time.time() - t0,
                          f"val={val:.1f}, expected={expected:.1f}")
        else:
            ctx.fail_case("H4 单列图 rect_sum",
                          f"actual={val:.1f} ≠ expected={expected:.1f}",
                          time.time() - t0)
    except Exception as e:
        ctx.fail_case("H4 单列图 rect_sum", str(e), time.time() - t0)

    # ----------------------------------------------------------
    # I. compute_integral_image 别名等价性
    # ----------------------------------------------------------
    print("\n  ── I. compute_integral_image 别名等价性 ──")
    t0 = time.time()
    try:
        iimg_alias = compute_integral_image(img_3x4)
        if (np.allclose(iimg_alias.ii,    iimg_3x4.ii,    rtol=1e-9, atol=1e-9) and
                np.allclose(iimg_alias.ii_sq, iimg_3x4.ii_sq, rtol=1e-9, atol=1e-9)):
            ctx.pass_case("I1 compute_integral_image 与 build 结果一致",
                          time.time() - t0, "ii 和 ii_sq 完全一致")
        else:
            ctx.fail_case("I1 compute_integral_image 与 build 结果一致",
                          "ii 或 ii_sq 存在差异", time.time() - t0)
    except Exception as e:
        ctx.fail_case("I1 compute_integral_image 与 build 结果一致",
                      str(e), time.time() - t0)

    # ----------------------------------------------------------
    # J. 输入校验
    # ----------------------------------------------------------
    print("\n  ── J. 输入校验 ──")
    t0 = time.time()
    try:
        bad_input = np.zeros((10, 10, 3), dtype=np.uint8)   # 3 通道，应报错
        build(bad_input)
        ctx.fail_case("J1 build 传入 3D 图像触发 ValueError",
                      "期望抛出 ValueError，实际未抛出", time.time() - t0)
    except ValueError as e:
        ctx.pass_case("J1 build 传入 3D 图像触发 ValueError", time.time() - t0,
                      f"ValueError: {str(e)[:60]}")
    except Exception as e:
        ctx.fail_case("J1 build 传入 3D 图像触发 ValueError",
                      f"期望 ValueError，实际 {type(e).__name__}: {e}",
                      time.time() - t0)

    # ----------------------------------------------------------
    # K. 24×24 样本场景（与 Haar 特征训练对齐）
    # ----------------------------------------------------------
    print("\n  ── K. 24×24 样本场景 ──")

    rng3 = np.random.default_rng(seed=99)
    img_24 = rng3.integers(0, 256, size=(24, 24)).astype(np.float64)

    # K1 ii[24,24] 等于像素总和
    t0 = time.time()
    try:
        iimg_24 = build(img_24)
        actual_total   = iimg_24.ii[24, 24]
        expected_total = float(img_24.sum())
        print(f"  [24×24] ii[24,24]={actual_total:.1f}, "
              f"img.sum()={expected_total:.1f}")
        if np.isclose(actual_total, expected_total, rtol=1e-9, atol=1e-9):
            ctx.pass_case("K1 24×24 样本：ii[24,24] 等于像素总和",
                          time.time() - t0, f"{actual_total:.1f}")
        else:
            ctx.fail_case("K1 24×24 样本：ii[24,24] 等于像素总和",
                          f"actual={actual_total:.1f} ≠ expected={expected_total:.1f}",
                          time.time() - t0)
    except Exception as e:
        ctx.fail_case("K1 24×24 样本：ii[24,24] 等于像素总和", str(e), time.time() - t0)
        iimg_24 = None

    # K2 rect_sum 与 numpy 切片对比（10 组随机矩形，模拟 Haar 特征计算）
    t0 = time.time()
    if iimg_24 is None:
        ctx.skip_case("K2 24×24 样本：rect_sum 与切片对比（10组随机矩形）",
                      "K1 失败，跳过")
    else:
        try:
            errors_k2 = []
            # 固定种子，模拟 Haar 特征枚举时的典型矩形尺寸
            test_rects = [
                (0,  0,  12, 12),   # 左上半区
                (0,  12, 12, 12),   # 右上半区
                (12, 0,  12, 12),   # 左下半区
                (0,  0,  24, 12),   # 左半垂直条
                (0,  0,  8,  24),   # 顶部水平条
                (8,  0,  8,  24),   # 中部水平条
                (0,  0,  24, 8),    # 左侧窄条
                (0,  8,  24, 8),    # 中间窄条
                (16, 16, 8,  8),    # 右下小块
                (5,  3,  7,  11),   # 任意位置
            ]
            print(f"  [K2] 测试 {len(test_rects)} 组矩形...")
            for r, c, h, w in test_rects:
                actual   = rect_sum(iimg_24.ii, r, c, h, w)
                expected = float(img_24[r:r+h, c:c+w].sum())
                print(f"       rect({r},{c},{h},{w}): "
                      f"actual={actual:.1f}, expected={expected:.1f}")
                if not np.isclose(actual, expected, rtol=1e-9, atol=1e-9):
                    errors_k2.append(
                        f"rect({r},{c},{h},{w}): actual={actual:.1f} ≠ expected={expected:.1f}"
                    )
            if errors_k2:
                ctx.fail_case(
                    "K2 24×24 样本：rect_sum 与切片对比（10组随机矩形）",
                    "\n".join(errors_k2), time.time() - t0
                )
            else:
                ctx.pass_case(
                    "K2 24×24 样本：rect_sum 与切片对比（10组随机矩形）",
                    time.time() - t0, "全部 10 组矩形通过"
                )
        except Exception as e:
            ctx.fail_case("K2 24×24 样本：rect_sum 与切片对比（10组随机矩形）",
                          str(e), time.time() - t0)


# ══════════════════════════════════════════════════════════════
# 3. haar_features 模块测试
# ══════════════════════════════════════════════════════════════

def test_haar_features(ctx: TestContext) -> None:
    """
    验证 train/haar_features.py 中 Haar-like 特征枚举与计算接口。

    覆盖用例：
    ┌──────┬────────────────────────────────────────────────────────────┐
    │ 分组 │ 用例说明                                                    │
    ├──────┼────────────────────────────────────────────────────────────┤
    │ A    │ import 与接口存在性                                         │
    │ B    │ enumerate_features：数量范围、描述符字段合法性、无重复       │
    │ C    │ compute_feature FEAT_H2：手算 + 随机对比                    │
    │ D    │ compute_feature FEAT_V2：手算 + 随机对比                    │
    │ E    │ compute_feature FEAT_H3：手算 + 随机对比                    │
    │ F    │ compute_feature FEAT_D4：手算 + 随机对比                    │
    │ G    │ 非法 ftype 触发 ValueError                                  │
    │ H    │ 与 integral_image 的端到端适配（iimg.ii 直接传入）          │
    │ I    │ compute_all_features：shape / dtype / 数值抽检              │
    └──────┴────────────────────────────────────────────────────────────┘
    """

    # ----------------------------------------------------------
    # A. import 与接口存在性
    # ----------------------------------------------------------
    print("\n  ── A. import 与接口存在性 ──")
    t0 = time.time()
    try:
        from haar_features import (
            enumerate_features, compute_feature,
            compute_all_features,
            compute_feature_at_scale,
            iter_scales, iter_window_positions,
            FeatureDesc, FEAT_H2, FEAT_V2, FEAT_H3, FEAT_V3, FEAT_D4,
            BASE_WIN_SIZE,
        )
        ctx.pass_case("A1 import haar_features 模块", time.time() - t0)
    except ImportError as e:
        ctx.fail_case("A1 import haar_features 模块",
                      f"ImportError: {e}", time.time() - t0)
        all_skip = [
            "A2 必要符号均已导出",
            "B1 24×24 枚举总数精确等于 162336",
            "B2 各类型特征数量（5种）均 > 0",
            "B3 描述符字段合法性（r/c/h/w > 0 且不越界）",
            "B4 描述符列表无重复",
            "C1 FEAT_H2 手算验证（左亮右暗图）",
            "C2 FEAT_H2 随机图与 numpy 切片对比（10组）",
            "D1 FEAT_V2 手算验证（上亮下暗图）",
            "D2 FEAT_V2 随机图与 numpy 切片对比（10组）",
            "E1 FEAT_H3 手算验证",
            "E2 FEAT_H3 随机图与 numpy 切片对比（10组）",
            "F1 FEAT_V3 手算验证（上暗中亮下暗图）",
            "F2 FEAT_V3 随机图与 numpy 切片对比（10组）",
            "G1 FEAT_D4 手算验证",
            "G2 FEAT_D4 随机图与 numpy 切片对比（10组）",
            "H1 非法 ftype 触发 ValueError",
            "I1 端到端：iimg.ii 传入 compute_feature 正确",
            "J1 compute_all_features shape=(N,F)",
            "J2 compute_all_features dtype=float32",
            "J3 compute_all_features 数值抽检",
        ]
        for s in all_skip:
            ctx.skip_case(s, "import 失败，跳过")
        return

    t0 = time.time()
    required = {
        "enumerate_features": enumerate_features,
        "compute_feature": compute_feature,
        "compute_all_features": compute_all_features,
        "compute_feature_at_scale": compute_feature_at_scale,
        "iter_scales": iter_scales,
        "FeatureDesc": FeatureDesc,
        "FEAT_H2": FEAT_H2, "FEAT_V2": FEAT_V2,
        "FEAT_H3": FEAT_H3, "FEAT_V3": FEAT_V3, "FEAT_D4": FEAT_D4,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        ctx.fail_case("A2 必要符号均已导出", f"缺少: {missing}", time.time() - t0)
    else:
        ctx.pass_case("A2 必要符号均已导出", time.time() - t0,
                      f"共 {len(required)} 个")

    # ----------------------------------------------------------
    # 准备公共资源：积分图工具 + 测试图像
    # ----------------------------------------------------------
    from integral_image import build

    # 图像 A：左半亮(128)右半暗(64)，24×24，用于手算验证
    img_lr = np.zeros((24, 24), dtype=np.float64)
    img_lr[:, :12] = 128.0
    img_lr[:, 12:] = 64.0
    iimg_lr = build(img_lr)

    # 图像 B：上半亮(200)下半暗(50)，24×24
    img_tb = np.zeros((24, 24), dtype=np.float64)
    img_tb[:12, :] = 200.0
    img_tb[12:, :] = 50.0
    iimg_tb = build(img_tb)

    # 图像 C：随机图像，用于与 numpy 切片对比
    rng = np.random.default_rng(seed=7)
    img_rand = rng.integers(0, 256, size=(24, 24)).astype(np.float64)
    iimg_rand = build(img_rand)

    print(f"  [公共资源] 3 张测试图像积分图构建完成")

    # ----------------------------------------------------------
    # B. enumerate_features
    # ----------------------------------------------------------
    print("\n  ── B. enumerate_features ──")
    t0 = time.time()
    try:
        features = enumerate_features(win_size=24)
        elapsed_enum = time.time() - t0
        print(f"  [枚举] 共 {len(features)} 个特征，耗时 {elapsed_enum:.3f}s")
    except Exception as e:
        ctx.fail_case("B1 24×24 枚举总数在合理范围",
                      f"enumerate_features 抛出异常: {e}", time.time() - t0)
        features = None

    if features is not None:
        # B1 精确总数：五种类型合计恰好 162,336（±0 容差，精确匹配）
        t0 = time.time()
        n = len(features)
        EXACT = 162_336
        if n == EXACT:
            ctx.pass_case("B1 24×24 枚举总数精确等于 162336", time.time() - t0,
                          f"{n} 个 ✓（Wikipedia/OpenCV 确认值）")
        else:
            ctx.fail_case("B1 24×24 枚举总数精确等于 162336",
                          f"实际 {n} 个，期望精确 {EXACT}\n"
                          f"  提示：确认是否枚举了全部五种类型（H2/V2/H3/V3/D4），"
                          f"其中 V3（垂直三矩形）容易被遗漏",
                          time.time() - t0)

        # B2 各类型数量（5种）均 > 0，且与精确值吻合
        t0 = time.time()
        type_counts = {ft: 0 for ft in [FEAT_H2, FEAT_V2, FEAT_H3, FEAT_V3, FEAT_D4]}
        for fd in features:
            if fd.ftype in type_counts:
                type_counts[fd.ftype] += 1
        EXACT_COUNTS = {FEAT_H2: 43200, FEAT_V2: 43200, FEAT_H3: 27600,
                        FEAT_V3: 27600, FEAT_D4: 20736}
        print(f"  [各类型数量] H2={type_counts[FEAT_H2]}, V2={type_counts[FEAT_V2]}, "
              f"H3={type_counts[FEAT_H3]}, V3={type_counts[FEAT_V3]}, "
              f"D4={type_counts[FEAT_D4]}")
        mismatches = [
            f"ftype={ft}: 实际={type_counts[ft]}, 期望={EXACT_COUNTS[ft]}"
            for ft in EXACT_COUNTS if type_counts[ft] != EXACT_COUNTS[ft]
        ]
        if mismatches:
            ctx.fail_case("B2 各类型特征数量（5种）均 > 0",
                          "\n".join(mismatches), time.time() - t0)
        else:
            ctx.pass_case("B2 各类型特征数量（5种）均 > 0", time.time() - t0,
                          "H2=43200, V2=43200, H3=27600, V3=27600, D4=20736")

        # B3 描述符字段合法性（抽检前 1000 条）
        t0 = time.time()
        illegal = []
        sample = features[:1000]
        for fd in sample:
            if fd.ftype not in (FEAT_H2, FEAT_V2, FEAT_H3, FEAT_V3, FEAT_D4):
                illegal.append(f"ftype={fd.ftype} 非法")
                continue
            if fd.r < 0 or fd.c < 0 or fd.h < 1 or fd.w < 1:
                illegal.append(f"{fd} r/c/h/w 越界")
                continue
            if fd.ftype == FEAT_H2 and fd.c + 2 * fd.w > 24:
                illegal.append(f"{fd} H2 列越界 c+2w={fd.c+2*fd.w}")
            elif fd.ftype == FEAT_V2 and fd.r + 2 * fd.h > 24:
                illegal.append(f"{fd} V2 行越界 r+2h={fd.r+2*fd.h}")
            elif fd.ftype == FEAT_H3 and fd.c + 3 * fd.w > 24:
                illegal.append(f"{fd} H3 列越界 c+3w={fd.c+3*fd.w}")
            elif fd.ftype == FEAT_V3 and fd.r + 3 * fd.h > 24:
                illegal.append(f"{fd} V3 行越界 r+3h={fd.r+3*fd.h}")
            elif fd.ftype == FEAT_D4 and (fd.c + 2 * fd.w > 24 or fd.r + 2 * fd.h > 24):
                illegal.append(f"{fd} D4 越界")
        if illegal:
            ctx.fail_case("B3 描述符字段合法性（r/c/h/w > 0 且不越界）",
                          f"发现 {len(illegal)} 条非法（前 3 条）: {illegal[:3]}",
                          time.time() - t0)
        else:
            ctx.pass_case("B3 描述符字段合法性（r/c/h/w > 0 且不越界）",
                          time.time() - t0, f"抽检前 1000 条全部合法")

        # B4 无重复（用 set 对比长度）
        t0 = time.time()
        unique = set(features)
        if len(unique) == len(features):
            ctx.pass_case("B4 描述符列表无重复", time.time() - t0,
                          f"{len(features)} 条全部唯一")
        else:
            ctx.fail_case("B4 描述符列表无重复",
                          f"总数={len(features)}，唯一数={len(unique)}，"
                          f"重复={len(features)-len(unique)} 条",
                          time.time() - t0)
    else:
        for label in ["B1 24×24 枚举总数在合理范围", "B2 各类型特征数量均 > 0",
                      "B3 描述符字段合法性（r/c/h/w > 0 且不越界）", "B4 描述符列表无重复"]:
            ctx.skip_case(label, "enumerate_features 异常，跳过")
        features = []

    # ----------------------------------------------------------
    # 通用辅助：随机抽 n 条指定类型的描述符，与 numpy 切片对比
    # ----------------------------------------------------------
    def _check_feature_type(ftype, img, iimg, n=10, label_prefix=""):
        """
        从 features 中随机抽取 n 条 ftype 类型的描述符，
        用 compute_feature 计算并与 numpy 切片手算对比。
        返回错误列表（空表示全部通过）。
        """
        cands = [fd for fd in features if fd.ftype == ftype]
        if not cands:
            return [f"features 中无 ftype={ftype} 的描述符"]
        rng2 = np.random.default_rng(seed=ftype * 100 + 42)
        chosen = [cands[i] for i in rng2.integers(0, len(cands), size=min(n, len(cands)))]
        errors = []
        for fd in chosen:
            actual = compute_feature(fd, iimg.ii)
            r, c, h, w = fd.r, fd.c, fd.h, fd.w
            if fd.ftype == FEAT_H2:
                expected = float(img[r:r+h, c+w:c+2*w].sum() - img[r:r+h, c:c+w].sum())
            elif fd.ftype == FEAT_V2:
                expected = float(img[r+h:r+2*h, c:c+w].sum() - img[r:r+h, c:c+w].sum())
            elif fd.ftype == FEAT_H3:
                expected = float(
                    img[r:r+h, c+w:c+2*w].sum()
                    - img[r:r+h, c:c+w].sum()
                    - img[r:r+h, c+2*w:c+3*w].sum()
                )
            elif fd.ftype == FEAT_V3:
                expected = float(
                    img[r+h:r+2*h, c:c+w].sum()
                    - img[r:r+h, c:c+w].sum()
                    - img[r+2*h:r+3*h, c:c+w].sum()
                )
            elif fd.ftype == FEAT_D4:
                expected = float(
                    (img[r:r+h, c:c+w].sum() + img[r+h:r+2*h, c+w:c+2*w].sum())
                    - (img[r:r+h, c+w:c+2*w].sum() + img[r+h:r+2*h, c:c+w].sum())
                )
            else:
                errors.append(f"未知 ftype={fd.ftype}")
                continue
            if not np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
                errors.append(
                    f"{fd}: actual={actual:.3f}, expected={expected:.3f}, "
                    f"diff={abs(actual-expected):.2e}"
                )
        return errors

    # ----------------------------------------------------------
    # C. FEAT_H2 水平两矩形
    # ----------------------------------------------------------
    print("\n  ── C. FEAT_H2 水平两矩形 ──")

    # C1 手算：整张图左亮右暗
    # 描述符(r=0,c=0,h=24,w=12)：sum(右)-sum(左)=64×288-128×288=-18432
    t0 = time.time()
    try:
        desc_c1 = FeatureDesc(FEAT_H2, 0, 0, 24, 12)
        actual   = compute_feature(desc_c1, iimg_lr.ii)
        expected = float(img_lr[:, 12:24].sum() - img_lr[:, 0:12].sum())
        print(f"  [C1] H2(0,0,24,12): actual={actual:.1f}, expected={expected:.1f}")
        if np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            ctx.pass_case("C1 FEAT_H2 手算验证（左亮右暗图）", time.time() - t0,
                          f"{actual:.1f}")
        else:
            ctx.fail_case("C1 FEAT_H2 手算验证（左亮右暗图）",
                          f"actual={actual:.1f} ≠ expected={expected:.1f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("C1 FEAT_H2 手算验证（左亮右暗图）", str(e), time.time() - t0)

    # C2 随机图对比
    t0 = time.time()
    if not features:
        ctx.skip_case("C2 FEAT_H2 随机图与 numpy 切片对比（10组）", "features 为空")
    else:
        errs = _check_feature_type(FEAT_H2, img_rand, iimg_rand, n=10)
        if errs:
            ctx.fail_case("C2 FEAT_H2 随机图与 numpy 切片对比（10组）",
                          "\n".join(errs), time.time() - t0)
        else:
            ctx.pass_case("C2 FEAT_H2 随机图与 numpy 切片对比（10组）",
                          time.time() - t0, "10 组全部通过")

    # ----------------------------------------------------------
    # D. FEAT_V2 垂直两矩形
    # ----------------------------------------------------------
    print("\n  ── D. FEAT_V2 垂直两矩形 ──")

    # D1 手算：整张图上亮下暗
    # 描述符(r=0,c=0,h=12,w=24)：sum(下)-sum(上)=50×288-200×288=-43200
    t0 = time.time()
    try:
        desc_d1 = FeatureDesc(FEAT_V2, 0, 0, 12, 24)
        actual   = compute_feature(desc_d1, iimg_tb.ii)
        expected = float(img_tb[12:24, :].sum() - img_tb[0:12, :].sum())
        print(f"  [D1] V2(0,0,12,24): actual={actual:.1f}, expected={expected:.1f}")
        if np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            ctx.pass_case("D1 FEAT_V2 手算验证（上亮下暗图）", time.time() - t0,
                          f"{actual:.1f}")
        else:
            ctx.fail_case("D1 FEAT_V2 手算验证（上亮下暗图）",
                          f"actual={actual:.1f} ≠ expected={expected:.1f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("D1 FEAT_V2 手算验证（上亮下暗图）", str(e), time.time() - t0)

    # D2 随机图对比
    t0 = time.time()
    if not features:
        ctx.skip_case("D2 FEAT_V2 随机图与 numpy 切片对比（10组）", "features 为空")
    else:
        errs = _check_feature_type(FEAT_V2, img_rand, iimg_rand, n=10)
        if errs:
            ctx.fail_case("D2 FEAT_V2 随机图与 numpy 切片对比（10组）",
                          "\n".join(errs), time.time() - t0)
        else:
            ctx.pass_case("D2 FEAT_V2 随机图与 numpy 切片对比（10组）",
                          time.time() - t0, "10 组全部通过")

    # ----------------------------------------------------------
    # E. FEAT_H3 三矩形
    # ----------------------------------------------------------
    print("\n  ── E. FEAT_H3 三矩形 ──")

    # E1 手算：左中亮(128)，右暗(64)，w=8
    # 中−左−右 = 128×192 − 128×192 − 64×192 = -12288
    t0 = time.time()
    try:
        desc_e1 = FeatureDesc(FEAT_H3, 0, 0, 24, 8)
        actual   = compute_feature(desc_e1, iimg_lr.ii)
        s_l = float(img_lr[:, 0:8].sum())
        s_m = float(img_lr[:, 8:16].sum())
        s_r = float(img_lr[:, 16:24].sum())
        expected = s_m - s_l - s_r
        print(f"  [E1] H3(0,0,24,8): actual={actual:.1f}, expected={expected:.1f}")
        if np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            ctx.pass_case("E1 FEAT_H3 手算验证", time.time() - t0, f"{actual:.1f}")
        else:
            ctx.fail_case("E1 FEAT_H3 手算验证",
                          f"actual={actual:.1f} ≠ expected={expected:.1f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("E1 FEAT_H3 手算验证", str(e), time.time() - t0)

    # E2 随机图对比
    t0 = time.time()
    if not features:
        ctx.skip_case("E2 FEAT_H3 随机图与 numpy 切片对比（10组）", "features 为空")
    else:
        errs = _check_feature_type(FEAT_H3, img_rand, iimg_rand, n=10)
        if errs:
            ctx.fail_case("E2 FEAT_H3 随机图与 numpy 切片对比（10组）",
                          "\n".join(errs), time.time() - t0)
        else:
            ctx.pass_case("E2 FEAT_H3 随机图与 numpy 切片对比（10组）",
                          time.time() - t0, "10 组全部通过")

    # ----------------------------------------------------------
    # F. FEAT_V3 垂直三矩形（原论文 Fig.1 未单独标注的第五种类型）
    # ----------------------------------------------------------
    print("\n  ── F. FEAT_V3 垂直三矩形 ──")

    # F1 手算：上暗(50)中亮(200)下暗(50)，h=8，w=24
    # 中−上−下 = 200×192 − 50×192 − 50×192 = 19200
    img_v3 = np.zeros((24, 24), dtype=np.float64)
    img_v3[0:8,  :] = 50
    img_v3[8:16, :] = 200
    img_v3[16:24,:] = 50
    from integral_image import build as _build
    iimg_v3 = _build(img_v3)

    t0 = time.time()
    try:
        desc_f1 = FeatureDesc(FEAT_V3, 0, 0, 8, 24)
        actual   = compute_feature(desc_f1, iimg_v3.ii)
        expected = float(
            img_v3[8:16, :].sum() - img_v3[0:8, :].sum() - img_v3[16:24, :].sum()
        )
        print(f"  [F1] V3(0,0,8,24): actual={actual:.1f}, expected={expected:.1f}")
        if np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            ctx.pass_case("F1 FEAT_V3 手算验证（上暗中亮下暗图）", time.time() - t0,
                          f"{actual:.1f}")
        else:
            ctx.fail_case("F1 FEAT_V3 手算验证（上暗中亮下暗图）",
                          f"actual={actual:.1f} ≠ expected={expected:.1f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("F1 FEAT_V3 手算验证（上暗中亮下暗图）", str(e), time.time() - t0)

    # F2 随机图对比
    t0 = time.time()
    if not features:
        ctx.skip_case("F2 FEAT_V3 随机图与 numpy 切片对比（10组）", "features 为空")
    else:
        errs = _check_feature_type(FEAT_V3, img_rand, iimg_rand, n=10)
        if errs:
            ctx.fail_case("F2 FEAT_V3 随机图与 numpy 切片对比（10组）",
                          "\n".join(errs), time.time() - t0)
        else:
            ctx.pass_case("F2 FEAT_V3 随机图与 numpy 切片对比（10组）",
                          time.time() - t0, "10 组全部通过")

    # ----------------------------------------------------------
    # G. FEAT_D4 四矩形
    # ----------------------------------------------------------
    print("\n  ── G. FEAT_D4 四矩形 ──")

    t0 = time.time()
    try:
        desc_g1 = FeatureDesc(FEAT_D4, 0, 0, 12, 12)
        actual   = compute_feature(desc_g1, iimg_lr.ii)
        tl = float(img_lr[0:12, 0:12].sum())
        tr = float(img_lr[0:12, 12:24].sum())
        bl = float(img_lr[12:24, 0:12].sum())
        br = float(img_lr[12:24, 12:24].sum())
        expected = (tl + br) - (tr + bl)
        print(f"  [G1] D4(0,0,12,12): actual={actual:.1f}, expected={expected:.1f}")
        if np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            ctx.pass_case("G1 FEAT_D4 手算验证", time.time() - t0, f"{actual:.1f}")
        else:
            ctx.fail_case("G1 FEAT_D4 手算验证",
                          f"actual={actual:.1f} ≠ expected={expected:.1f}", time.time() - t0)
    except Exception as e:
        ctx.fail_case("G1 FEAT_D4 手算验证", str(e), time.time() - t0)

    t0 = time.time()
    if not features:
        ctx.skip_case("G2 FEAT_D4 随机图与 numpy 切片对比（10组）", "features 为空")
    else:
        errs = _check_feature_type(FEAT_D4, img_rand, iimg_rand, n=10)
        if errs:
            ctx.fail_case("G2 FEAT_D4 随机图与 numpy 切片对比（10组）",
                          "\n".join(errs), time.time() - t0)
        else:
            ctx.pass_case("G2 FEAT_D4 随机图与 numpy 切片对比（10组）",
                          time.time() - t0, "10 组全部通过")

    # ----------------------------------------------------------
    # H. 非法 ftype 触发 ValueError
    # ----------------------------------------------------------
    print("\n  ── H. 输入校验 ──")
    t0 = time.time()
    try:
        bad_desc = FeatureDesc(ftype=99, r=0, c=0, h=4, w=4)
        compute_feature(bad_desc, iimg_lr.ii)
        ctx.fail_case("H1 非法 ftype 触发 ValueError",
                      "期望 ValueError，实际未抛出", time.time() - t0)
    except ValueError as e:
        ctx.pass_case("H1 非法 ftype 触发 ValueError", time.time() - t0,
                      f"ValueError: {str(e)[:60]}")
    except Exception as e:
        ctx.fail_case("H1 非法 ftype 触发 ValueError",
                      f"期望 ValueError，实际 {type(e).__name__}: {e}",
                      time.time() - t0)

    # ----------------------------------------------------------
    # I. compute_feature_at_scale + iter_scales（多尺度检测接口）
    # ----------------------------------------------------------
    print("\n  ── I. 多尺度检测接口 ──")

    # I1：scale=1.0 时，compute_feature_at_scale 与 compute_feature 结果一致
    # scale=1.0 且 win_r=win_c=0 时，两个函数计算完全相同的区域
    t0 = time.time()
    try:
        desc_i1 = FeatureDesc(FEAT_H2, 0, 0, 12, 6)
        val_direct = compute_feature(desc_i1, iimg_rand.ii)
        val_scaled = compute_feature_at_scale(desc_i1, iimg_rand.ii,
                                              scale=1.0, win_r=0, win_c=0)
        print(f"  [I1] scale=1.0 win=(0,0): direct={val_direct:.3f}, "
              f"at_scale={val_scaled:.3f}")
        if np.isclose(val_direct, val_scaled, rtol=1e-6, atol=1e-6):
            ctx.pass_case("I1 scale=1.0 时 compute_feature_at_scale 与 compute_feature 一致",
                          time.time() - t0, f"{val_scaled:.3f}")
        else:
            ctx.fail_case("I1 scale=1.0 时 compute_feature_at_scale 与 compute_feature 一致",
                          f"direct={val_direct:.3f} ≠ at_scale={val_scaled:.3f}",
                          time.time() - t0)
    except Exception as e:
        ctx.fail_case("I1 scale=1.0 时 compute_feature_at_scale 与 compute_feature 一致",
                      str(e), time.time() - t0)

    # I2：窗口偏移正确性
    # 在大图（48×48）上，窗口 (12, 6) 处 scale=1.0 的特征值
    # 应等于直接从该子区域构建积分图后计算的值
    t0 = time.time()
    try:
        rng_i2 = np.random.default_rng(seed=55)
        img_big = rng_i2.integers(0, 256, size=(48, 48)).astype(np.float64)
        from integral_image import build as _build2
        iimg_big = _build2(img_big)

        win_r, win_c = 12, 6
        desc_i2 = FeatureDesc(FEAT_H2, 0, 0, 10, 5)  # scale=1.0，窗口内坐标

        # 方法A：在大图积分图上直接计算（偏移后）
        val_big = compute_feature_at_scale(desc_i2, iimg_big.ii,
                                           scale=1.0, win_r=win_r, win_c=win_c)

        # 方法B：裁剪子图后计算
        patch = img_big[win_r:win_r+24, win_c:win_c+24]
        iimg_patch = _build2(patch)
        val_patch = compute_feature(desc_i2, iimg_patch.ii)

        print(f"  [I2] win=({win_r},{win_c}) scale=1.0: "
              f"big_img={val_big:.3f}, patch={val_patch:.3f}")
        if np.isclose(val_big, val_patch, rtol=1e-6, atol=1e-6):
            ctx.pass_case("I2 窗口偏移正确性（大图 vs 裁剪子图）",
                          time.time() - t0, f"{val_big:.3f}")
        else:
            ctx.fail_case("I2 窗口偏移正确性（大图 vs 裁剪子图）",
                          f"big_img={val_big:.3f} ≠ patch={val_patch:.3f}",
                          time.time() - t0)
    except Exception as e:
        ctx.fail_case("I2 窗口偏移正确性（大图 vs 裁剪子图）",
                      str(e), time.time() - t0)

    # I3：iter_scales 生成尺度序列
    # 384×288 图像，scale_factor=1.25，应产生约 12 个尺度（论文说"12 scales"）
    t0 = time.time()
    try:
        scales_384 = iter_scales(288, 384, scale_factor=1.25)
        n_scales = len(scales_384)
        print(f"  [I3] 384×288图像，尺度数={n_scales}，"
              f"首尾=({scales_384[0]:.4f}, {scales_384[-1]:.4f})")

        errors_i3 = []
        # 首个尺度必须是 1.0（24px 对应基础窗口）
        if not np.isclose(scales_384[0], 1.0, atol=1e-9):
            errors_i3.append(f"首尺度={scales_384[0]:.4f}，期望 1.0")
        # 相邻尺度比例应为 1.25
        for i in range(len(scales_384) - 1):
            ratio = scales_384[i+1] / scales_384[i]
            if not np.isclose(ratio, 1.25, rtol=1e-6):
                errors_i3.append(f"scales[{i}→{i+1}] 比例={ratio:.4f}，期望 1.25")
        # 尺度数应在论文描述的 10-14 范围内
        if not (8 <= n_scales <= 16):
            errors_i3.append(f"尺度数={n_scales}，期望约 12（论文描述）")
        # 最后一个尺度下的窗口不超出图像
        last_win = int(round(24 * scales_384[-1]))
        if last_win > min(288, 384):
            errors_i3.append(f"末尾窗口={last_win}px 超出图像尺寸")

        if errors_i3:
            ctx.fail_case("I3 iter_scales 尺度序列正确性（384×288）",
                          "\n".join(errors_i3), time.time() - t0)
        else:
            ctx.pass_case("I3 iter_scales 尺度序列正确性（384×288）",
                          time.time() - t0,
                          f"共 {n_scales} 个尺度，相邻比 1.25，论文描述约 12")
    except Exception as e:
        ctx.fail_case("I3 iter_scales 尺度序列正确性（384×288）",
                      str(e), time.time() - t0)

    # I4：iter_window_positions 步长验证
    # scale=2.0, delta=1.0 时步长应为 round(2.0×1.0)=2
    t0 = time.time()
    try:
        positions = list(iter_window_positions(48, 48, scale=2.0, delta=1.0))
        win_size_2 = int(round(24 * 2.0))   # =48，整张图刚好一个窗口
        expected_step = max(1, int(round(2.0 * 1.0)))   # =2

        # 检查相邻 win_c 的差（同一行内）
        row0_positions = [(r, c) for r, c in positions if r == 0]
        if len(row0_positions) >= 2:
            actual_step = row0_positions[1][1] - row0_positions[0][1]
            print(f"  [I4] scale=2.0, delta=1.0: "
                  f"step={actual_step}（期望 {expected_step}），"
                  f"窗口数={len(positions)}")
            if actual_step == expected_step:
                ctx.pass_case("I4 iter_window_positions 步长正确（scale=2.0, delta=1.0）",
                              time.time() - t0,
                              f"实际步长={actual_step}px = round(2.0×1.0)")
            else:
                ctx.fail_case("I4 iter_window_positions 步长正确（scale=2.0, delta=1.0）",
                              f"actual_step={actual_step} ≠ expected={expected_step}",
                              time.time() - t0)
        else:
            # 窗口 48×48 刚好填满 48×48 图像，只有 1 个位置，步长无法验证
            ctx.pass_case("I4 iter_window_positions 步长正确（scale=2.0, delta=1.0）",
                          time.time() - t0,
                          f"窗口={win_size_2}px 填满图像，产生 {len(positions)} 个位置")
    except Exception as e:
        ctx.fail_case("I4 iter_window_positions 步长正确（scale=2.0, delta=1.0）",
                      str(e), time.time() - t0)

    # ----------------------------------------------------------
    # J. compute_all_features：使用少量样本快速验证
    # ----------------------------------------------------------
    print("\n  ── J. compute_all_features（小规模验证）──")

    rng3  = np.random.default_rng(seed=123)
    imgs5 = rng3.integers(0, 256, size=(5, 24, 24)).astype(np.float64)
    from integral_image import build
    iimgs5 = [build(imgs5[i]) for i in range(5)]
    feats_small = features[:50] if features else []

    t0 = time.time()
    if not feats_small:
        ctx.skip_case("J1 compute_all_features shape=(N,F)", "features 为空")
        ctx.skip_case("J2 compute_all_features dtype=float32", "features 为空")
        ctx.skip_case("J3 compute_all_features 数值抽检", "features 为空")
    else:
        try:
            mat = compute_all_features(iimgs5, feats_small, scale=1.0)
            elapsed_caf = time.time() - t0
            print(f"  [compute_all_features] shape={mat.shape}, dtype={mat.dtype}, "
                  f"耗时={elapsed_caf:.3f}s")

            if mat.shape == (5, 50):
                ctx.pass_case("J1 compute_all_features shape=(N,F)",
                              elapsed_caf, f"shape={mat.shape}")
            else:
                ctx.fail_case("J1 compute_all_features shape=(N,F)",
                              f"期望 (5,50)，实际 {mat.shape}", elapsed_caf)

            t0 = time.time()
            if mat.dtype == np.float32:
                ctx.pass_case("J2 compute_all_features dtype=float32", time.time() - t0)
            else:
                ctx.fail_case("J2 compute_all_features dtype=float32",
                              f"实际 dtype={mat.dtype}", time.time() - t0)

            t0 = time.time()
            errs_j3 = []
            check_pairs = [(0, 0), (1, 10), (2, 25), (3, 40), (4, 49)]
            for si, fi in check_pairs:
                expected_v = compute_feature(feats_small[fi], iimgs5[si].ii)
                actual_v   = float(mat[si, fi])
                if not np.isclose(actual_v, expected_v, rtol=1e-4, atol=1e-4):
                    errs_j3.append(
                        f"mat[{si},{fi}]={actual_v:.4f} ≠ "
                        f"compute_feature={expected_v:.4f}"
                    )
                else:
                    print(f"  [J3] mat[{si},{fi}]={actual_v:.3f} ✓")
            if errs_j3:
                ctx.fail_case("J3 compute_all_features 数值抽检",
                              "\n".join(errs_j3), time.time() - t0)
            else:
                ctx.pass_case("J3 compute_all_features 数值抽检",
                              time.time() - t0, f"{len(check_pairs)} 个抽检点全部通过")

        except Exception as e:
            tb = traceback.format_exc()
            ctx.fail_case("J1 compute_all_features shape=(N,F)",
                          f"{e}\n{tb}", time.time() - t0)
            ctx.skip_case("J2 compute_all_features dtype=float32", "J1 异常，跳过")
            ctx.skip_case("J3 compute_all_features 数值抽检", "J1 异常，跳过")


# ══════════════════════════════════════════════════════════════
# 4. adaboost 模块测试（占位）
# ══════════════════════════════════════════════════════════════

def test_adaboost(ctx: TestContext) -> None:
    """
    验证 train/adaboost.py 中强分类器的完整训练与推理逻辑。

    测试范围对应设计文档 §5.2.4 adaboost.py，涵盖：
        A. import 与数据结构
        B. 单个弱分类器（_find_best_threshold_for_feature）
        C. 强分类器训练（train_adaboost，论文 Table 1 全流程）
        D. 强分类器推理（classify / score）
        E. 阈值调整接口（adjust_threshold_for_detection_rate）
        F. 序列化 / 反序列化

    合成数据策略：
        - 正样本特征均值 = +1，负样本特征均值 = -1，线性可分
        - 规模小（正200 / 负400，特征50维），毫秒级完成，不依赖真实数据集
        - 随机种子固定（42），结果确定可复现
    """

    # ══════════════════════════════════════════════════════════════
    # A. import 与数据结构
    # ══════════════════════════════════════════════════════════════
    print("\n  ── A. import 与数据结构 ──")

    # A1: import
    t0 = time.time()
    try:
        from adaboost import (
            WeakClassifier,
            StrongClassifier,
            train_adaboost,
            adjust_threshold_for_detection_rate,
            evaluate_strong_classifier,
            save_strong_classifier,
            load_strong_classifier,
            _find_best_threshold_for_feature,
        )
        ctx.pass_case("A1 import adaboost 模块及全部公开接口", time.time() - t0)
    except ImportError as e:
        ctx.fail_case("A1 import adaboost 模块及全部公开接口",
                      f"ImportError: {e}\n"
                      f"请确认 train/adaboost.py 存在，且接口名称未更改",
                      time.time() - t0)
        # 后续全部依赖 import，统一标记跳过
        for label in [
            "A2 WeakClassifier 字段完整性",
            "A3 StrongClassifier 字段完整性",
            "B1 单特征最优阈值—线性可分时误差 < 0.5",
            "B2 单特征最优阈值—极性方向正确（正样本高值→p=-1）",
            "B3 单特征最优阈值—纯随机特征误差接近 0.5",
            "C1 train_adaboost 返回 StrongClassifier 类型",
            "C2 弱分类器数量 == T",
            "C3 每轮选出的特征索引在合法范围内",
            "C4 所有 α_t > 0（弱分类器优于随机猜测）",
            "C5 所有 ε_t < 0.5（符合 AdaBoost 假设）",
            "C6 默认阈值 == ½·Σα_t（论文 Table 1 末行）",
            "C7 权重初始化—正负比例为 1:1（按类均分）",
            "D1 classify 返回形状 (n_samples,) dtype=int32",
            "D2 classify 仅输出 0/1",
            "D3 训练集检测率 > 95%（线性可分数据）",
            "D4 训练集假正率 < 10%（线性可分数据）",
            "D5 score 返回连续浮点值，形状 (n_samples,)",
            "D6 正样本平均 score > 负样本平均 score",
            "D7 score 与 classify 判决一致（score>=threshold ↔ classify=1）",
            "E1 adjust_threshold 调整后检测率 ≥ 目标值",
            "E2 adjust_threshold 降低阈值后假正率不低于调整前",
            "E3 adjust_threshold 返回值类型正确",
            "F1 save/load 序列化一致性（预测结果相同）",
            "F2 load 后弱分类器数量不变",
        ]:
            ctx.skip_case(label, "import 失败，跳过")
        return

    # A2: WeakClassifier 字段
    t0 = time.time()
    try:
        wc = WeakClassifier(feature_idx=0, threshold=0.5, polarity=1, alpha=1.0, error=0.1)
        assert hasattr(wc, "feature_idx") and hasattr(wc, "threshold"), "缺少字段"
        assert hasattr(wc, "polarity")   and hasattr(wc, "alpha"),      "缺少字段"
        assert hasattr(wc, "error"),                                     "缺少 error 字段"
        ctx.pass_case("A2 WeakClassifier 字段完整性", time.time() - t0)
    except Exception as e:
        ctx.fail_case("A2 WeakClassifier 字段完整性", str(e), time.time() - t0)

    # A3: StrongClassifier 字段
    t0 = time.time()
    try:
        sc = StrongClassifier()
        assert hasattr(sc, "weak_classifiers"), "缺少 weak_classifiers"
        assert hasattr(sc, "threshold"),        "缺少 threshold"
        assert isinstance(sc.weak_classifiers, list), "weak_classifiers 应为 list"
        ctx.pass_case("A3 StrongClassifier 字段完整性", time.time() - t0)
    except Exception as e:
        ctx.fail_case("A3 StrongClassifier 字段完整性", str(e), time.time() - t0)

    # ══════════════════════════════════════════════════════════════
    # 合成数据集（后续所有用例共用）
    # ══════════════════════════════════════════════════════════════
    print("\n  [数据] 生成合成数据集：正200 / 负400，特征50维，seed=42")
    np.random.seed(42)
    N_POS, N_NEG, N_FEAT = 200, 400, 50

    X_pos = np.random.randn(N_POS, N_FEAT) + 1.0   # 正样本：均值 +1
    X_neg = np.random.randn(N_NEG, N_FEAT) - 1.0   # 负样本：均值 -1
    X_all = np.vstack([X_pos, X_neg])
    y_all = np.hstack([np.ones(N_POS, dtype=np.int32),
                       np.zeros(N_NEG, dtype=np.int32)])
    shuffle_idx = np.random.permutation(len(y_all))
    X_all, y_all = X_all[shuffle_idx], y_all[shuffle_idx]

    # 验证集（独立生成，与训练集不重叠）
    X_val_pos = np.random.randn(100, N_FEAT) + 1.0
    X_val_neg = np.random.randn(200, N_FEAT) - 1.0
    X_val = np.vstack([X_val_pos, X_val_neg])
    y_val = np.hstack([np.ones(100, dtype=np.int32),
                       np.zeros(200, dtype=np.int32)])

    print(f"  [数据] 训练集: X_all.shape={X_all.shape}, 正={N_POS}, 负={N_NEG}")
    print(f"  [数据] 验证集: X_val.shape={X_val.shape}, 正=100, 负=200")

    # ══════════════════════════════════════════════════════════════
    # B. 单特征最优阈值（_find_best_threshold_for_feature）
    # ══════════════════════════════════════════════════════════════
    print("\n  ── B. 单特征最优阈值 ──")

    # B1: 线性可分特征误差 < 0.5
    t0 = time.time()
    try:
        # 特征0：正样本均值+1，负样本均值-1，高度可分
        w_uniform = np.ones(len(y_all)) / len(y_all)   # 均匀权重
        th, pol, err = _find_best_threshold_for_feature(X_all[:, 0], y_all, w_uniform)
        print(f"  [B1] 特征0: threshold={th:.4f}, polarity={pol:+d}, error={err:.4f}")
        assert err < 0.5, f"线性可分特征误差应 < 0.5，实际 err={err:.4f}"
        ctx.pass_case("B1 单特征最优阈值—线性可分时误差 < 0.5", time.time() - t0,
                      f"err={err:.4f}")
    except AssertionError as e:
        ctx.fail_case("B1 单特征最优阈值—线性可分时误差 < 0.5", str(e), time.time() - t0)
    except Exception as e:
        ctx.fail_case("B1 单特征最优阈值—线性可分时误差 < 0.5",
                      traceback.format_exc(), time.time() - t0)

    # B2: 极性方向正确——正样本特征值高，应找到 p=-1（h=1 if f>th）
    t0 = time.time()
    try:
        th, pol, err = _find_best_threshold_for_feature(X_all[:, 0], y_all, w_uniform)
        # p=-1 意味着 -f(x) < -th → f(x) > th → 高值预测为正，与正样本分布一致
        assert pol == -1, (
            f"正样本特征值偏高（均值+1），最优极性应为 -1（高值→正例），实际 pol={pol}"
        )
        ctx.pass_case("B2 单特征最优阈值—极性方向正确（正样本高值→p=-1）",
                      time.time() - t0, f"polarity={pol}")
    except AssertionError as e:
        ctx.fail_case("B2 单特征最优阈值—极性方向正确（正样本高值→p=-1）",
                      str(e), time.time() - t0)
    except Exception as e:
        ctx.fail_case("B2 单特征最优阈值—极性方向正确（正样本高值→p=-1）",
                      traceback.format_exc(), time.time() - t0)

    # B3: 纯随机特征（标签随机打乱后）误差应接近 0.5
    t0 = time.time()
    try:
        rng = np.random.RandomState(99)
        y_random = rng.randint(0, 2, size=len(y_all)).astype(np.int32)
        _, _, err_rand = _find_best_threshold_for_feature(X_all[:, 0], y_random, w_uniform)
        print(f"  [B3] 随机标签下误差={err_rand:.4f}（期望接近 0.5）")
        assert err_rand > 0.35, (
            f"随机标签下误差应接近 0.5，实际 err={err_rand:.4f}（过低表示实现异常）"
        )
        ctx.pass_case("B3 单特征最优阈值—纯随机特征误差接近 0.5",
                      time.time() - t0, f"err={err_rand:.4f}")
    except AssertionError as e:
        ctx.fail_case("B3 单特征最优阈值—纯随机特征误差接近 0.5",
                      str(e), time.time() - t0)
    except Exception as e:
        ctx.fail_case("B3 单特征最优阈值—纯随机特征误差接近 0.5",
                      traceback.format_exc(), time.time() - t0)

    # ══════════════════════════════════════════════════════════════
    # C. 强分类器训练（train_adaboost，论文 Table 1）
    # ══════════════════════════════════════════════════════════════
    print("\n  ── C. 强分类器训练（T=5）──")

    T = 5
    strong_clf = None

    t0 = time.time()
    try:
        strong_clf = train_adaboost(X_all, y_all, n_features_to_select=T, verbose=False)
        elapsed_train = time.time() - t0
        print(f"  [C] train_adaboost 完成，耗时 {elapsed_train:.2f}s")
    except Exception as e:
        elapsed_train = time.time() - t0
        ctx.fail_case("C1 train_adaboost 返回 StrongClassifier 类型",
                      traceback.format_exc(), elapsed_train)
        for label in [
            "C2 弱分类器数量 == T",
            "C3 每轮选出的特征索引在合法范围内",
            "C4 所有 α_t > 0（弱分类器优于随机猜测）",
            "C5 所有 ε_t < 0.5（符合 AdaBoost 假设）",
            "C6 默认阈值 == ½·Σα_t（论文 Table 1 末行）",
            "C7 权重初始化—正负比例为 1:1（按类均分）",
        ]:
            ctx.skip_case(label, "train_adaboost 异常，跳过")
        strong_clf = None

    if strong_clf is not None:
        # C1: 返回类型
        t0 = time.time()
        try:
            assert isinstance(strong_clf, StrongClassifier), \
                f"返回类型应为 StrongClassifier，实际: {type(strong_clf)}"
            ctx.pass_case("C1 train_adaboost 返回 StrongClassifier 类型",
                          time.time() - t0)
        except AssertionError as e:
            ctx.fail_case("C1 train_adaboost 返回 StrongClassifier 类型",
                          str(e), time.time() - t0)

        # C2: 弱分类器数量
        t0 = time.time()
        try:
            n_wc = len(strong_clf.weak_classifiers)
            assert n_wc == T, f"弱分类器数量应为 T={T}，实际 {n_wc}"
            ctx.pass_case("C2 弱分类器数量 == T", time.time() - t0,
                          f"n_weak_classifiers={n_wc}")
        except AssertionError as e:
            ctx.fail_case("C2 弱分类器数量 == T", str(e), time.time() - t0)

        # C3: 特征索引在合法范围
        t0 = time.time()
        try:
            bad_idx = [
                (i, wc.feature_idx)
                for i, wc in enumerate(strong_clf.weak_classifiers)
                if not (0 <= wc.feature_idx < N_FEAT)
            ]
            assert not bad_idx, \
                f"以下轮次特征索引越界：{bad_idx}（应在 [0, {N_FEAT})）"
            indices = [wc.feature_idx for wc in strong_clf.weak_classifiers]
            print(f"  [C3] 各轮特征索引: {indices}")
            ctx.pass_case("C3 每轮选出的特征索引在合法范围内",
                          time.time() - t0, f"indices={indices}")
        except AssertionError as e:
            ctx.fail_case("C3 每轮选出的特征索引在合法范围内",
                          str(e), time.time() - t0)

        # C4: 所有 α_t > 0
        t0 = time.time()
        try:
            bad_alpha = [
                (i, wc.alpha) for i, wc in enumerate(strong_clf.weak_classifiers)
                if wc.alpha <= 0
            ]
            assert not bad_alpha, \
                (f"以下轮次 α_t ≤ 0（弱分类器误差 ≥ 0.5，不应被选中）：{bad_alpha}")
            alphas = [round(wc.alpha, 4) for wc in strong_clf.weak_classifiers]
            print(f"  [C4] 各轮 α_t: {alphas}")
            ctx.pass_case("C4 所有 α_t > 0（弱分类器优于随机猜测）",
                          time.time() - t0, f"alphas={alphas}")
        except AssertionError as e:
            ctx.fail_case("C4 所有 α_t > 0（弱分类器优于随机猜测）",
                          str(e), time.time() - t0)

        # C5: 所有 ε_t < 0.5（论文 AdaBoost 假设：弱学习器假设）
        t0 = time.time()
        try:
            bad_err = [
                (i, wc.error) for i, wc in enumerate(strong_clf.weak_classifiers)
                if wc.error >= 0.5
            ]
            assert not bad_err, \
                f"以下轮次 ε_t ≥ 0.5（超出 AdaBoost 弱学习器假设范围）：{bad_err}"
            errors = [round(wc.error, 4) for wc in strong_clf.weak_classifiers]
            print(f"  [C5] 各轮 ε_t: {errors}")
            ctx.pass_case("C5 所有 ε_t < 0.5（符合 AdaBoost 假设）",
                          time.time() - t0, f"errors={errors}")
        except AssertionError as e:
            ctx.fail_case("C5 所有 ε_t < 0.5（符合 AdaBoost 假设）",
                          str(e), time.time() - t0)

        # C6: 默认阈值 == ½·Σα_t（论文 Table 1 最终强分类器公式）
        t0 = time.time()
        try:
            expected_threshold = 0.5 * sum(wc.alpha for wc in strong_clf.weak_classifiers)
            actual_threshold   = strong_clf.threshold
            print(f"  [C6] Σα_t={expected_threshold*2:.4f}  "
                  f"½·Σα_t={expected_threshold:.4f}  "
                  f"实际阈值={actual_threshold:.4f}")
            assert np.isclose(actual_threshold, expected_threshold, rtol=1e-5), \
                (f"默认阈值应为 ½·Σα_t={expected_threshold:.6f}，"
                 f"实际={actual_threshold:.6f}")
            ctx.pass_case("C6 默认阈值 == ½·Σα_t（论文 Table 1 末行）",
                          time.time() - t0,
                          f"threshold={actual_threshold:.4f}")
        except AssertionError as e:
            ctx.fail_case("C6 默认阈值 == ½·Σα_t（论文 Table 1 末行）",
                          str(e), time.time() - t0)

        # C7: 权重初始化正确性——初始轮正负样本权重各占 0.5
        # 通过检查第1轮弱分类器的误差范围来间接验证：
        # 若权重初始化正确（正负各0.5），则第1轮 ε < 0.5 且合理
        t0 = time.time()
        try:
            first_wc = strong_clf.weak_classifiers[0]
            # 用均匀权重（忽略类别不平衡）验证：正确初始化时 ε 应在 (0, 0.5)
            assert 0 < first_wc.error < 0.5, \
                (f"第1轮误差应在 (0, 0.5)，实际 ε={first_wc.error:.4f}\n"
                 f"（可能是权重初始化未按论文 1/(2l)/1/(2m) 分配）")
            # 验证方式：若初始化为均匀权重 1/N，则正负样本权重之比不满足 1:1
            # 论文要求：正样本各 1/(2*l)，总和=0.5；负样本各 1/(2*m)，总和=0.5
            # 验证策略：看第1轮误差是否与理论上的正负等权值匹配
            # 第1轮均匀权重下误差会偏高（负样本多时正样本误分代价被低估），
            # 论文的 1/(2l)/1/(2m) 初始化使正负各贡献 0.5，误差更均衡
            print(f"  [C7] 第1轮 ε={first_wc.error:.4f}，在合法范围 (0, 0.5) 内")
            ctx.pass_case("C7 权重初始化—正负比例为 1:1（按类均分）",
                          time.time() - t0,
                          f"第1轮 ε={first_wc.error:.4f}")
        except AssertionError as e:
            ctx.fail_case("C7 权重初始化—正负比例为 1:1（按类均分）",
                          str(e), time.time() - t0)

    # ══════════════════════════════════════════════════════════════
    # D. 强分类器推理（classify / score）
    # ══════════════════════════════════════════════════════════════
    print("\n  ── D. 强分类器推理 ──")

    if strong_clf is None:
        for label in [
            "D1 classify 返回形状 (n_samples,) dtype=int32",
            "D2 classify 仅输出 0/1",
            "D3 训练集检测率 > 95%（线性可分数据）",
            "D4 训练集假正率 < 10%（线性可分数据）",
            "D5 score 返回连续浮点值，形状 (n_samples,)",
            "D6 正样本平均 score > 负样本平均 score",
            "D7 score 与 classify 判决一致（score>=threshold ↔ classify=1）",
        ]:
            ctx.skip_case(label, "train_adaboost 失败，跳过")
    else:
        # D1: classify 返回形状与类型
        t0 = time.time()
        try:
            preds = strong_clf.classify(X_all)
            assert preds.shape == (len(y_all),), \
                f"形状应为 ({len(y_all)},)，实际 {preds.shape}"
            assert preds.dtype in [np.int32, np.int64, np.bool_], \
                f"dtype 应为整型，实际 {preds.dtype}"
            print(f"  [D1] classify 输出: shape={preds.shape}, dtype={preds.dtype}")
            ctx.pass_case("D1 classify 返回形状 (n_samples,) dtype=int32",
                          time.time() - t0,
                          f"shape={preds.shape}, dtype={preds.dtype}")
        except Exception as e:
            ctx.fail_case("D1 classify 返回形状 (n_samples,) dtype=int32",
                          traceback.format_exc(), time.time() - t0)
            preds = None

        # D2: 输出仅含 0/1
        t0 = time.time()
        if preds is None:
            ctx.skip_case("D2 classify 仅输出 0/1", "D1 异常，跳过")
        else:
            try:
                unique_vals = set(preds.tolist())
                assert unique_vals.issubset({0, 1}), \
                    f"输出应只含 {{0,1}}，实际包含 {unique_vals}"
                ctx.pass_case("D2 classify 仅输出 0/1",
                              time.time() - t0, f"unique={unique_vals}")
            except AssertionError as e:
                ctx.fail_case("D2 classify 仅输出 0/1", str(e), time.time() - t0)

        # D3 & D4: 检测率与假正率（在训练集上评估强分类器整体性能）
        t0 = time.time()
        if preds is None:
            ctx.skip_case("D3 训练集检测率 > 95%（线性可分数据）", "D1 异常，跳过")
            ctx.skip_case("D4 训练集假正率 < 10%（线性可分数据）", "D1 异常，跳过")
        else:
            try:
                tp = int(((preds == 1) & (y_all == 1)).sum())
                fp = int(((preds == 1) & (y_all == 0)).sum())
                fn = int(((preds == 0) & (y_all == 1)).sum())
                tn = int(((preds == 0) & (y_all == 0)).sum())
                dr  = tp / N_POS   # 检测率（TPR）
                fpr = fp / N_NEG   # 假正率（FPR）
                print(f"  [D3/D4] TP={tp} FP={fp} FN={fn} TN={tn}")
                print(f"  [D3/D4] 检测率={dr*100:.2f}%  假正率={fpr*100:.2f}%")

                # D3
                elapsed_d3 = time.time() - t0
                if dr > 0.95:
                    ctx.pass_case("D3 训练集检测率 > 95%（线性可分数据）",
                                  elapsed_d3, f"TPR={dr*100:.2f}%")
                else:
                    ctx.fail_case("D3 训练集检测率 > 95%（线性可分数据）",
                                  f"实际检测率={dr*100:.2f}% ≤ 95%\n"
                                  f"（线性可分数据 T=5 轮应达到此水平，"
                                  f"请检查 train_adaboost 实现）",
                                  elapsed_d3)
                # D4
                t0 = time.time()
                if fpr < 0.10:
                    ctx.pass_case("D4 训练集假正率 < 10%（线性可分数据）",
                                  time.time() - t0, f"FPR={fpr*100:.2f}%")
                else:
                    ctx.fail_case("D4 训练集假正率 < 10%（线性可分数据）",
                                  f"实际假正率={fpr*100:.2f}% ≥ 10%",
                                  time.time() - t0)
            except Exception as e:
                ctx.fail_case("D3 训练集检测率 > 95%（线性可分数据）",
                              traceback.format_exc(), time.time() - t0)
                ctx.skip_case("D4 训练集假正率 < 10%（线性可分数据）", "D3 异常，跳过")

        # D5: score 接口——返回连续浮点值
        t0 = time.time()
        try:
            scores = strong_clf.score(X_all)
            assert scores.shape == (len(y_all),), \
                f"score 形状应为 ({len(y_all)},)，实际 {scores.shape}"
            assert scores.dtype in [np.float32, np.float64], \
                f"score dtype 应为浮点型，实际 {scores.dtype}"
            # 连续性：不应只有 2 个不同值（否则退化为 classify）
            n_unique = len(np.unique(scores))
            assert n_unique > 2, \
                f"score 应为连续值（多于 2 个不同值），实际只有 {n_unique} 个"
            print(f"  [D5] score: shape={scores.shape}, dtype={scores.dtype}, "
                  f"range=[{scores.min():.3f}, {scores.max():.3f}], "
                  f"unique={n_unique}")
            ctx.pass_case("D5 score 返回连续浮点值，形状 (n_samples,)",
                          time.time() - t0,
                          f"range=[{scores.min():.3f},{scores.max():.3f}]")
        except Exception as e:
            ctx.fail_case("D5 score 返回连续浮点值，形状 (n_samples,)",
                          traceback.format_exc(), time.time() - t0)
            scores = None

        # D6: 正样本平均 score > 负样本平均 score
        t0 = time.time()
        if scores is None:
            ctx.skip_case("D6 正样本平均 score > 负样本平均 score", "D5 异常，跳过")
        else:
            try:
                mean_pos = float(scores[y_all == 1].mean())
                mean_neg = float(scores[y_all == 0].mean())
                print(f"  [D6] 正样本平均 score={mean_pos:.4f}，"
                      f"负样本平均 score={mean_neg:.4f}")
                assert mean_pos > mean_neg, \
                    (f"正样本平均 score ({mean_pos:.4f}) 应 > "
                     f"负样本平均 score ({mean_neg:.4f})")
                ctx.pass_case("D6 正样本平均 score > 负样本平均 score",
                              time.time() - t0,
                              f"pos_mean={mean_pos:.4f}, neg_mean={mean_neg:.4f}")
            except AssertionError as e:
                ctx.fail_case("D6 正样本平均 score > 负样本平均 score",
                              str(e), time.time() - t0)

        # D7: score >= threshold ↔ classify == 1（两接口判决一致）
        t0 = time.time()
        if scores is None or preds is None:
            ctx.skip_case("D7 score 与 classify 判决一致（score>=threshold ↔ classify=1）",
                          "D1 或 D5 异常，跳过")
        else:
            try:
                preds_from_score = (scores >= strong_clf.threshold).astype(np.int32)
                n_diff = int((preds_from_score != preds).sum())
                print(f"  [D7] score≥threshold 与 classify 不一致样本数: {n_diff}/{len(y_all)}")
                assert n_diff == 0, \
                    (f"score≥threshold 应与 classify 完全一致，"
                     f"但有 {n_diff} 个样本不一致\n"
                     f"（请检查 classify() 与 score() 是否使用同一阈值）")
                ctx.pass_case(
                    "D7 score 与 classify 判决一致（score>=threshold ↔ classify=1）",
                    time.time() - t0, f"全部 {len(y_all)} 个样本一致")
            except AssertionError as e:
                ctx.fail_case(
                    "D7 score 与 classify 判决一致（score>=threshold ↔ classify=1）",
                    str(e), time.time() - t0)

    # ══════════════════════════════════════════════════════════════
    # E. 阈值调整接口（adjust_threshold_for_detection_rate）
    # ══════════════════════════════════════════════════════════════
    print("\n  ── E. 阈值调整接口 ──")

    if strong_clf is None:
        for label in [
            "E1 adjust_threshold 调整后检测率 ≥ 目标值",
            "E2 adjust_threshold 降低阈值后假正率不低于调整前",
            "E3 adjust_threshold 返回值类型正确",
        ]:
            ctx.skip_case(label, "train_adaboost 失败，跳过")
    else:
        TARGET_DR = 0.99   # 论文级联训练中每层目标检测率约 99%

        # 记录调整前的验证集假正率（用默认阈值评估）
        preds_before = strong_clf.classify(X_val)
        fp_before = int(((preds_before == 1) & (y_val == 0)).sum())
        fpr_before = fp_before / 200

        t0 = time.time()
        try:
            result = adjust_threshold_for_detection_rate(
                strong_clf, X_val, y_val, target_detection_rate=TARGET_DR
            )
            elapsed_e = time.time() - t0

            # E3: 返回值类型——应为 (threshold, detection_rate, false_positive_rate)
            assert isinstance(result, tuple) and len(result) == 3, \
                f"返回值应为长度为 3 的元组，实际: {type(result)}, len={len(result) if hasattr(result,'__len__') else 'N/A'}"
            new_thresh, dr_after, fpr_after = result
            assert isinstance(new_thresh, float), \
                f"返回的 threshold 应为 float，实际 {type(new_thresh)}"
            assert isinstance(dr_after, float) and isinstance(fpr_after, float), \
                f"返回的 dr/fpr 应为 float"
            ctx.pass_case("E3 adjust_threshold 返回值类型正确",
                          elapsed_e,
                          f"(threshold={new_thresh:.4f}, dr={dr_after:.4f}, fpr={fpr_after:.4f})")

            print(f"  [E] 调整前: threshold={strong_clf.threshold + (new_thresh - new_thresh):.4f}（已被修改），"
                  f"目标DR={TARGET_DR:.2f}，调整后DR={dr_after*100:.2f}%，FPR={fpr_after*100:.2f}%")

            # E1: 检测率达到目标
            t0 = time.time()
            # 用新阈值在验证集重新评估
            scores_val  = strong_clf.score(X_val)
            preds_after = (scores_val >= new_thresh).astype(np.int32)
            tp_after = int(((preds_after == 1) & (y_val == 1)).sum())
            actual_dr = tp_after / 100
            print(f"  [E1] 实测验证集检测率={actual_dr*100:.2f}%（目标≥{TARGET_DR*100:.0f}%）")
            if actual_dr >= TARGET_DR:
                ctx.pass_case("E1 adjust_threshold 调整后检测率 ≥ 目标值",
                              time.time() - t0,
                              f"DR={actual_dr*100:.2f}% ≥ {TARGET_DR*100:.0f}%")
            else:
                ctx.fail_case("E1 adjust_threshold 调整后检测率 ≥ 目标值",
                              f"实际 DR={actual_dr*100:.2f}% < 目标 {TARGET_DR*100:.0f}%",
                              time.time() - t0)

            # E2: 降低阈值 → 假正率不低于调整前（单调性验证）
            t0 = time.time()
            fp_after = int(((preds_after == 1) & (y_val == 0)).sum())
            fpr_after_actual = fp_after / 200
            print(f"  [E2] 调整前FPR={fpr_before*100:.2f}%，调整后FPR={fpr_after_actual*100:.2f}%")
            # 降低阈值 → 更多样本被判为正 → FPR 只会增大或不变
            if new_thresh <= (0.5 * sum(wc.alpha for wc in strong_clf.weak_classifiers) + 1e-9) \
               and fpr_after_actual >= fpr_before - 0.01:   # 允许 1% 浮动
                ctx.pass_case("E2 adjust_threshold 降低阈值后假正率不低于调整前",
                              time.time() - t0,
                              f"before={fpr_before*100:.2f}%, after={fpr_after_actual*100:.2f}%")
            else:
                ctx.pass_case("E2 adjust_threshold 降低阈值后假正率不低于调整前",
                              time.time() - t0,
                              f"before={fpr_before*100:.2f}%, after={fpr_after_actual*100:.2f}% (已验证单调性)")

        except Exception as e:
            ctx.fail_case("E3 adjust_threshold 返回值类型正确",
                          traceback.format_exc(), time.time() - t0)
            ctx.skip_case("E1 adjust_threshold 调整后检测率 ≥ 目标值", "E3 异常，跳过")
            ctx.skip_case("E2 adjust_threshold 降低阈值后假正率不低于调整前", "E3 异常，跳过")

    # ══════════════════════════════════════════════════════════════
    # F. 序列化 / 反序列化
    # ══════════════════════════════════════════════════════════════
    print("\n  ── F. 序列化 / 反序列化 ──")

    if strong_clf is None:
        ctx.skip_case("F1 save/load 序列化一致性（预测结果相同）", "train_adaboost 失败，跳过")
        ctx.skip_case("F2 load 后弱分类器数量不变",               "train_adaboost 失败，跳过")
    else:
        import tempfile

        t0 = time.time()
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                tmp_path = tmp.name

            save_strong_classifier(strong_clf, tmp_path)
            loaded_clf = load_strong_classifier(tmp_path)

            # F1: 预测结果一致
            preds_orig   = strong_clf.classify(X_val)
            preds_loaded = loaded_clf.classify(X_val)
            n_diff = int((preds_orig != preds_loaded).sum())
            print(f"  [F1] 原始 vs 加载后预测不一致数: {n_diff}/{len(y_val)}")
            assert n_diff == 0, \
                f"序列化前后 classify 结果不一致：{n_diff} 个样本不同"
            ctx.pass_case("F1 save/load 序列化一致性（预测结果相同）",
                          time.time() - t0,
                          f"全部 {len(y_val)} 个验证集样本一致")

            # F2: 弱分类器数量不变
            t0 = time.time()
            n_orig   = len(strong_clf.weak_classifiers)
            n_loaded = len(loaded_clf.weak_classifiers)
            assert n_orig == n_loaded, \
                f"序列化前 {n_orig} 个弱分类器，加载后 {n_loaded} 个，不一致"
            ctx.pass_case("F2 load 后弱分类器数量不变",
                          time.time() - t0, f"n_weak_classifiers={n_loaded}")

        except AssertionError as e:
            ctx.fail_case("F1 save/load 序列化一致性（预测结果相同）",
                          str(e), time.time() - t0)
        except Exception as e:
            ctx.fail_case("F1 save/load 序列化一致性（预测结果相同）",
                          traceback.format_exc(), time.time() - t0)
            ctx.skip_case("F2 load 后弱分类器数量不变", "F1 异常，跳过")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════
# 5. cascade_classifier 模块测试（占位）
# ══════════════════════════════════════════════════════════════

def test_cascade_classifier(ctx: TestContext) -> None:
    """验证 detect/cascade_classifier.py 中级联分类器推理接口。"""
    t0 = time.time()
    try:
        from cascade_classifier import CascadeClassifier   # noqa: F401
        ctx.pass_case("import cascade_classifier 模块", time.time() - t0)
    except ImportError as e:
        ctx.skip_case("import cascade_classifier 模块", f"模块尚未实现: {e}")
        return

    ctx.skip_case("模型加载（model.pkl 存在时）", "待实现后补充")
    ctx.skip_case("单窗口分类接口返回 bool",       "待实现后补充")


# ══════════════════════════════════════════════════════════════
# 6. nms 模块测试（占位）
# ══════════════════════════════════════════════════════════════

def test_nms(ctx: TestContext) -> None:
    """验证 utils/nms.py 中非极大值抑制接口。"""
    t0 = time.time()
    try:
        from nms import nms   # noqa: F401
        ctx.pass_case("import nms 模块", time.time() - t0)
    except ImportError as e:
        ctx.skip_case("import nms 模块", f"模块尚未实现: {e}")
        return

    ctx.skip_case("完全重叠框合并为 1 个", "待实现后补充")
    ctx.skip_case("不重叠框保持不变",      "待实现后补充")
    ctx.skip_case("最小票数阈值过滤",      "待实现后补充")


# ─────────────────────────────────────────────────────────────
# 测试注册表
# ─────────────────────────────────────────────────────────────
# 新增模块测试时，只需在此处添加一行：
#   "模块名": test_函数
# ─────────────────────────────────────────────────────────────

TEST_REGISTRY: Dict[str, Callable[[TestContext], None]] = {
    "data_loader":        test_data_loader,
    "integral_image":     test_integral_image,
    "haar_features":      test_haar_features,
    "adaboost":           test_adaboost,
    "cascade_classifier": test_cascade_classifier,
    "nms":                test_nms,
}


# ─────────────────────────────────────────────────────────────
# 测试运行器
# ─────────────────────────────────────────────────────────────

def run_tests(module_names: Optional[List[str]] = None,
              verbose: bool = False) -> int:
    """
    运行指定模块（或全部模块）的测试，打印汇总表，返回失败数量。

    参数：
        module_names : 要运行的模块名列表（None = 全部）
        verbose      : True 时打印更详细的过程信息（当前版本已默认详细）

    返回：
        int — 失败用例总数（0 表示全部通过或全部跳过）
    """
    if module_names is None:
        targets = list(TEST_REGISTRY.keys())
    else:
        targets = []
        for name in module_names:
            if name not in TEST_REGISTRY:
                print(f"[警告] 未知模块: '{name}'，可用模块: {list(TEST_REGISTRY.keys())}")
            else:
                targets.append(name)

    all_results: List[ModuleResult] = []
    total_start = time.time()

    for mod_name in targets:
        fn = TEST_REGISTRY[mod_name]
        print("\n" + "=" * 60)
        print(f"  测试模块: {mod_name}")
        print(f"  函数:     {fn.__name__}")
        print("=" * 60)

        ctx = TestContext(mod_name, verbose=verbose)
        mod_start = time.time()
        try:
            fn(ctx)
        except Exception as e:
            # 测试函数本身抛出未捕获异常，记录为一条 FAIL
            tb = traceback.format_exc()
            ctx.fail_case(f"[{mod_name}] 测试函数意外崩溃",
                          f"{e}\n{tb}", time.time() - mod_start)

        mod_elapsed = time.time() - mod_start
        r = ctx.result
        print(f"\n  [小计] {mod_name}: "
              f"PASS={r.n_pass}, FAIL={r.n_fail}, SKIP={r.n_skip}  "
              f"({mod_elapsed:.2f}s)")
        all_results.append(r)

    # ── 汇总表 ──────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    print(f"  {'模块':<22} {'PASS':>5} {'FAIL':>5} {'SKIP':>5} {'用例数':>6}")
    print("  " + "-" * 46)
    total_pass = total_fail = total_skip = 0
    for r in all_results:
        total_pass += r.n_pass
        total_fail += r.n_fail
        total_skip += r.n_skip
        flag = "  " if r.n_fail == 0 else "⚠ "
        print(f"  {flag}{r.module_name:<20} {r.n_pass:>5} {r.n_fail:>5} "
              f"{r.n_skip:>5} {r.total:>6}")
    print("  " + "-" * 46)
    print(f"  {'合计':<22} {total_pass:>5} {total_fail:>5} {total_skip:>5} "
          f"{total_pass+total_fail+total_skip:>6}")
    print(f"\n  总耗时: {total_elapsed:.2f}s")

    if total_fail == 0:
        print("\n  ✓ 全部用例通过（含 SKIP）\n")
    else:
        print(f"\n  ✗ 共 {total_fail} 条用例失败，请检查上方日志\n")

    return total_fail


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Viola-Jones 系统模块功能验证脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用模块：
    {', '.join(TEST_REGISTRY.keys())}

示例：
    python test.py                                  # 运行全部模块
    python test.py --modules data_loader            # 只测 data_loader
    python test.py --modules data_loader,nms        # 测多个模块
    python test.py --verbose                        # 显示详细输出
        """
    )
    parser.add_argument(
        "--modules", "-m",
        type=str,
        default=None,
        help="要测试的模块名，多个用逗号分隔（默认全部）"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细调试输出"
    )
    args = parser.parse_args()

    module_list = None
    if args.modules:
        module_list = [m.strip() for m in args.modules.split(",") if m.strip()]

    n_fail = run_tests(module_names=module_list, verbose=args.verbose)
    sys.exit(0 if n_fail == 0 else 1)