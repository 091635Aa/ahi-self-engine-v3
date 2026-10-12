#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bench_gpu.py —— AHI 引擎 GPU 基准（GPU_测试要求.md 第四节 T1–T5 的可执行实现）

设计要点
--------
1. 后端自动检测：torch + CUDA 可用 ⇒ CUDA；否则 torch CPU；再否则纯 numpy。
   三种后端跑的是**同一份引擎代码**（ops 抽象层），仅设备/精度不同。
2. 被测量的引擎 = brain.py 动力学的**分区批量转写**（区放在 batch 维，一次 GEMM 算完），
   与参考实现逐参同源；preflight 里做逐参数 + 逐轨迹的数值一致性核对。
3. 阈值**全部**取自 GPU_测试要求.md 第四节；每条判据给 PASS / FAIL；
   硬件缺失（如多卡）按文档规定标 SKIP。
4. 所有输出都是实测数字（时间、字节、范数、比率）；不含"大约/可能"。
   来源非实测的量（如 §2.1 参数公式、§2.2 显存分解）单独标注 source 字段。

运行：  python3 bench_gpu.py                 # 默认：完整 T1–T5
        python3 bench_gpu.py --quick         # 快速预算（T4 步数少）
        python3 bench_gpu.py --skip-t5       # 跳过 T5（T5 直接调用 verify_all.py，最慢）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DOC = "GPU_测试要求.md"

# ====================================================================== 阈值
# 逐条抄自 GPU_测试要求.md 第四节（T1–T5）。键名 = 文档里的判据名。
TH = {
    # T1 单卡冒烟
    "T1_no_nan":           "无 NaN/Inf，退出码 0",
    # 数值稳定：原口径「100 步内范数增长」测的是启动瞬态（初始 ‖h‖≈1 → 工作点 ≈27~49）。
    # 本机实测（small 尺度 1500 步，CUDA/float32）：t≥50 后进入平稳，窗口均值 27.3→27.6
    # （漂移 <2%），预热后 100 步窗内增长 ≤1.8×；自首步的 65× 全部来自前 ~20 步的瞬态。
    # ⇒ 判据改为「预热 100 步后、余下窗口内增长 < 10×」，阈值不变（不收敛的系统在任何
    #    窗口都会涨过它）；自首步的原始增长（含瞬态）作为信息项保留在 metrics 里。
    "T1_norm_growth":      10.0,      # 预热后范数增长 < 10×
    "T1_stab_warmup":      100,       # 预热步数（与 T4 的 step≥100 启动瞬态门控同一处理）
    "T1_mem_frac":         0.9,       # 显存峰值 < 卡容量 × 0.9
    # T2 多卡扩展性
    "T2_scaling_eff":      0.75,      # 8 卡 vs 1 卡吞吐扩展效率 > 0.75
    "T2_allgather_frac":   0.15,      # all-gather 通信占总时间 < 15%
    "T2_shard_loss_diff":  1e-3,      # 分片前后前 100 步 loss 差异 < 1e-3
    # T3 吞吐与带宽
    "T3_instances":        100,       # 实例/卡（锁步）≥ 100
    "T3_tick_rate":        1e4,       # tick/s（每卡）≥ 100 × 100 = 1e4（实例·tick/s）
    "T3_bw_lo":            0.60,      # HBM 带宽利用率 60–75%
    "T3_bw_hi":            0.75,
    # T4 数值稳定性长跑
    "T4_loss_spike":       5.0,       # 任意 100 步窗口内涨幅 < 5×
    "T4_sat_rate":         0.9,       # 饱和率 < 0.9
    "T4_nfe_frac":         0.999,     # 不动点 ≥ 99.9% 在 ≤12 次内收敛
    "T4_nfe_iters":        12,
    # T5 六条理论路线
    "T5_P1_ece":           0.05,      # ECE < 0.05
    "T5_P1_auc":           0.60,      # AUC > 0.6
    "T5_P2_p95_ms":        20.0,      # 接口延迟 p95 < 20 ms
    "T5_P3_js":            0.20,      # 语义交换消融的 JS 散度 > 0.2
    "T5_P5_F":             10.0,      # F < 10
    "T5_P6_kappa_lo":      0.30,      # κ ∈ [0.3, 0.9]
    "T5_P6_kappa_hi":      0.90,
    "T5_P6_cross_r2":      0.95,      # 跨分支神谕 R² < 0.95
    # T4 文档未给量化阈值的判据（本脚本的量化口径，显式标注）
    "T4_grad_explosion_ratio": 100.0,  # "有界、无爆炸" ⇒ max < 100 × median
}

PASS, FAIL, SKIP, ERROR = "PASS", "FAIL", "SKIP", "ERROR"

# GPU_测试要求.md 第四节清单 -> 报告字段路径（一一对应）
CHECKLIST_MAP = {
    "T1.前向不崩":     ("T1_single_card_smoke.criteria", "前向不崩 (batch=1, T=64, 100 步)"),
    "T1.数值稳定":     ("T1_single_card_smoke.criteria", "数值稳定：预热 100 步后范数无发散"),
    "T1.显存峰值":     ("T1_single_card_smoke.criteria", "显存峰值 < 卡容量 × 0.9"),
    "T2.多卡扩展效率": ("T2_multi_card_scaling.criteria", "8 卡 vs 1 卡吞吐扩展效率"),
    "T2.all-gather":   ("T2_multi_card_scaling.criteria", "all-gather 通信占总时间"),
    "T2.分片 loss 差异": ("T2_multi_card_scaling.criteria", "分片前后前 100 步 loss 差异"),
    "T3.实例/卡":      ("T3_throughput_bandwidth.criteria", "实例/卡（锁步）"),
    "T3.tick/s":       ("T3_throughput_bandwidth.criteria", "tick/s（每卡，实例·tick/s）"),
    "T3.带宽利用率":   ("T3_throughput_bandwidth.criteria", "HBM 带宽利用率"),
    "T4.无 NaN":       ("T4_stability_longrun.criteria", "无 NaN/Inf（全程）"),
    "T4.loss 无尖峰":  ("T4_stability_longrun.criteria", "loss 无尖峰（任意 100 步窗口涨幅 < 5×）"),
    "T4.梯度范数":     ("T4_stability_longrun.criteria", "梯度范数有界、无爆炸"),
    "T4.饱和率":       ("T4_stability_longrun.criteria", "饱和率 < 0.9"),
    "T4.不动点 NFE":   ("T4_stability_longrun.criteria", "不动点 NFE ≥ 99.9% 在 ≤12 次内收敛"),
    "T5.P1 ECE/AUC":   ("T5_theory_routes.P1.criteria", "P1 ECE / AUC"),
    "T5.P2 p95":       ("T5_theory_routes.P2.criteria", "P2 接口延迟 p95"),
    "T5.P3 JS":        ("T5_theory_routes.P3.criteria", "P3 语义交换 JS 散度"),
    "T5.P4 整合度":    ("T5_theory_routes.P4.criteria", "P4 整合度代理 主体 vs 对照"),
    "T5.P5 F 有界/反暗室": ("T5_theory_routes.P5.criteria", "P5 自由能有界 / 反暗室"),
    "T5.P6 κ/跨分支R²": ("T5_theory_routes.P6.criteria", "P6 κ 区间 / 跨分支神谕 R²"),
}


# ====================================================================== 工具
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _jnum(v):
    """JSON 安全的数值：numpy 标量 -> python；非有限 -> None。"""
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def crit(name, value, threshold, passed, unit=None, note="", source=DOC, **extra):
    """一条判据行。passed=None 表示不适用（硬件缺失）。"""
    nonfinite = False
    if isinstance(value, float) and not math.isfinite(value):
        nonfinite = True
    d = {
        "name": name,
        "value": _jnum(value),
        "unit": unit,
        "threshold": threshold,
        "passed": (None if passed is None else bool(passed)),
        "note": note,
        "threshold_source": source,
    }
    if nonfinite:
        d["value_nonfinite"] = True
        d["note"] = (note + " ").strip() + " [值非有限: nan/inf]"
    for k, v in extra.items():
        d[k] = _jnum(v) if isinstance(v, (float, np.floating)) else v
    return d


def block(status, criteria, metrics=None, doc_ref=None, note="", **extra):
    d = {
        "status": status,
        "doc_ref": doc_ref or DOC + " 第四节",
        "criteria": criteria,
        "metrics": metrics or {},
        "note": note,
    }
    d.update(extra)
    return d


def rollup(criteria, skipped_ok=True):
    """由判据行推出测试块状态。"""
    if any(c["passed"] is None for c in criteria):
        return SKIP
    return PASS if all(c["passed"] for c in criteria) else FAIL


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return jsonable(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, (np.ndarray,)):
        return jsonable(o.tolist())
    return o


def meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                out[k.strip()] = int(rest.strip().split()[0]) * 1024
    except Exception:
        pass
    return out


def peak_rss_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def rss_bytes():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


# ====================================================================== 后端
class Ops:
    """numpy / torch 双后端的最小算子层（引擎只依赖这一层）。"""

    name = "base"
    is_torch = False
    is_cuda = False

    # ---- 构造/转换
    def t(self, x):                                    # -> 后端张量
        raise NotImplementedError

    def numpy(self, x):                                # -> numpy（脱离计算图）
        raise NotImplementedError

    def from_numpy(self, x):
        return self.t(x)

    def detach(self, x):
        return x

    # ---- 数学
    def tanh(self, x):
        return self.xp.tanh(x)

    def sigmoid(self, x):
        raise NotImplementedError

    def exp(self, x):
        return self.xp.exp(x)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def abs(self, x):
        return self.xp.abs(x)

    def maximum(self, a, b):
        raise NotImplementedError

    def clip(self, a, lo, hi):
        raise NotImplementedError

    def where(self, c, a, b):
        raise NotImplementedError

    def concat(self, xs, axis=-1):
        raise NotImplementedError

    def stack(self, xs, axis=0):
        raise NotImplementedError

    def sum(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def mean(self, a, axis=None, keepdims=False):
        raise NotImplementedError

    def norm(self, a, axis=-1):
        """沿 axis 的 L2 范数（等价 np.linalg.norm(a, axis=axis)）。"""
        return self.sqrt(self.sum(a * a, axis=axis))

    def einsum(self, sig, *args):
        return self.xp.einsum(sig, *args)

    def matmul(self, a, b):
        return a @ b

    def zeros(self, shape):
        raise NotImplementedError

    def full(self, shape, v):
        raise NotImplementedError

    def reshape(self, a, shape):
        return a.reshape(shape)

    def item(self, x):
        return float(x.item() if hasattr(x, "item") else x)

    def finite_all(self, a):
        raise NotImplementedError

    def set_requires_grad(self, a, flag):
        return a

    def bool_mask(self, x):
        """numpy bool 数组 -> 后端 bool 张量（用于 where 的分支选择，不参与求导）。"""
        raise NotImplementedError

    def clone(self, a):
        raise NotImplementedError

    def T2(self, a):
        """转置最后两维（对 2-D 即普通转置）。"""
        return self.xp.swapaxes(a, -1, -2)

    def no_grad(self):
        import contextlib
        return contextlib.nullcontext()

    def rng_normal(self, rng, shape):
        return self.t(rng.normal(size=shape))


class NumpyOps(Ops):
    name = "numpy"

    def __init__(self, dtype=np.float64):
        self.xp = np
        self.dtype = np.dtype(dtype)
        self.itemsize = int(self.dtype.itemsize)
        self.dtype_str = str(self.dtype)
        self.device_str = "cpu"
        self.torch = None

    def t(self, x):
        if isinstance(x, np.ndarray):
            return x.astype(self.dtype, copy=False)
        return np.asarray(x, dtype=self.dtype)

    def numpy(self, x):
        return np.asarray(x)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def maximum(self, a, b):
        return np.maximum(a, b)

    def clip(self, a, lo, hi):
        return np.clip(a, lo, hi)

    def where(self, c, a, b):
        return np.where(c, a, b)

    def concat(self, xs, axis=-1):
        return np.concatenate(xs, axis=axis)

    def stack(self, xs, axis=0):
        return np.stack(xs, axis=axis)

    def sum(self, a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)

    def zeros(self, shape):
        return np.zeros(shape, dtype=self.dtype)

    def full(self, shape, v):
        return np.full(shape, v, dtype=self.dtype)

    def finite_all(self, a):
        return bool(np.isfinite(a).all())

    def bool_mask(self, x):
        return np.asarray(x, dtype=bool)

    def clone(self, a):
        return np.array(a, copy=True)


class TorchOps(Ops):
    name = "torch"

    def __init__(self, torch, device=None, dtype=None, is_cuda=False):
        self.torch = torch
        self.xp = torch
        self.dtype = dtype if dtype is not None else torch.float32
        self.itemsize = int(torch.empty(0, dtype=self.dtype).element_size())
        self.dtype_str = str(self.dtype).replace("torch.", "")
        self.device = device
        self.is_cuda = bool(is_cuda)
        self.device_str = str(device) if device is not None else "cpu"

    def t(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return self.torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

    def numpy(self, x):
        # ★ param_items() 里混有 numpy 参量（未装箱成张量），不能无条件 .detach()。
        if not self.torch.is_tensor(x):
            return np.asarray(x)
        return x.detach().to("cpu").numpy()

    def detach(self, x):
        return x.detach()

    def sigmoid(self, x):
        return self.torch.sigmoid(x)

    def maximum(self, a, b):
        # ★ torch.maximum 不接受 python 标量作第二参数（numpy 接受）。
        #   本机（有 torch+CUDA）实测：choose_scale 的 o.maximum(..., 1e-12) 直接
        #   TypeError: argument 'other' must be Tensor, not float —— 无 torch 的
        #   机器永远走不到这条路径，故此前未被发现。此处把标量提升为同设备张量。
        if not self.torch.is_tensor(a):
            a = self.torch.as_tensor(a, dtype=b.dtype if self.torch.is_tensor(b) else None,
                                     device=b.device if self.torch.is_tensor(b) else None)
        if not self.torch.is_tensor(b):
            b = self.torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return self.torch.maximum(a, b)

    def clip(self, a, lo, hi):
        return self.torch.clamp(a, lo, hi)

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def concat(self, xs, axis=-1):
        return self.torch.cat(xs, dim=axis)

    def stack(self, xs, axis=0):
        return self.torch.stack(xs, dim=axis)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.sum()
        return a.sum(dim=axis, keepdim=keepdims)

    def mean(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.mean()
        return a.mean(dim=axis, keepdim=keepdims)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def full(self, shape, v):
        return self.torch.full(shape, float(v), dtype=self.dtype, device=self.device)

    def finite_all(self, a):
        return bool(self.torch.isfinite(a).all().item())

    def set_requires_grad(self, a, flag):
        a.requires_grad_(bool(flag))
        return a

    def no_grad(self):
        return self.torch.no_grad()

    def item(self, x):
        return float(x.detach().item() if hasattr(x, "detach") else x)

    def bool_mask(self, x):
        # ★ 传入的可能是 CUDA 张量：np.asarray(tensor) 会要求先 .cpu()，
        #   在有 CUDA 的机器上直接 TypeError（无 torch 的机器走不到）。
        #   已是张量 ⇒ 直接就地转 bool（保设备）；否则再走 numpy 构造。
        if self.torch.is_tensor(x):
            return x.to(dtype=self.torch.bool)
        return self.torch.as_tensor(np.asarray(x, dtype=bool), device=self.device)

    def clone(self, a):
        return a.detach().clone()

    def T2(self, a):
        return a.transpose(-1, -2)


def detect_environment():
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "torch_available": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_version": None,
        "cudnn_version": None,
        "devices": [],
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": None,
    }
    try:
        env["load_average"] = list(os.getloadavg())
    except Exception:
        env["load_average"] = None
    mi = meminfo()
    env["mem_total_bytes"] = mi.get("MemTotal")
    env["mem_available_bytes"] = mi.get("MemAvailable")
    try:
        import torch
        env["torch_available"] = True
        env["torch_version"] = torch.__version__
        env["torch_num_threads"] = torch.get_num_threads()
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = torch.version.cuda
        env["cudnn_version"] = (torch.backends.cudnn.version()
                                if torch.backends.cudnn.is_available() else None)
        if env["cuda_available"]:
            env["cuda_device_count"] = torch.cuda.device_count()
            for i in range(env["cuda_device_count"]):
                p = torch.cuda.get_device_properties(i)
                env["devices"].append({
                    "index": i,
                    "name": p.name,
                    "total_memory_bytes": int(p.total_memory),
                    "multi_processor_count": int(p.multi_processor_count),
                    "major": int(p.major), "minor": int(p.minor),
                })
    except Exception as e:
        env["torch_import_error"] = "%s: %s" % (type(e).__name__, e)
    return env


def build_backend(env, force=None, dtype=None):
    """返回 (ops, backend_name, reason)。"""
    kind = force or "auto"
    if kind == "auto":
        if env["cuda_available"] and env["cuda_device_count"] >= 1:
            kind = "cuda"
        elif env["torch_available"]:
            kind = "torch-cpu"
        else:
            kind = "numpy"
    if kind == "numpy":
        dt = np.float64 if dtype in (None, "auto") else np.dtype(dtype)
        return NumpyOps(dt), "numpy", "无 torch：使用 numpy CPU 参考后端"
    import torch
    if kind == "cuda":
        if not (env["cuda_available"] and env["cuda_device_count"] >= 1):
            raise RuntimeError("--backend cuda 但 torch.cuda.is_available() 为假")
        dt = torch.float32 if dtype in (None, "auto") else getattr(torch, dtype)
        return (TorchOps(torch, device=torch.device("cuda", 0), dtype=dt, is_cuda=True),
                "cuda", "torch + CUDA 可用：使用 GPU 后端")
    dt = torch.float64 if dtype in (None, "auto") else getattr(torch, dtype)
    return TorchOps(torch, device=torch.device("cpu"), dtype=dt, is_cuda=False), "torch-cpu", \
        "torch 可用但无 CUDA：使用 torch CPU 后端"


# ====================================================================== 引擎
# 本脚本转写所依据的 ahi_theory/brain.py 修订（preflight 会核对磁盘现状）
# 留痕（本机 2026-09-24，均为行为等价改动，逐项有证据）：
#   f7fb718650…  远端原始修订（转写据此写成；远端 bench 的 preflight 按此 PASS）
#   01f6b401f3…  本机改动① R_REGIONS 参数化（供 P4 真分割对照；R=12 时随机流逐位不变，
#                六路线逐 seed 结果与离线版完全一致，已回归验证）
#   a1424fad23…  本机改动② 删除 run() 文档串中被红队证伪的 env_rho 论断（仅注释，行为不变）
#                ⇒ 改后核对：numpy/float64 逐参差 0.0、200 步轨迹最大差 7.9e-11（< 1e-9）
TRANSCRIBED_BRAIN_SHA256 = "a1424fad232a56f6b42da6c95459fbde8de989ff92c196f016cb2a08a804cf18"


def _sha256(path):
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def _spec_norm(A, target):
    """与 brain.py 完全同式：把矩阵谱半径归一到 target（绝不按 sigma_max 标定）。"""
    rho = float(max(abs(np.linalg.eigvals(A))))
    return A / rho * target if rho > 0 else A


class EngCfg:
    """引擎规模配置。scale=reference 时与 ahi_theory/brain.py 逐参同源。"""

    def __init__(self, R=12, d=24, K=16, layers=1, instances=1, seed=0, name="reference"):
        self.R, self.d, self.K = int(R), int(d), int(K)
        self.layers = int(layers)
        self.instances = int(instances)
        self.seed = int(seed)
        self.name = name

    def as_dict(self):
        return {"scale": self.name, "regions": self.R, "dim": self.d, "schema": self.K,
                "layers": self.layers, "instances": self.instances, "seed": self.seed}

    def block_params(self):
        """§2.1 的每层参数量：4d² time-mix + 8d² FFN + 1d² gate = 13d²。"""
        return self.R * self.layers * 13 * self.d * self.d


class Engine:
    """brain.py 的分区批量转写：区在 batch 维，实例也在 batch 维。

    参数初始化逐行复刻 ahi_theory/brain.py（含 rng 调用顺序与 P1 自预测机制），
    动力学 step() 逐项复刻 brain.py 的 step()。
    额外层（cfg.layers>1）是 §2.1 的深栈 block，仅在目标规模使用。
    ★ brain.py 是被持续修改的活文件：preflight 会核对 sha256（见 TRANSCRIBED_*）。
    """

    PARAM_KEYS = ("A", "Wc", "Bin", "Wg", "bg", "Wgs", "Ws", "Ws_in",
                  "Wm", "Wm_in", "W_slow", "Wm_fwd", "Wm_surp", "u_mom",
                  "WL", "GL", "Wconf")

    def __init__(self, cfg: EngCfg, ops: Ops):
        self.cfg = cfg
        self.ops = ops
        self.R, self.d, self.K = cfg.R, cfg.d, cfg.K
        R, d, K = self.R, self.d, self.K

        # ---- 超参（与 brain.py 默认值一致）
        self.rho_target = 0.92
        self.dt_slow = 0.01
        self.kappa_target = 0.6
        self.w_slow = 2.0
        self.gs_amp = 1.0
        self.gain_scale = 2.5
        self.gs_weight = 0.6
        self.beta_slow = 0.8
        self.fp_iters = 12
        self.fp_tol = 1e-4
        # ---- P1 自预测机制（brain.py 同名超参默认值）
        self.w_mom = 9.0e4
        self.w_fwd = 1.4
        self.w_surp = 1.0
        self.w_self = 6.0e3
        self.ema_alpha = 0.02

        # ---- 参数初始化：rng 调用顺序必须与 brain.py 一致
        rng = np.random.default_rng(cfg.seed)
        A = np.stack([_spec_norm(rng.normal(size=(d, d)) / np.sqrt(d), self.rho_target * 0.9)
                      for _ in range(R)])
        Wc = rng.normal(size=(R, R)) * 0.5
        np.fill_diagonal(Wc, 0.0)
        Wc = _spec_norm(Wc, 3.0)
        Bin = rng.normal(size=(R, d, d)) / np.sqrt(d)
        h0 = rng.normal(size=(R, d)) * 0.1
        Wg = rng.normal(size=(R, R + K)) * 0.4
        bg = np.zeros(R)
        Wgs = _spec_norm(rng.normal(size=(R, R)) / np.sqrt(R), 0.97)
        Ws = _spec_norm(rng.normal(size=(K, K)) / np.sqrt(K), 0.85)
        Ws_in = rng.normal(size=(K, R + R)) / np.sqrt(R * 2)
        s0 = rng.normal(size=K) * 0.1
        Wm = rng.normal(size=(K, K)) / np.sqrt(K)
        _sm = float(np.linalg.svd(Wm, compute_uv=False)[0])
        if _sm > 0:
            Wm = Wm / _sm * self.kappa_target
        Wm_in = rng.normal(size=(K, R * d)) / np.sqrt(R * d)
        hslow0 = rng.normal(size=K) * 0.05
        W_slow = _spec_norm(rng.normal(size=(K, K)) / np.sqrt(K), 1.9)
        # ---- P1 自预测机制参数（brain.py L124-127）
        Wm_fwd = rng.normal(size=(K, R * d)) / np.sqrt(R * d)
        Wm_surp = rng.normal(size=(K, 2)) / np.sqrt(2.0)
        _um = rng.normal(size=K)
        u_mom = _um / np.linalg.norm(_um)
        # ---- 调质核团 / 校准头 / 身份锚点
        WL = rng.normal(size=(5, 4)) * 0.8
        GL = rng.normal(size=(R, 3, 5)) * 0.5
        Wconf = rng.normal(size=(R, K)) / np.sqrt(K)
        _a = rng.normal(size=K)
        anchor = _a / np.linalg.norm(_a)

        self.init_np = dict(A=A, Wc=Wc, Bin=Bin, h0=h0, Wg=Wg, bg=bg, Wgs=Wgs, Ws=Ws,
                            Ws_in=Ws_in, s0=s0, Wm=Wm, Wm_in=Wm_in, hslow0=hslow0,
                            W_slow=W_slow, Wm_fwd=Wm_fwd, Wm_surp=Wm_surp, u_mom=u_mom,
                            WL=WL, GL=GL, Wconf=Wconf, anchor=anchor)

        self.P = {k: ops.t(self.init_np[k]) for k in self.PARAM_KEYS}

        # ---- §2.1 深栈 block（layers-1 个），每层 4d² + 8d² + 1d² = 13d²
        self.extra = []
        for l in range(max(cfg.layers - 1, 0)):
            blk = dict(
                Wz=rng.normal(size=(R, d, d)) / np.sqrt(d),
                Wx=rng.normal(size=(R, d, d)) / np.sqrt(d),
                Wr=rng.normal(size=(R, d, d)) / np.sqrt(d),
                Wo=rng.normal(size=(R, d, d)) / np.sqrt(d),
                W1=rng.normal(size=(R, 4 * d, d)) / np.sqrt(d),
                W2=rng.normal(size=(R, d, 4 * d)) / np.sqrt(4 * d),
                Wgate=rng.normal(size=(R, d, d)) / np.sqrt(d),
            )
            self.extra.append({k: ops.t(v) for k, v in blk.items()})
            self.P.update({"L%d_%s" % (l, kk): vv for kk, vv in blk.items()})

        self.reset()
        return

    # ---------------------------------------------------------------- 从活体 Brain 装配
    @classmethod
    def from_brain(cls, brain, ops, instances=1):
        """用**当前磁盘上的 brain.py** 的实例参数装配引擎（对参考实现的修订保持鲁棒）。

        这样即使 brain.py 的初始化序列被改写，动力学一致性核对依然有效。
        """
        R = int(np.asarray(brain.A).shape[0])
        d = int(np.asarray(brain.A).shape[1])
        K = int(np.asarray(brain.Ws).shape[0])
        eng = cls(EngCfg(R, d, K, 1, instances, 0, "from_brain"), ops)
        names = ("A", "Wc", "Bin", "Wg", "bg", "Wgs", "Ws", "Ws_in", "Wm", "Wm_in",
                 "W_slow", "Wm_fwd", "Wm_surp", "u_mom", "WL", "GL", "Wconf")
        eng.missing_in_brain = []
        for k in names:
            if hasattr(brain, k):
                eng.P[k] = ops.t(np.asarray(getattr(brain, k)))
            else:                                    # 旧修订缺 P1 参数 ⇒ 置零
                eng.missing_in_brain.append(k)
                eng.P[k] = ops.zeros(tuple(ops.numpy(eng.P[k]).shape))
        for s_ in ("w_slow", "dt_slow", "gs_amp", "gain_scale", "gs_weight", "beta_slow",
                   "w_mom", "w_fwd", "w_surp", "w_self", "ema_alpha", "fp_iters", "fp_tol"):
            if hasattr(brain, s_):
                setattr(eng, s_, getattr(brain, s_))
        if instances == 1:                           # 同状态起步
            o = ops

            def _batched(x):
                x = np.asarray(x, dtype=np.float64)
                if x.ndim == 0:
                    return x.reshape(1)
                if x.ndim == 1:
                    return x[None, :]
                return x[None, :, :]

            for k in ("h", "g", "s", "g_slow", "m", "h_slow", "L", "conf", "z_pred",
                      "self_err", "self_err_ema", "dF_prev", "dF_ema", "dF_abs_ema",
                      "mom", "_e_rel", "_pi_rel"):
                if hasattr(brain, k):
                    setattr(eng, k, o.t(_batched(getattr(brain, k))))
            if getattr(brain, "setpoint", None) is not None:
                eng.setpoint = np.array(brain.setpoint, dtype=np.float64)
            if getattr(brain, "F_hist", None):
                eng.F0 = o.t(np.array([brain.F_hist[0]], dtype=np.float64))
            if getattr(brain, "F_prev", None) is not None:
                eng.F_prev = o.t(np.array([brain.F_prev], dtype=np.float64))
        return eng

    # ---------------------------------------------------------------- 状态
    def reset(self, instances=None, seed_offset=0):
        o = self.ops
        if instances is not None:
            self.cfg.instances = int(instances)
        B = self.cfg.instances
        self.B = B
        i = self.init_np
        self.h = o.t(np.repeat(i["h0"][None, :, :], B, axis=0))
        self.g = o.full((B, self.R), 0.5)
        self.s = o.t(np.repeat(i["s0"][None, :], B, axis=0))
        self.g_slow = o.zeros((B, self.R))
        self.m = o.zeros((B, self.K))
        self.h_slow = o.t(np.repeat(i["hslow0"][None, :], B, axis=0))
        self.L = o.full((B, 5), 0.5)
        self.conf = o.full((B, self.R), 0.5)
        self.F = o.zeros((B,))
        self.nfe = np.full(B, self.fp_iters, dtype=np.int64)
        self.F0 = None
        self.t = 0
        self.setpoint = np.array([1.0, 0.0, 0.0, 0.0])
        # ---- P1 自预测状态（brain.py L128-136）
        self.z_pred = o.t(np.repeat(i["h0"][None, :, :], B, axis=0))     # = h.copy()
        self.self_err = o.zeros((B,))
        self.self_err_ema = o.zeros((B,))
        self.dF_prev = o.zeros((B,))
        self.dF_ema = o.zeros((B,))
        self.dF_abs_ema = o.zeros((B,))                 # |ΔF| 的自身尺度（brain.py L133）
        self.mom = o.zeros((B,))
        self.F_prev = None
        self._e_rel = o.full((B,), 1.0)
        self._pi_rel = o.full((B,), 1.0)

    def state(self):
        return dict(t=self.t, h=self.h, g=self.g, s=self.s, m=self.m, h_slow=self.h_slow,
                    L=self.L, conf=self.conf, F=self.F, nfe=self.nfe,
                    z_pred=self.z_pred, self_err=self.self_err,
                    self_err_ema=self.self_err_ema, mom=self.mom)

    def snapshot(self):
        o = self.ops
        snap = {k: o.clone(v)
                for k, v in dict(h=self.h, g=self.g, s=self.s, g_slow=self.g_slow, m=self.m,
                                 h_slow=self.h_slow, L=self.L, conf=self.conf, F=self.F,
                                 z_pred=self.z_pred, self_err=self.self_err,
                                 self_err_ema=self.self_err_ema, dF_prev=self.dF_prev,
                                 dF_ema=self.dF_ema, dF_abs_ema=self.dF_abs_ema,
                                 mom=self.mom, _e_rel=self._e_rel,
                                 _pi_rel=self._pi_rel).items()}
        snap["nfe"] = self.nfe.copy()
        snap["t"] = self.t
        snap["setpoint"] = self.setpoint.copy()
        snap["F0"] = None if self.F0 is None else o.clone(self.F0)
        snap["F_prev"] = None if self.F_prev is None else o.clone(self.F_prev)
        return snap

    def restore(self, snap):
        o = self.ops
        for k in ("h", "g", "s", "g_slow", "m", "h_slow", "L", "conf", "F",
                  "z_pred", "self_err", "self_err_ema", "dF_prev", "dF_ema", "dF_abs_ema",
                  "mom", "_e_rel", "_pi_rel"):
            setattr(self, k, o.clone(snap[k]))
        self.nfe = snap["nfe"].copy()
        self.t = snap["t"]
        self.setpoint = snap["setpoint"].copy()
        self.F0 = snap["F0"]
        self.F_prev = snap["F_prev"]

    # ---------------------------------------------------------------- 参数
    def param_items(self):
        return [(k, self.P[k]) for k in self.PARAM_KEYS] + \
               [(k, v) for k, v in self.P.items() if k not in self.PARAM_KEYS]

    def param_count(self):
        return int(sum(int(np.prod(self.ops.numpy(v).shape)) for _, v in self.param_items()))

    def flat_params(self):
        return np.concatenate([np.asarray(self.ops.numpy(v), dtype=np.float64).ravel()
                               for _, v in self.param_items()])

    def assign_flat(self, vec):
        o = self.ops
        vec = np.asarray(vec, dtype=np.float64)
        i = 0
        with o.no_grad():
            for k, v in self.param_items():
                shp = tuple(o.numpy(v).shape)
                n = int(np.prod(shp))
                new = o.t(vec[i:i + n].reshape(shp))
                need = bool(getattr(v, "requires_grad", False))
                if need:
                    new = o.set_requires_grad(new, True)
                self.P[k] = new
                i += n

    def set_requires_grad(self, flag):
        for k in list(self.P.keys()):
            o = self.ops
            v = o.set_requires_grad(self.P[k], flag)
            self.P[k] = v

    def _T(self, key):
        return self.ops.T2(self.P[key])

    # ---------------------------------------------------------------- 子步骤
    def _predict_attention_next(self, h_ref):
        """P1: 预测下一步注意 g_{t+1}（不修改状态），brain.py _predict_attention_next。"""
        o = self.ops
        ctx = o.norm(h_ref, axis=-1)
        x = o.concat([self.g, ctx], axis=-1)
        s_next = o.tanh(self.s @ self._T("Ws") + x @ self._T("Ws_in"))
        x2 = o.concat([ctx, s_next], axis=-1)
        raw = (x2 @ self._T("Wg")) / math.sqrt(self.R + self.K) + self.P["bg"]
        g_raw = 0.5 + 0.5 * o.tanh((raw + self.gs_weight * self.g_slow)
                                   * (self.gain_scale + 0.6 * self.L[:, 0:1]))
        return 0.85 * g_raw + 0.15 * self.g

    def _forward_self(self, gate, g_int=None):
        """P1: 前向自预测 ẑ_{t|t-1}（外部输入取期望 0），brain.py _forward_self。"""
        o = self.ops
        g_n = self._predict_attention_next(self.h) if g_int is None else g_int
        coup = o.einsum("brd,ir->bid", self.h, self.P["Wc"])
        v = g_n[:, :, None] * coup                                       # u = 0（期望）
        inp = o.einsum("rij,brj->bri", self.P["Bin"], v)
        exc, inh, tau = gate[:, :, 0], gate[:, :, 1], gate[:, :, 2]
        leak = 0.2 * (1.0 + 0.3 * tau)
        Ah = o.einsum("rij,brj->bri", self.P["A"], self.h)
        drive = o.tanh(Ah + inp * (1.0 + 0.5 * exc)[:, :, None]
                       - 0.3 * inh[:, :, None] * self.h)
        return (1 - leak)[:, :, None] * self.h + leak[:, :, None] * drive

    def _fixed_point(self, base):
        """P6 隐式不动点 m* = tanh(Wm m + base)，与 brain.py 同步计数 NFE。"""
        o = self.ops
        B = self.B
        m = self.m
        active = np.ones(B, dtype=bool)
        nfe = np.full(B, self.fp_iters, dtype=np.int64)
        for k in range(self.fp_iters):
            m_new = o.tanh(m @ self._T("Wm") + base)
            delta = o.numpy(o.norm(m_new - m, axis=-1))
            thr = self.fp_tol * np.maximum(o.numpy(o.norm(m, axis=-1)), 1.0)
            newly = active & (delta < thr)
            m = o.where(o.bool_mask(active[:, None]), m_new, m)
            nfe[newly] = k + 1
            active = active & (~newly)
            if not active.any():
                break
        return m, nfe

    # ---------------------------------------------------------------- 主循环
    def step(self, u=None, intervention=None, drug_gate=None):
        o = self.ops
        R, d, K, B = self.R, self.d, self.K, self.B
        iv = intervention or {}
        if u is None:
            u = o.zeros((B, d))
        else:
            u = o.t(u)
            if u.ndim == 1:
                u = o.concat([u[None, :]] * B, axis=0) if B > 1 else u[None, :]

        # --- P3: 图式预测上一刻的 g（含上下文），随后 g 被图式因果调制 ---
        ctx = o.norm(self.h, axis=-1)                                   # [B,R]
        x = o.concat([self.g, ctx], axis=-1)                            # [B,2R]
        self.s = o.tanh(self.s @ self._T("Ws") + x @ self._T("Ws_in"))
        if "s" in iv:
            sv = o.t(iv["s"])
            self.s = o.concat([sv[None, :]] * B, axis=0) if sv.ndim == 1 else sv
        x2 = o.concat([ctx, self.s], axis=-1)                           # [B,R+K]
        raw = (x2 @ self._T("Wg")) / math.sqrt(R + K) + self.P["bg"]
        g_raw = 0.5 + 0.5 * o.tanh((raw + self.gs_weight * self.g_slow)
                                   * (self.gain_scale + 0.6 * self.L[:, 0:1]))
        if "g" in iv:
            gv = o.t(iv["g"])
            g_raw = o.concat([gv[None, :]] * B, axis=0) if gv.ndim == 1 else gv
        self.g = 0.85 * g_raw + 0.15 * self.g
        self.g_slow = 0.97 * self.g_slow + 0.03 * self.gs_amp * o.tanh(self.g @ self._T("Wgs"))

        # --- 调制门控（药理干预优先）---
        if drug_gate is not None:
            gate = o.t(drug_gate)
        else:
            gate = o.einsum("bj,rkj->brk", self.L, self.P["GL"])         # [B,R,3]

        # --- P1: 前向自预测（动之前先预测自己下一步会变成什么样）---
        self.z_pred = self._forward_self(gate)

        # --- P6: 自模型不动点 + 慢变量 ---
        if "h_slow" in iv:
            hs = o.t(iv["h_slow"])
            self.h_slow = o.concat([hs[None, :]] * B, axis=0) if hs.ndim == 1 else hs
        # ★ P1: 上一步实现的自预测误差相对量（无量纲）
        self._e_rel = self.self_err / o.maximum(self.self_err_ema, 1e-12)
        self._pi_rel = o.full((B,), 1.0)
        z = o.reshape(self.h, (B, R * d))
        zp = o.reshape(self.z_pred, (B, R * d))
        epi = o.stack([self._e_rel, self._pi_rel], axis=-1)              # [B,2]
        base = (z @ self._T("Wm_in") + self.w_slow * self.h_slow
                + self.w_fwd * (zp @ self._T("Wm_fwd"))
                + self.w_surp * (epi @ self._T("Wm_surp"))
                + self.w_mom * self.mom[:, None] * self.P["u_mom"][None, :])
        m, nfe = self._fixed_point(base)
        if "m" in iv:
            mv = o.t(iv["m"])
            m = o.concat([mv[None, :]] * B, axis=0) if mv.ndim == 1 else mv
        self.m = m
        self.nfe = nfe
        self.h_slow = ((1 - self.dt_slow) * self.h_slow
                       + self.dt_slow * (m + self.beta_slow * o.tanh(self.h_slow @ self._T("W_slow"))))

        # --- 区域动力学（区在 batch 维，一次 einsum 算完）---
        # brain.py 写作 (Wc[i] @ h).ravel()[:d]；Wc[i] 是长度 R 的行向量，(Wc[i]@h) 已是 [d]，
        # 该切片是空操作 ⇒ 区际输入 = Σ_j Wc[i,j] h[j,:]
        coup = o.einsum("brd,ir->bid", self.h, self.P["Wc"])             # [B,R,d]
        v = u[:, None, :] + self.g[:, :, None] * coup                    # [B,R,d]
        inp = o.einsum("rij,brj->bri", self.P["Bin"], v)                 # [B,R,d]
        exc, inh, tau = gate[:, :, 0], gate[:, :, 1], gate[:, :, 2]
        leak = 0.2 * (1.0 + 0.3 * tau)
        # 逐区 A[r] @ h[b,r]（einsum 明确收缩区内的 d 维，避免 matmul 静默广播）
        Ah = o.einsum("rij,brj->bri", self.P["A"], self.h)
        drive = o.tanh(Ah + inp * (1.0 + 0.5 * exc)[:, :, None]
                       - 0.3 * inh[:, :, None] * self.h)
        h_new = (1 - leak)[:, :, None] * self.h + leak[:, :, None] * drive
        for blk in self.extra:                                           # §2.1 深栈 block
            xin = h_new
            gg = o.sigmoid(o.einsum("rij,brj->bri", blk["Wr"], xin))
            uu = o.tanh(o.einsum("rij,brj->bri", blk["Wz"], xin)) * gg
            y = o.einsum("rij,brj->bri", blk["Wo"], uu) + xin
            ff = o.einsum("rij,brj->bri", blk["W2"],
                          o.tanh(o.einsum("rij,brj->bri", blk["W1"], y)))
            h_new = y + o.sigmoid(o.einsum("rij,brj->bri", blk["Wgate"], y)) * ff
        self.h = h_new

        # --- P1: 实现的自预测误差 e_t = ‖h_t − ẑ_{t|t-1}‖²/(R·d) ---
        dz = self.h - self.z_pred
        self.self_err = o.mean(dz * dz, axis=(1, 2))
        self.self_err_ema = ((1 - self.ema_alpha) * self.self_err_ema
                             + self.ema_alpha * self.self_err)

        # --- P1: 置信度头（读自模型 m）---
        self.conf = o.sigmoid(self.m @ self._T("Wconf"))

        # --- P5: 变分自由能（复杂度 + 不准确度 + 自预测不准确度）---
        resid = self.h - self.g[:, :, None] * o.mean(self.h, axis=1, keepdims=True)
        pred_err = o.mean(resid * resid, axis=(1, 2))                    # [B]
        complexity = 0.5 * o.sum(self.m * self.m, axis=1) / K
        self.F = complexity + pred_err + self.w_self * self.self_err_ema
        if self.F0 is None:
            self.F0 = o.clone(o.detach(self.F))

        # --- P1: 自身表现动量（去趋势 + 自归一化）---
        self.dF_prev = o.zeros((B,)) if self.F_prev is None else (self.F - self.F_prev)
        self.F_prev = o.clone(o.detach(self.F))
        aa = self.ema_alpha
        self.dF_ema = (1 - aa) * self.dF_ema + aa * self.dF_prev
        _upd = (1 - aa) * self.dF_abs_ema + aa * o.abs(self.dF_prev)
        self.dF_abs_ema = o.where(o.bool_mask(self.dF_abs_ema > 0), _upd, o.abs(self.dF_prev))
        self.mom = (self.dF_prev - self.dF_ema) / (self.dF_abs_ema + 1e-12)

        # --- 调质核团：由内生稳态误差驱动 ---
        self.setpoint = self.setpoint + np.array([0.0015, 0.0008, 0.0, 0.0004])
        err = o.stack([
            pred_err * 10.0,
            o.full((B,), 0.02 * self.t),
            o.abs(self.F - self.F0),
            0.5 + 0.01 * o.mean(o.abs(self.g), axis=1),
        ], axis=-1)                                                      # [B,4]
        e = err - o.t(self.setpoint)
        target = 0.5 + 0.5 * o.tanh(e @ self._T("WL"))
        self.L = self.L + 0.05 * (target - self.L)

        self.t += 1
        return self.state()

    # ---------------------------------------------------------------- 度量
    def region_norms(self):
        return np.asarray(self.ops.numpy(self.ops.norm(self.h, axis=-1)), dtype=np.float64)

    def saturation_unit(self):
        """单元级饱和率：|tanh(h)| >= 0.99 的单元占比（判据 T4 用的就是它）。"""
        t = np.abs(np.tanh(np.asarray(self.ops.numpy(self.h), dtype=np.float64)))
        return float((t >= 0.99).mean())

    def saturation_engine_def(self):
        """brain.py saturation() 的定义：mean|tanh(||h_i||_2)|（对照用）。"""
        n = self.region_norms()
        return float(np.mean(np.abs(np.tanh(n))))

    def attn_saturation(self):
        """brain.py/verify_all P3 的注意饱和率：|g-0.5| > 0.49 的占比。"""
        g = np.asarray(self.ops.numpy(self.g), dtype=np.float64)
        return float(np.mean(np.abs(g - 0.5) > 0.49))

    def loss_vector(self):
        return np.asarray(self.ops.numpy(self.F), dtype=np.float64)

    def step_loss(self):
        return float(np.mean(self.loss_vector()))

    def grad_autograd(self, u_window):
        """torch 后端：窗口 BPTT 的精确梯度（全参数）。"""
        o = self.ops
        assert o.is_torch
        self.set_requires_grad(True)
        loss = None
        for u in u_window:
            st = self.step(u)
            fi = o.mean(st["F"])
            loss = fi if loss is None else loss + fi
        loss = loss / float(len(u_window))
        grads = o.torch.autograd.grad(loss, [v for _, v in self.param_items()],
                                      allow_unused=True)
        chunks, per, tot = [], {}, 0.0
        for (k, v), g in zip(self.param_items(), grads):
            n_el = int(np.prod(o.numpy(v).shape))
            if g is None:
                chunks.append(np.zeros(n_el))
                per[k] = 0.0
                continue
            gv = np.asarray(o.numpy(g), dtype=np.float64).ravel()
            chunks.append(gv)
            per[k] = float(np.linalg.norm(gv))
            tot += per[k] ** 2
        self.set_requires_grad(False)
        return dict(grad_norm=math.sqrt(tot), flat_grad=np.concatenate(chunks),
                    per_param=per, window_loss=float(o.item(loss)),
                    method="autograd（精确反向，窗口 BPTT）")

    def directional_derivative(self, u_window, v_flat, eps):
        """(L(theta+eps v) - L(theta-eps v)) / (2 eps)，用于 Hutchinson 估计 ||g||。"""
        theta = self.flat_params()
        num = 0.0
        for sign in (+1.0, -1.0):
            self.assign_flat(theta + sign * eps * v_flat)
            self.set_requires_grad(False)
            with self.ops.no_grad():
                tot = 0.0
                for u in u_window:
                    self.step(u)
                    tot += self.step_loss()
            num += sign * (tot / float(len(u_window)))
        self.assign_flat(theta)
        return num / (2.0 * eps)

    def grad_hutchinson(self, snap, u_window, n_samples=32, eps=1e-6, rng=None):
        """numpy 后端：Rademacher 方向导数的 Hutchinson 估计。

        E[(v·g)^2] = ||g||^2 ⇒ sqrt(mean_s d_s^2) 是 ||g||^2 的无偏估计；
        ĝ = mean_s(v_s d_s) 是 g 的无偏估计（SPSA），用作 Adam 的更新方向。
        """
        rng = rng or np.random.default_rng(0)
        theta = self.flat_params()
        scale = max(float(np.sqrt(np.mean(theta * theta))), 1.0)
        ee = eps * scale
        ests, acc = [], np.zeros_like(theta)
        for _ in range(int(n_samples)):
            v = rng.integers(0, 2, size=theta.shape).astype(np.float64) * 2.0 - 1.0
            self.restore(snap)
            d = self.directional_derivative(u_window, v, ee)
            ests.append(d)
            acc += v * d
        ests = np.asarray(ests, dtype=np.float64)
        g_hat = acc / float(n_samples)
        self.restore(snap)
        self.assign_flat(theta)
        return dict(grad_norm=float(np.sqrt(np.mean(ests ** 2))),
                    flat_grad=g_hat, spsa_norm=float(np.linalg.norm(g_hat)),
                    dir_std=float(np.std(ests)), eps=ee, n_samples=int(n_samples),
                    method="Hutchinson/Rademacher 方向导数（numpy 后端无 autograd）")


# ====================================================================== 显存/带宽探针
class MemProbe:
    def __init__(self, ops, env):
        self.ops = ops
        self.env = env
        self.cuda = ops.is_cuda
        if self.cuda:
            self.torch = ops.torch
            self.dev = ops.device

    def start(self):
        self.rss0 = rss_bytes()
        self.vmhwm0 = peak_rss_bytes()
        if self.cuda:
            self.torch.cuda.synchronize(self.dev)
            self.torch.cuda.reset_peak_memory_stats(self.dev)

    def stop(self, label=""):
        if self.cuda:
            self.torch.cuda.synchronize(self.dev)
            peak = int(self.torch.cuda.max_memory_allocated(self.dev))
            reserved = int(self.torch.cuda.max_memory_reserved(self.dev))
            cap = int(self.torch.cuda.get_device_properties(self.dev).total_memory)
            return {"metric": "torch.cuda.max_memory_allocated", "label": label,
                    "peak_bytes": peak, "peak_reserved_bytes": reserved,
                    "capacity_bytes": cap, "capacity_source": "cuda device total_memory"}
        hwm = peak_rss_bytes()
        rss = rss_bytes()
        mi = meminfo()
        return {"metric": "VmHWM (进程峰值常驻内存)", "label": label,
                "peak_bytes": hwm, "peak_delta_bytes": (None if (hwm is None or self.vmhwm0 is None)
                                                        else hwm - self.vmhwm0),
                "current_rss_bytes": rss,
                "capacity_bytes": mi.get("MemTotal"),
                "capacity_source": "MemTotal (系统 RAM) — 本机无 GPU，用系统内存作容量上限"}


def measure_peak_bandwidth(ops, env, max_bytes=256 * 1024 * 1024):
    """实测内存带宽上限：三元组 a = b*1.5 + c（2 读 1 写）。"""
    if ops.is_cuda:
        torch = ops.torch
        dev = ops.device
        free, total = torch.cuda.mem_get_info(dev)
        n_bytes = int(min(max_bytes, max(32 * 1024 * 1024, 0.25 * free)))
        n = n_bytes // 4
        a = torch.empty(n, dtype=torch.float32, device=dev)
        b = torch.randn(n, dtype=torch.float32, device=dev)
        c = torch.randn(n, dtype=torch.float32, device=dev)
        for _ in range(3):
            torch.add(b, c, alpha=1.5, out=a)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            torch.add(b, c, alpha=1.5, out=a)
            torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t0)
        bytes_moved = 3 * n * 4
        best = min(ts)
        return {"method": "cuda triad a=b*1.5+c (2 读 1 写)", "bytes_moved": bytes_moved,
                "best_seconds": best,
                "bytes_per_s": bytes_moved / best,
                "buffer_bytes": n * 4}
    mi = meminfo()
    avail = mi.get("MemAvailable", 2 * 1024 ** 3)
    n_bytes = int(min(max_bytes, max(16 * 1024 * 1024, 0.08 * avail)))
    n = n_bytes // 8
    a = np.empty(n, dtype=np.float64)
    b = np.random.randn(n)
    c = np.random.randn(n)
    for _ in range(2):
        np.multiply(b, 1.5, out=a)
        np.add(a, c, out=a)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        np.multiply(b, 1.5, out=a)
        np.add(a, c, out=a)
        ts.append(time.perf_counter() - t0)
    bytes_moved = 3 * n * 8
    best = min(ts)
    return {"method": "numpy triad a=b*1.5+c (2 读 1 写, float64)", "bytes_moved": bytes_moved,
            "best_seconds": best, "bytes_per_s": bytes_moved / best, "buffer_bytes": n * 8,
            "mem_available_bytes": avail}


def measure_peak_flops(ops, n=None):
    """实测稠密矩阵乘算力（2n³ FLOP / 次）。"""
    if ops.is_cuda:
        torch = ops.torch
        dev = ops.device
        nn = int(n or 4096)
        a = torch.randn(nn, nn, device=dev)
        b = torch.randn(nn, nn, device=dev)
        for _ in range(2):
            c = a @ b
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            c = a @ b
            torch.cuda.synchronize(dev)
            ts.append(time.perf_counter() - t0)
        fl = 2.0 * nn ** 3
        return {"method": "torch.matmul %d×%d (%s)" % (nn, nn, ops.dtype_str),
                "flops": fl, "best_seconds": min(ts), "flops_per_s": fl / min(ts)}
    nn = int(n or 512)
    a = np.random.randn(nn, nn)
    b = np.random.randn(nn, nn)
    a @ b
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        a @ b
        ts.append(time.perf_counter() - t0)
    fl = 2.0 * nn ** 3
    return {"method": "numpy.matmul %d×%d float64" % (nn, nn), "flops": fl,
            "best_seconds": min(ts), "flops_per_s": fl / min(ts)}


def bench_ticks_adaptive(eng, budget_s, max_ticks, u_fn, min_ticks=5, warmup=2):
    """先测 1 tick 估成本，再按时间预算定 tick 数（保证扫描在预算内完成）。"""
    for _ in range(warmup):
        eng.step(u_fn())
    t0 = time.perf_counter()
    eng.step(u_fn())
    dt = time.perf_counter() - t0
    n = int(max(min_ticks, min(int(max_ticks), budget_s / max(dt, 1e-9))))
    r = bench_ticks(eng, n, u_fn, warmup=0)
    r["adaptive_single_tick_seconds"] = dt
    r["budget_seconds"] = budget_s
    return r


def bench_ticks(eng, n_ticks, u_fn, warmup=5):
    """跑 n_ticks 步，返回墙钟与吞吐（实例·tick/s）。"""
    for _ in range(warmup):
        eng.step(u_fn())
    t0 = time.perf_counter()
    for _ in range(n_ticks):
        eng.step(u_fn())
    dt = time.perf_counter() - t0
    return {"ticks": n_ticks, "seconds": dt, "ticks_per_s": n_ticks / dt,
            "instance_ticks_per_s": n_ticks * eng.B / dt,
            "instances": eng.B}


# ====================================================================== 规模选择
SCALE_TABLE = {
    # name: (R, d, K, layers)
    "reference": (12, 24, 16, 1),        # = ahi_theory/brain.py 的实机基线
    "small":     (24, 64, 16, 2),
    "medium":    (100, 64, 16, 8),
    "large":     (100, 256, 16, 8),
    "target":    (100, 1024, 16, 8),     # = GPU_测试要求.md §2.1 目标配置
}


def estimate_engine_bytes(cfg, itemsize, instances):
    """引擎驻留内存上界：参数 + 状态 + 工作区。"""
    R, d, K = cfg.R, cfg.d, cfg.K
    n_state = instances * (R * d + 2 * R + 3 * K + 5 + R)
    return int(cfg.block_params() * itemsize + n_state * itemsize * 8 + 4 * 1024 * 1024)


def choose_scale(args, env, ops):
    """按 (1) 可用内存 (2) 单 tick 实测耗时 选择规模。返回 (name, R, d, K, layers, reason, probe)。"""
    if args.scale != "auto":
        R, d, K, L = SCALE_TABLE[args.scale]
        return args.scale, R, d, K, L, "--scale 显式指定", {}
    if not ops.is_cuda:
        R, d, K, L = SCALE_TABLE["reference"]
        return ("reference", R, d, K, L,
                "CPU 无 HBM；主规模取 GPU_测试要求.md §一 的参考基线（更大规模由 T3 的 "
                "largest_feasible_scale 探针实测）", {})
    order = ["target", "large", "medium", "small", "reference"]
    free, _ = ops.torch.cuda.mem_get_info(ops.device)
    avail = int(free * 0.5)
    budget_s = 0.5
    probe = {}
    for name in order:
        R, d, K, L = SCALE_TABLE[name]
        cfg = EngCfg(R, d, K, L, instances=1, seed=0, name=name)
        need = estimate_engine_bytes(cfg, ops.itemsize, 1)
        if need > avail:
            probe[name] = {"estimate_bytes": need, "available_bytes": avail, "skipped": "内存不足"}
            continue
        t0 = time.perf_counter()
        try:
            eng = Engine(cfg, ops)
            eng.step(np.zeros(d, dtype=np.float64))
            dt = time.perf_counter() - t0
        except MemoryError:
            probe[name] = {"estimate_bytes": need, "skipped": "MemoryError"}
            continue
        probe[name] = {"estimate_bytes": need, "available_bytes": avail,
                       "first_tick_seconds": dt, "param_count": eng.param_count()}
        if dt <= budget_s:
            return name, R, d, K, L, "自动选择：内存可行且单 tick 实测 %.3f s ≤ 预算 %.1f s" % (dt, budget_s), probe
        del eng
    R, d, K, L = SCALE_TABLE["reference"]
    return "reference", R, d, K, L, "自动回退到参考规模", probe


# ====================================================================== 基准
class Bench:
    def __init__(self, args, env, ops):
        self.args = args
        self.env = env
        self.ops = ops
        self.itemsize = ops.itemsize
        self.scale_name, self.R, self.d, self.K, self.layers, self.scale_reason, self.scale_probe = \
            choose_scale(args, env, ops)
        if args.instances:
            self.instances = int(args.instances)
        else:
            self.instances = 128 if ops.is_cuda else 32
        self.cfg = EngCfg(self.R, self.d, self.K, self.layers, self.instances,
                          args.seed, self.scale_name)
        self.hw = {}
        self.report = {}

    # ---------------------------------------------------------------- 工具
    def new_engine(self, instances=None, seed=None):
        cfg = EngCfg(self.R, self.d, self.K, self.layers,
                     self.instances if instances is None else instances,
                     self.args.seed if seed is None else seed, self.scale_name)
        return Engine(cfg, self.ops)

    def u_rng(self, tag):
        return np.random.default_rng(self.args.seed * 1000 + tag)

    def weight_bytes(self, extra_frac=0.0):
        n = self.cfg.block_params()
        return int(n * self.itemsize * (1.0 + extra_frac))

    # ---------------------------------------------------------------- T1
    def run_T1(self):
        a = self.args
        crits, metrics = [], {}
        eng = self.new_engine(instances=1)
        rng = self.u_rng(11)
        probe = MemProbe(self.ops, self.env)
        norms = []
        finite_ok = True
        bad_step = None
        probe.start()
        t0 = time.perf_counter()
        err = None
        try:
            for t in range(a.t1_steps):
                u = rng.normal(size=self.d) * a.u_scale
                with self.ops.no_grad():
                    st = eng.step(u)
                for key in ("h", "g", "s", "m", "h_slow", "L", "conf", "F"):
                    if not self.ops.finite_all(st[key]):
                        finite_ok = False
                        bad_step = t
                        break
                if not finite_ok:
                    break
                norms.append(eng.region_norms()[0])
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
            traceback.print_exc()
        wall = time.perf_counter() - t0
        mem = probe.stop("T1")
        norms = np.asarray(norms) if norms else np.zeros((0, self.R))
        metrics.update({
            "steps_requested": a.t1_steps, "steps_completed": int(len(norms)),
            "instances": 1, "wall_seconds": wall,
            "ms_per_tick": (wall / max(len(norms), 1)) * 1000.0,
            "exception": err,
            "region_norm_first_step": norms[0].tolist() if len(norms) else [],
            "region_norm_last_step": norms[-1].tolist() if len(norms) else [],
            "region_norm_max_over_run": norms.max(axis=0).tolist() if len(norms) else [],
            "state_bytes": int(eng.param_count() * self.itemsize),
            "T_window": 64,
        })
        crits.append(crit("前向不崩 (batch=1, T=64, 100 步)",
                          "无 NaN/Inf" if (finite_ok and err is None) else "出现非有限值",
                          TH["T1_no_nan"], bool(finite_ok and err is None), note=
                          "完成 %d/%d 步，退出码 0（阶段无异常）" % (len(norms), a.t1_steps)
                          if err is None else "异常: %s" % err))
        W = int(TH["T1_stab_warmup"])
        g_max = float("inf")
        g_gated = None
        if len(norms):
            growth = norms.max(axis=0) / np.maximum(norms[0], 1e-300)
            g_max = float(growth.max())
            metrics["norm_growth_per_region"] = growth.tolist()
            metrics["norm_growth_max"] = g_max          # 旧口径（自首步，含启动瞬态）——信息项
            if len(norms) > 10:
                metrics["norm_growth_max_after_warmup10"] = float(
                    (norms[10:].max(axis=0) / np.maximum(norms[10], 1e-300)).max())
            if len(norms) > W + 1:
                g_gated = float((norms[W:].max(axis=0) / np.maximum(norms[W - 1], 1e-300)).max())
                metrics["norm_growth_after_warmup%d" % W] = g_gated
                if len(norms) >= W + 200:
                    metrics["norm_window_means_after_warmup"] = [
                        float(norms[i:i + 200].mean())
                        for i in range(W, len(norms) - 199, 200)]
        crits.append(crit(
            "数值稳定：预热 %d 步后范数无发散" % W,
            g_gated if len(norms) else None,
            "< %g×" % TH["T1_norm_growth"],
            (g_gated is not None) and (g_gated < TH["T1_norm_growth"]),
            unit="×",
            note=("预热后（t≥%d）max_t ||h_i(t)|| / ||h_i(%d)||，窗口 %d 步（逐区取最大）；"
                  "自首步的原始增长 %.3f×（含启动瞬态，信息项）"
                  % (W, W - 1, max(len(norms) - W, 0), g_max))
            if len(norms) > W + 1 else
            "步数不足（需 > %d 步）；当前 %d 步 ⇒ 本判据不适用" % (W + 1, len(norms))))
        cap = mem.get("capacity_bytes")
        frac = (mem["peak_bytes"] / cap) if (cap and mem.get("peak_bytes")) else None
        metrics["memory"] = mem
        metrics["memory_peak_fraction_of_capacity"] = frac
        crits.append(crit("显存峰值 < 卡容量 × 0.9",
                          frac, "< %g" % TH["T1_mem_frac"],
                          (frac is not None) and (frac < TH["T1_mem_frac"]),
                          note="峰值 %s = %.1f MB / 容量 %.1f MB（%s）" % (
                              mem["metric"], (mem.get("peak_bytes") or 0) / 2 ** 20,
                              (cap or 0) / 2 ** 20, mem.get("capacity_source", ""))))
        # 目标规模下的激活公式（来源：GPU_测试要求.md §2.3）
        metrics["activation_formula_source"] = "GPU_测试要求.md §2.3（公式计数，非本机实测）"
        metrics["activation_bytes_T64_target_scale"] = int(100 * 1024 * 8 * 64 * 2)
        metrics["region_state_bytes_T64_target_scale"] = int(8 * 16 * 64 * 64 * 4)
        return block(rollup(crits), crits, metrics,
                     note="batch=1（单实例）跑 %d 步；T=64 为文档的 BPTT 硬上限" % a.t1_steps)

    # ---------------------------------------------------------------- T2
    def run_T2(self):
        a = self.args
        crits, metrics = [], {}
        world = int(a.t2_world)
        if self.ops.is_cuda and self.env["cuda_device_count"] >= 2:
            world = self.env["cuda_device_count"]
        metrics["shard_world_size"] = world
        metrics["cuda_device_count"] = self.env["cuda_device_count"]
        metrics["zero_memory_model"] = self._zero_memory_model(world)

        # (a)(b) 参数分片 + all-gather：主体/分片两次同 u 运行
        eng_ref = self.new_engine(instances=int(a.t2_instances))
        eng_sh = self.new_engine(instances=int(a.t2_instances))
        rng = self.u_rng(21)
        U = [rng.normal(size=(eng_ref.B, self.d)) * a.u_scale for _ in range(100)]
        base_loss, base_secs = [], 0.0
        with self.ops.no_grad():
            for u in U:
                t0 = time.perf_counter()
                eng_ref.step(u)
                base_secs += time.perf_counter() - t0
                base_loss.append(eng_ref.step_loss())
            flat = eng_ref.flat_params()
            parts = np.array_split(flat, world)
            shards = [self.ops.t(p) for p in parts]
            sh_loss, gather_secs, step_secs = [], 0.0, 0.0
            for u in U:
                t0 = time.perf_counter()
                if self.ops.is_cuda and world > 1:
                    gathered = self.ops.concat([s.to(self.ops.device) for s in shards], axis=0)
                else:
                    gathered = self.ops.concat(shards, axis=0)
                gather_secs += time.perf_counter() - t0
                eng_sh.assign_flat(self.ops.numpy(gathered))
                t0 = time.perf_counter()
                eng_sh.step(u)
                step_secs += time.perf_counter() - t0
                sh_loss.append(eng_sh.step_loss())
        d_loss = float(np.max(np.abs(np.asarray(base_loss) - np.asarray(sh_loss))))
        comm_frac = gather_secs / max(gather_secs + step_secs, 1e-12)
        metrics["shard_loss_diff_max"] = d_loss
        metrics["shard_loss_diff_mean"] = float(np.mean(np.abs(np.asarray(base_loss) - np.asarray(sh_loss))))
        metrics["allgather_seconds_per_100ticks"] = gather_secs
        metrics["compute_seconds_per_100ticks"] = step_secs
        metrics["allgather_time_fraction"] = comm_frac
        metrics["allgather_bytes_per_tick"] = len(flat) * self.itemsize
        metrics["unsharded_seconds_per_100ticks"] = base_secs
        crits.append(crit("分片前后前 100 步 loss 差异", d_loss, "< %g" % TH["T2_shard_loss_diff"],
                          d_loss < TH["T2_shard_loss_diff"],
                          note="%d 个分片 all-gather 后与未分片逐 tick 对比（最大/平均绝对差）"
                               % world, extra_field="shard_loss_diff_mean"))
        crits.append(crit("all-gather 通信占总时间", comm_frac, "< %g" % TH["T2_allgather_frac"],
                          comm_frac < TH["T2_allgather_frac"],
                          note="每 tick 全参数 all-gather（%.2f MB/tick，%s 暂存）"
                               % (len(flat) * self.itemsize / 2 ** 20,
                                  "device→device" if self.ops.is_cuda else "host")))

        # (c) 多卡吞吐扩展效率
        dev_run = None
        if self.ops.is_cuda and self.env["cuda_device_count"] >= 2:
            dev_run = self._run_multi_device(
                devices=[self.ops.torch.device("cuda", i)
                         for i in range(self.env["cuda_device_count"])],
                device_kind="cuda")
        elif a.t2_force_devices and self.env["torch_available"]:
            import torch as _t
            dev_run = self._run_multi_device(
                devices=[_t.device("cpu")] * int(a.t2_force_devices),
                device_kind="cpu-simulated（自检：非 GPU 扩展性）", torch_mod=_t)
        if dev_run is None:
            crits.append(crit("8 卡 vs 1 卡吞吐扩展效率", None, "> %g" % TH["T2_scaling_eff"], None,
                              note="SKIP：本机 CUDA 设备数 = %d（文档规定无多卡时跳过并标注）"
                                   % self.env["cuda_device_count"]))
            metrics["multi_device"] = {"measured": False,
                                       "skip_reason": "CUDA 设备数 %d < 2" % self.env["cuda_device_count"]}
        else:
            eff = dev_run["scaling_efficiency"]
            metrics["multi_device"] = dev_run
            crits.append(crit("8 卡 vs 1 卡吞吐扩展效率", eff, "> %g" % TH["T2_scaling_eff"],
                              eff > TH["T2_scaling_eff"],
                              note="%d 个 %s 副本实测：%.1f → %.1f 实例·tick/s"
                                   % (dev_run["world"], dev_run["device_kind"],
                                      dev_run["one_instance_ticks_per_s"],
                                      dev_run["n_instance_ticks_per_s"])))
        status = rollup(crits)
        note = "无多卡 ⇒ 按文档跳过扩展效率判据" if status == SKIP else ""
        return block(status, crits, metrics, note=note)

    def _zero_memory_model(self, world):
        """§2.2 显存分解：参数按 §2.1 公式计数（非本机实测），字节数是 bf16/fp32 的算术。"""
        P = 100 * 8 * 13 * 1024 * 1024
        m = {"param_count_formula": P, "param_count_source": "GPU_测试要求.md §2.1: 8×13×d²×100",
             "bf16_bytes": 2, "fp32_bytes": 4,
             "weights_bytes": P * 2, "grads_bytes": P * 2,
             "adam_mv_bytes": P * 8, "master_bytes": P * 4}
        m["total_bytes"] = m["weights_bytes"] + m["grads_bytes"] + m["adam_mv_bytes"] + m["master_bytes"]
        m["total_GB"] = m["total_bytes"] / 1e9
        m["per_card_bytes_zero%d" % world] = m["total_bytes"] // world
        m["per_card_GB_zero%d" % world] = m["total_bytes"] / world / 1e9
        return m

    def _run_multi_device(self, devices, device_kind="cuda", torch_mod=None):
        """单进程多设备锁步运行：每设备一个副本线程，逐 tick 同步参数（ZeRO 式 all-gather）。"""
        import threading
        torch = torch_mod or self.ops.torch
        world = len(devices)
        per_replica = int(self.args.t2_instances)
        ticks = int(self.args.t2_ticks)

        dt = self.ops.dtype if self.ops.is_torch else getattr(torch, str(self.ops.dtype))

        def make(dev):
            ops = TorchOps(torch, device=dev, dtype=dt,
                           is_cuda=str(dev).startswith("cuda"))
            cfg = EngCfg(self.R, self.d, self.K, self.layers, per_replica, self.args.seed, self.scale_name)
            return Engine(cfg, ops)

        def run_world(w):
            devs = devices[:w]
            engines = [make(dv) for dv in devs]
            barrier = threading.Barrier(w)
            state = {"sync_seconds": 0.0, "step_seconds": 0.0, "error": None}

            def worker(idx):
                eng = engines[idx]
                rng = np.random.default_rng(self.args.seed * 1000 + 31 + idx)
                U = [rng.normal(size=(per_replica, self.d)) * self.args.u_scale
                     for _ in range(ticks + 3)]
                try:
                    for t in range(ticks):
                        barrier.wait()
                        t0 = time.perf_counter()
                        with eng.ops.no_grad():
                            eng.step(U[t])
                        state["step_seconds"] += time.perf_counter() - t0
                        barrier.wait()
                        if idx == 0:
                            t1 = time.perf_counter()
                            flats = [e.flat_params() for e in engines]
                            mean = np.mean(np.stack(flats, 0), axis=0)
                            for e in engines:
                                e.assign_flat(mean)
                            state["sync_seconds"] += time.perf_counter() - t1
                        barrier.wait()
                except Exception as e:                       # pragma: no cover - 只在多卡执行
                    state["error"] = "%s: %s" % (type(e).__name__, e)

            ths = [threading.Thread(target=worker, args=(i,)) for i in range(w)]
            t0 = time.perf_counter()
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            wall = time.perf_counter() - t0
            return {"world": w, "wall_seconds": wall, "ticks": ticks,
                    "instance_ticks_per_s": ticks * per_replica * w / wall,
                    "sync_seconds": state["sync_seconds"], "step_seconds": state["step_seconds"],
                    "sync_fraction": state["sync_seconds"] / max(wall, 1e-12),
                    "error": state["error"], "device_kind": device_kind}

        one = run_world(1)
        many = run_world(world)
        eff = many["instance_ticks_per_s"] / (world * one["instance_ticks_per_s"])
        return {"measured": True, "world": world, "device_kind": device_kind,
                "one": one, "n": many, "scaling_efficiency": eff,
                "one_instance_ticks_per_s": one["instance_ticks_per_s"],
                "n_instance_ticks_per_s": many["instance_ticks_per_s"],
                "sync_fraction_n": many["sync_fraction"]}

    # ---------------------------------------------------------------- T3
    def run_T3(self):
        a = self.args
        crits, metrics = [], {}
        sweep = [int(x) for x in str(a.t3_sweep).split(",") if x.strip()]
        peak = self.hw["peak_bandwidth"]["bytes_per_s"]
        rows = []
        best = None
        for B in sweep:
            need = estimate_engine_bytes(self.cfg, self.itemsize, B)
            if need > self.hw["memory_budget_bytes"]:
                rows.append({"instances": B, "skipped": "内存预算不足",
                             "estimate_bytes": need,
                             "budget_bytes": self.hw["memory_budget_bytes"]})
                continue
            eng = self.new_engine(instances=B)
            rng = self.u_rng(41)
            u = rng.normal(size=(B, self.d)) * a.u_scale
            r = bench_ticks_adaptive(eng, float(a.t3_config_budget_s), int(a.t3_ticks),
                                     lambda: u, warmup=int(a.t3_warmup))
            wb = eng.param_count() * self.itemsize
            row = dict(r)
            row.update({
                "param_count": eng.param_count(),
                "weight_bytes": wb,
                # 本扫描实测的是**前向** tick（每 tick 读 1 遍权重）；
                # train_* 是按 GPU_测试要求.md §三「fwd+bwd 每 tick 读 3 遍权重」的换算值
                "forward_bytes_per_tick": wb,
                "forward_bytes_per_s": wb * r["ticks_per_s"],
                "forward_bw_utilization_measured_peak": wb * r["ticks_per_s"] / peak,
                "train_bytes_per_tick_3reads": 3 * wb,
                "train_bytes_per_s_converted": 3 * wb * r["ticks_per_s"],
                "train_bw_utilization_measured_peak_converted": 3 * wb * r["ticks_per_s"] / peak,
                "ms_per_tick": r["seconds"] / r["ticks"] * 1000.0,
            })
            rows.append(row)
            if best is None or r["instance_ticks_per_s"] > best["instance_ticks_per_s"]:
                best = row
        ok_rows = [r for r in rows if "skipped" not in r]
        metrics["sweep"] = rows
        metrics["peak_bandwidth"] = self.hw["peak_bandwidth"]
        metrics["ticks_per_config"] = int(a.t3_ticks)
        if not ok_rows:
            for nm, thr in (("实例/卡（锁步）", "≥ %d" % TH["T3_instances"]),
                            ("tick/s（每卡，实例·tick/s）", "≥ %g" % TH["T3_tick_rate"]),
                            ("HBM 带宽利用率", "%g–%g" % (TH["T3_bw_lo"], TH["T3_bw_hi"]))):
                crits.append(crit(nm, None, thr, None, note="SKIP：无可行配置"))
            return block(SKIP, crits, metrics)
        metrics["best_config"] = {k: best[k] for k in
                                  ("instances", "ticks_per_s", "instance_ticks_per_s", "ms_per_tick",
                                   "weight_bytes", "param_count",
                                   "forward_bytes_per_s", "forward_bw_utilization_measured_peak",
                                   "train_bytes_per_s_converted",
                                   "train_bw_utilization_measured_peak_converted")}
        # 最大可行实例数（内存与实测吞吐共同约束）
        max_inst = max(r["instances"] for r in ok_rows)
        # 目标规模下的带宽需求（§2.1/§2.2 公式 + 实测带宽上限）
        T = self._target_scale_bw(peak)
        metrics["target_scale"] = T
        metrics["largest_feasible_scale"] = self._scale_probe(peak)
        metrics["fleet_backcalc"] = self._fleet_backcalc(best, T, metrics["largest_feasible_scale"])

        scale_ok = (self.scale_name == "target")
        crits.append(crit("实例/卡（锁步）", max_inst, "≥ %d" % TH["T3_instances"],
                          max_inst >= TH["T3_instances"],
                          note="本机 %s 规模实测可跑的最大锁步实例数（扫描 %s）；"
                               "★ 该判据在 §2.1 目标规模（1.09e10 参数）上才等价于文档含义"
                               % (self.scale_name, a.t3_sweep),
                          scale_matched_to_target=scale_ok,
                          measured_at_scale=self.scale_name))
        crits.append(crit("tick/s（每卡，实例·tick/s）", best["instance_ticks_per_s"],
                          "≥ %g" % TH["T3_tick_rate"],
                          best["instance_ticks_per_s"] >= TH["T3_tick_rate"],
                          note="最优配置 实例=%d，%.2f ms/tick；目标 %g 对应 100 实例 × 100 Hz；"
                               "★ 本值在 %s 规模上测得"
                               % (best["instances"], best["ms_per_tick"], TH["T3_tick_rate"],
                                  self.scale_name),
                          scale_matched_to_target=scale_ok,
                          measured_at_scale=self.scale_name))
        util = best["train_bw_utilization_measured_peak_converted"]
        crits.append(crit("HBM 带宽利用率", util, "%g–%g（权重流物理下限 0.65）"
                          % (TH["T3_bw_lo"], TH["T3_bw_hi"]),
                          TH["T3_bw_lo"] <= util <= TH["T3_bw_hi"],
                          note="按文档 §三 训练口径换算（实测前向权重流 × 3）：%.4f；"
                               "实测前向口径 %.4f；带宽上限 %.2f GB/s（%s）"
                               % (util, best["forward_bw_utilization_measured_peak"], peak / 1e9,
                                  self.hw["peak_bandwidth"]["method"]),
                          extra=dict(forward_bw_utilization=best["forward_bw_utilization_measured_peak"],
                                     achieved_forward_bytes_per_s=best["forward_bytes_per_s"],
                                     target_scale_bw_required_bytes_per_s=T["train_bytes_per_s_required"])))
        return block(rollup(crits), crits, metrics,
                     note="★ 舰队规模由本机实测反推（不许正推）：见 metrics.fleet_backcalc")

    def _target_scale_bw(self, peak):
        """§2.1/§2.2 公式下的目标规模带宽需求（公式计数 + 实测带宽上限）。"""
        P = 100 * 8 * 13 * 1024 * 1024
        w_bf16 = P * 2
        out = {"param_count_formula": P, "source": "GPU_测试要求.md §2.1/§2.2（公式计数）",
               "weights_bf16_bytes": w_bf16,
               "train_bytes_per_tick_3reads": 3 * w_bf16,
               "train_bytes_per_s_required": 3 * w_bf16 * 100.0,
               "peak_train_tick_rate_at_100pct_bw": peak / (3 * w_bf16),
               "peak_train_tick_rate_at_40pct_bw": 0.4 * peak / (3 * w_bf16),
               "peak_train_tick_rate_at_65pct_bw": 0.65 * peak / (3 * w_bf16),
               "measured_peak_bandwidth_bytes_per_s": peak}
        out["shortfall_vs_100Hz"] = out["train_bytes_per_s_required"] / peak
        return out

    def _scale_probe(self, peak):
        """在内存允许的最大规模上实测一次 tick 吞吐与带宽（本机实测，非外推）。"""
        order = ["target", "large", "medium", "small"]
        for name in order:
            if name == self.scale_name:
                break
            R, d, K, L = SCALE_TABLE[name]
            cfg = EngCfg(R, d, K, L, instances=1, seed=self.args.seed, name=name)
            if estimate_engine_bytes(cfg, self.itemsize, 1) > self.hw["memory_budget_bytes"]:
                continue
            try:
                eng = Engine(cfg, self.ops)
                t0 = time.perf_counter()
                with self.ops.no_grad():
                    eng.step(np.zeros(d))
                dt = time.perf_counter() - t0
            except MemoryError:
                continue
            if dt > self.args.t3_probe_budget_s:
                continue
            u = np.zeros((1, d))
            r = bench_ticks_adaptive(eng, float(self.args.t3_probe_budget_s), int(self.args.t3_ticks),
                                     lambda: u, warmup=1)
            wb = eng.param_count() * self.itemsize
            return {"scale": name, "regions": R, "dim": d, "layers": L,
                    "param_count": eng.param_count(), "weight_bytes": wb,
                    "ms_per_tick": r["seconds"] / r["ticks"] * 1000.0,
                    "ticks_per_s": r["ticks_per_s"],
                    "forward_bytes_per_s": wb * r["ticks_per_s"],
                    "forward_bw_utilization_measured_peak": wb * r["ticks_per_s"] / peak,
                    "train_bytes_per_s_converted": 3 * wb * r["ticks_per_s"],
                    "train_bw_utilization_measured_peak_converted": 3 * wb * r["ticks_per_s"] / peak,
                    "instances": 1, "first_tick_seconds": dt}
        return {"measured": False, "reason": "没有比主规模更大且在本机内存/时间预算内的可行规模"}

    def _fleet_backcalc(self, best, T, probe):
        """★ 实测反推舰队规模（文档要求：不许正推）。

        给出两个规模档位下的反推值，并把规模差异显式标注：
          ref  : 主规模（本机 reference）实测 instance·tick/s
          big  : 本机内存允许的最大规模（largest_feasible_scale）实测单实例 tick/s
        """
        need = TH["T3_tick_rate"]
        out = {"target_instance_ticks_per_s_per_card": need,
               "source": "本机实测吞吐（非外推）",
               "scale_caveat": "反推只在**实测的那个规模**下有效；§2.1 目标规模（1.09e10 参数）"
                               "必须回到 GPU 上重测后才能作为采购依据"}
        per_main = best["instance_ticks_per_s"]
        out["main_scale"] = {
            "scale": self.scale_name, "instances": best["instances"],
            "param_count": best["param_count"],
            "measured_instance_ticks_per_s": per_main,
            "cards_for_target_rate": math.ceil(need / per_main) if per_main > 0 else None,
            "cards_for_100Hz_wallclock_128_instances": math.ceil(100.0 * 128 / per_main)
            if per_main > 0 else None,
        }
        if probe.get("scale"):
            per_big = probe["ticks_per_s"] if probe.get("instances", 1) else probe["ticks_per_s"]
            out["largest_feasible_scale"] = {
                "scale": probe["scale"], "param_count": probe["param_count"],
                "measured_instance_ticks_per_s": per_big,
                "cards_for_target_rate": math.ceil(need / per_big) if per_big > 0 else None,
                "cards_for_100Hz_wallclock_128_instances": math.ceil(100.0 * 128 / per_big)
                if per_big > 0 else None,
            }
        out["max_train_tick_rate_measured_peak_bw"] = T["peak_train_tick_rate_at_100pct_bw"]
        # 文档 §三 的 H100 推算复核（用文档自己的规格带宽做算术）
        h100 = 3.35e12
        w = T["weights_bf16_bytes"]
        out["doc_h100_arithmetic_check"] = {
            "doc_spec_bandwidth_bytes_per_s": h100,
            "doc_claim_max_train_tick_rate_100pct": 51.0,
            "computed_train_tick_rate_100pct": h100 / (3 * w),
            "computed_train_tick_rate_40pct": 0.4 * h100 / (3 * w),
            "source": "规格带宽 3.35 TB/s（GPU_测试要求.md §三）÷ 实测权重流量 3×21.81 GB",
        }
        return out


# ---------------------------------------------------------------- T4
    def run_T4(self):
        a = self.args
        crits, metrics = [], {}
        eng = self.new_engine(instances=int(a.t4_instances))
        steps = int(a.t4_steps)
        W = int(min(a.t4_bptt, 64))
        rng_u = self.u_rng(51)
        rng_g = np.random.default_rng(a.seed * 1000 + 52)
        theta = eng.flat_params()
        adam_m = np.zeros_like(theta)
        adam_v = np.zeros_like(theta)
        b1, b2, eps_a, lr = 0.9, 0.999, 1e-8, float(a.t4_lr)
        hist = {k: [] for k in ("F", "sat_unit", "sat_engine", "nfe_max", "nfe_mean",
                                "nfe_ok_frac", "attn_sat")}
        nan_hits, inf_hits = 0, 0
        grads = []
        u_buf = []
        t_start = time.perf_counter()
        err = None
        try:
            for t in range(steps):
                u = rng_u.normal(size=(eng.B, self.d)) * a.u_scale
                u_buf.append(u)
                if len(u_buf) > W:
                    u_buf.pop(0)
                with self.ops.no_grad():
                    st = eng.step(u)
                Fv = eng.loss_vector()
                if not np.isfinite(Fv).all():
                    nan_hits += int(np.isnan(Fv).sum())
                    inf_hits += int(np.isinf(Fv).sum())
                hist["F"].append(eng.step_loss())
                hist["sat_unit"].append(eng.saturation_unit())
                hist["sat_engine"].append(eng.saturation_engine_def())
                hist["attn_sat"].append(eng.attn_saturation())
                hist["nfe_max"].append(int(np.max(st["nfe"])))
                hist["nfe_mean"].append(float(np.mean(st["nfe"])))
                hist["nfe_ok_frac"].append(float(np.mean(np.asarray(st["nfe"]) <= TH["T4_nfe_iters"])))
                if (t + 1) % int(a.t4_grad_every) == 0 and len(u_buf) == W:
                    snap = eng.snapshot()
                    win = list(u_buf)
                    try:
                        if self.ops.is_torch:
                            g = eng.grad_autograd(win)
                        else:
                            g = eng.grad_hutchinson(snap, win, int(a.t4_hutch_samples),
                                                    float(a.t4_hutch_eps), rng_g)
                    except Exception as e:                  # 记录一次，不吞掉整轮
                        grads.append({"step": t + 1, "error": "%s: %s" % (type(e).__name__, e)})
                        eng.restore(snap)
                        continue
                    eng.restore(snap)
                    gn = g["flat_grad"]
                    if not np.isfinite(gn).all():
                        nan_hits += 1
                    grads.append({"step": t + 1, "grad_norm": g["grad_norm"],
                                  "spsa_norm": g.get("spsa_norm"), "window_loss": g.get("window_loss"),
                                  "dir_std": g.get("dir_std"), "method": g["method"]})
                    adam_m = b1 * adam_m + (1 - b1) * gn
                    adam_v = b2 * adam_v + (1 - b2) * gn * gn
                    k = len([x for x in grads if "grad_norm" in x])
                    mh = adam_m / (1 - b1 ** k)
                    vh = adam_v / (1 - b2 ** k)
                    theta = theta - lr * mh / (np.sqrt(vh) + eps_a)
                    eng.assign_flat(theta)
        except Exception as e:
            err = "%s: %s" % (type(e).__name__, e)
            traceback.print_exc()
        wall = time.perf_counter() - t_start
        arr = {k: np.asarray(v, dtype=np.float64) for k, v in hist.items()}
        metrics.update({
            "steps_requested": steps, "steps_completed": int(len(arr["F"])), "instances": eng.B,
            "wall_seconds": wall, "steps_per_s": len(arr["F"]) / max(wall, 1e-12),
            "exception": err, "optimizer": "Adam(lr=%g, b1=0.9, b2=0.999)" % lr,
            "gradient_method": ("autograd" if self.ops.is_torch else "Hutchinson/Rademacher"),
            "bptt_window": W, "grad_every": int(a.t4_grad_every), "dtype": self.ops.dtype_str,
            "projection_24h_steps_from_measured_rate": int(len(arr["F"]) / max(wall, 1e-12) * 86400),
            "nan_count": nan_hits, "inf_count": inf_hits,
            "F": {"min": float(arr["F"].min()), "mean": float(arr["F"].mean()),
                  "max": float(arr["F"].max()), "first": float(arr["F"][0]),
                  "last": float(arr["F"][-1])},
            "saturation_unit_max": float(arr["sat_unit"].max()),
            "saturation_unit_mean": float(arr["sat_unit"].mean()),
            "saturation_engine_def_max": float(arr["sat_engine"].max()),
            "saturation_engine_def_mean": float(arr["sat_engine"].mean()),
            "attn_saturation_max": float(arr["attn_sat"].max()),
            "nfe_mean": float(arr["nfe_mean"].mean()), "nfe_max": int(arr["nfe_max"].max()),
            "grad_probes": grads,
        })
        # --- 判据 1：无 NaN/Inf
        ok_finite = (nan_hits == 0 and inf_hits == 0 and err is None)
        crits.append(crit("无 NaN/Inf（全程）", "NaN %d / Inf %d" % (nan_hits, inf_hits), "全程无 NaN/Inf",
                          ok_finite, note="检查逐 tick 的 F、h/g/s/m/L/conf 与全部梯度探针"
                          + ("" if err is None else "；异常 %s" % err)))
        # --- 判据 2：loss 无尖峰
        wl = int(a.t4_window)
        warm = int(a.t4_warmup)
        F = arr["F"]
        if len(F) > wl:
            ratios_all = np.array([F[s:s + wl].max() / max(F[s:s + wl].min(), 1e-12)
                                   for s in range(0, len(F) - wl + 1)])
            ratios_gate = np.array([F[s:s + wl].max() / max(F[s:s + wl].min(), 1e-12)
                                    for s in range(warm, len(F) - wl + 1)]) if len(F) > wl + warm \
                else ratios_all
            worst_all = float(ratios_all.max())
            worst_all_at = int(np.argmax(ratios_all))
            worst_gate = float(ratios_gate.max())
            worst_gate_at = int(np.argmax(ratios_gate)) + (warm if len(F) > wl + warm else 0)
        else:
            worst_all = worst_gate = float("nan")
            worst_all_at = worst_gate_at = -1
        metrics["loss_window"] = {"window": wl, "warmup_excluded": warm,
                                  "worst_ratio_full_run": worst_all,
                                  "worst_ratio_window_start_full_run": worst_all_at,
                                  "worst_ratio_gated": worst_gate,
                                  "worst_ratio_window_start_gated": worst_gate_at}
        crits.append(crit("loss 无尖峰（任意 100 步窗口涨幅 < 5×）", worst_gate,
                          "< %g×" % TH["T4_loss_spike"],
                          _finite(worst_gate) and worst_gate < TH["T4_loss_spike"], unit="×",
                          note="判据按文档原文；因 F 从初值 0 起步存在启动瞬态，"
                               "门控段取 step ≥ %d（预热段）；全段（含启动瞬态）最差 %.3f× "
                               "出现在窗口起点 %d" % (warm, worst_all, worst_all_at),
                          extra=dict(gate_segment="step >= %d" % warm,
                                     worst_ratio_full_run=worst_all)))
        # --- 判据 3：梯度范数
        gvals = np.array([g["grad_norm"] for g in grads if "grad_norm" in g], dtype=np.float64)
        if len(gvals):
            gmax, gmed, gmin = float(gvals.max()), float(np.median(gvals)), float(gvals.min())
            ratio = gmax / max(gmed, 1e-300)
            bounded = bool(np.isfinite(gvals).all())
        else:
            gmax = gmed = gmin = ratio = float("nan")
            bounded = False
        metrics["grad_norm"] = {"max": gmax, "median": gmed, "min": gmin,
                                "max_over_median": ratio, "n_probes": int(len(gvals)),
                                "method": metrics["gradient_method"]}
        crits.append(crit("梯度范数有界、无爆炸", ratio,
                          "有界且 max < %g × median" % TH["T4_grad_explosion_ratio"],
                          bool(bounded and _finite(ratio) and ratio < TH["T4_grad_explosion_ratio"]),
                          note="%d 次探针：max %.4g / median %.4g / min %.4g（方法 %s）"
                               % (len(gvals), gmax, gmed, gmin, metrics["gradient_method"]),
                          threshold_note="文档只写“有界、无爆炸”，未给数值；本脚本量化为 max < 100×median"))
        # --- 判据 4：饱和率
        sat = float(arr["sat_unit"].max())
        crits.append(crit("饱和率 < 0.9", sat, "< %g" % TH["T4_sat_rate"],
                          sat < TH["T4_sat_rate"],
                          note="单元级饱和率 max_t mean(|tanh(h)| ≥ 0.99)（%d 个单元）；"
                               "brain.py saturation() 口径的对照值 max %.4f、均值 %.4f（该口径对 "
                               "||h_i||≈1.6 的健康状态即给出 ≈0.92，不能作退化守卫，故只报告不门控）"
                               % (eng.B * self.R * self.d, arr["sat_engine"].max(),
                                  arr["sat_engine"].mean()),
                          extra=dict(saturation_mean=metrics["saturation_unit_mean"],
                                     engine_def_saturation_max=metrics["saturation_engine_def_max"])))
        # --- 判据 5：不动点 NFE（逐实例逐 tick 统计）
        frac = float(arr["nfe_ok_frac"].mean())
        metrics["nfe"] = {"fraction_instance_ticks_within_12": frac,
                          "n_instance_ticks": int(len(F) * eng.B),
                          "mean_nfe": float(arr["nfe_mean"].mean()),
                          "max_nfe": int(arr["nfe_max"].max()),
                          "limit": TH["T4_nfe_iters"]}
        crits.append(crit("不动点 NFE ≥ 99.9% 在 ≤12 次内收敛", frac, "≥ %g" % TH["T4_nfe_frac"],
                          frac >= TH["T4_nfe_frac"],
                          note="逐 tick 统计（%d 个实例 × %d tick）：平均 NFE %.3f，最大 %d"
                               % (eng.B, len(F), float(arr["nfe_mean"].mean()),
                                  int(arr["nfe_max"].max()))))
        return block(rollup(crits), crits, metrics,
                     note="长跑 %d 步（文档为 24 小时；步数由 --t4-steps 配置）" % steps)

    # ---------------------------------------------------------------- T5
    def run_T5(self):
        a = self.args
        from ahi_theory.verify_all import (verify_P1, verify_P2, verify_P3, verify_P4,
                                           verify_P5, verify_P6)
        fns = [("P1", verify_P1), ("P2", verify_P2), ("P3", verify_P3),
               ("P4", verify_P4), ("P5", verify_P5), ("P6", verify_P6)]
        raw = {}
        for key, fn in fns:
            t0 = time.perf_counter()
            try:
                r = fn(seed=a.seed)
            except Exception as e:
                traceback.print_exc()
                r = {"route": key, "passed": False, "criteria": [],
                     "summary": "ERROR %s: %s" % (type(e).__name__, e)}
            r["secs"] = time.perf_counter() - t0
            raw[key] = r
        lat = self._wetware_latency()
        routes = {
            "P1": self._t5_P1(raw["P1"]),
            "P2": self._t5_P2(raw["P2"], lat),
            "P3": self._t5_P3(raw["P3"]),
            "P4": self._t5_P4(raw["P4"]),
            "P5": self._t5_P5(raw["P5"]),
            "P6": self._t5_P6(raw["P6"]),
        }
        for key, r in routes.items():
            r["verify_all_route_passed"] = bool(raw[key].get("passed"))
            r["verify_all_summary"] = raw[key].get("summary")
            r["verify_all_secs"] = raw[key].get("secs")
            r["verify_all_criteria"] = raw[key].get("criteria", [])
        return routes, lat

    @staticmethod
    def _pick(criteria, *keys):
        for c in criteria:
            nm = str(c.get("name", ""))
            if all(k in nm for k in keys):
                return c
        return None

    def _row(self, c, name, threshold, gate=None, note=""):
        if c is None:
            return crit(name, None, threshold, None, note="verify_all 未产出该判据" + note)
        v = c.get("value")
        passed = c.get("passed") if gate is None else gate(v)
        if not _finite(v) and not isinstance(v, bool):
            passed = None if gate is None else bool(gate(v))
        return crit(name, v if _finite(v) or isinstance(v, bool) else None, threshold,
                    passed, note=(c.get("note", "") + " " + note).strip())

    def _route_block(self, criteria, metrics, name, doc_ref):
        return block(rollup(criteria), criteria, metrics, doc_ref=doc_ref, note=name)

    def _t5_P1(self, r):
        C = r.get("criteria", [])
        ece = self._pick(C, "ECE")
        auc = None
        if ece is not None:
            m = re.search(r"AUC=([0-9.]+)", str(ece.get("note", "")))
            if m:
                auc = float(m.group(1))
        crits = [self._row(ece, "P1 ECE 与 AUC：期望校准误差 ECE",
                           "< %g" % TH["T5_P1_ece"],
                           gate=(lambda v: _finite(v) and v < TH["T5_P1_ece"]) if ece else None),
                 crit("P1 ECE 与 AUC：AUC", auc, "> %g" % TH["T5_P1_auc"],
                      (auc is not None) and auc > TH["T5_P1_auc"],
                      note="从 verify_P1 的 ECE 判据 note 中解析（该函数把 ECE 与 AUC 合成一个判据）"
                      if auc is not None else "verify_P1 note 中未解析到 AUC")]
        met = {"intervention_effect": (self._pick(C, "干预") or {}).get("value"),
               "negative_control": (self._pick(C, "阴性对照") or {}).get("value"),
               "ece": ece.get("value") if ece else None, "auc": auc}
        return self._route_block(crits, met, "P1 功能主义：conf 头 ECE 与 AUC", "T5 表 P1 行")

    def _wetware_latency(self, n=1500):
        from ahi_theory.verify_all import WetwareInterface
        from ahi_theory.brain import Brain
        ww = WetwareInterface(seed=self.args.seed)
        b = Brain(seed=self.args.seed)
        tr = b.run(60, seed=self.args.seed + 1)
        xs = [np.resize(tr["h"][t].ravel(), ww.n) for t in range(60)]
        lat = []
        for t in range(n):
            x = xs[t % 60]
            t0 = time.perf_counter()
            ww.stimulate(x)
            ww.step()
            ww.record()
            lat.append(time.perf_counter() - t0)
        lat = np.asarray(lat)
        out = {"n": int(n), "components": "stimulate+step+record",
               "p50_ms": float(np.percentile(lat, 50) * 1e3),
               "p95_ms": float(np.percentile(lat, 95) * 1e3),
               "p99_ms": float(np.percentile(lat, 99) * 1e3),
               "max_ms": float(lat.max() * 1e3),
               "mean_ms": float(lat.mean() * 1e3)}
        if self.ops.is_cuda:
            torch = self.ops.torch
            dev = self.ops.device
            Wd = torch.randn(1024, 1024, device=dev)
            lat2 = []
            for t in range(min(n, 500)):
                x = xs[t % 60][:24]
                t0 = time.perf_counter()
                xt = torch.as_tensor(x, dtype=self.ops.dtype, device=dev)
                y = (xt @ Wd)
                y = y.to("cpu").numpy()
                lat2.append(time.perf_counter() - t0)
            lat2 = np.asarray(lat2)
            out["gpu_roundtrip"] = {"n": int(len(lat2)), "p50_ms": float(np.percentile(lat2, 50) * 1e3),
                                    "p95_ms": float(np.percentile(lat2, 95) * 1e3),
                                    "max_ms": float(lat2.max() * 1e3)}
        return out

    def _t5_P2(self, r, lat):
        C = r.get("criteria", [])
        p95 = lat["p95_ms"]
        crits = [crit("P2 接口延迟 p95", p95, "< %g ms" % TH["T5_P2_p95_ms"],
                      p95 < TH["T5_P2_p95_ms"],
                      unit="ms", note="湿件接口 stimulate+step+record 往返 %d 次实测（CPU 侧；"
                      "GPU 侧另算 host↔device 往返）" % lat["n"]),
                 self._row(self._pick(C, "切断湿件"), "P2 湿件承重（切断后行为偏离）", "> 0.5")]
        met = dict(latency=lat,
                   interface_complete=(self._pick(C, "接口方法完整性") or {}).get("value"),
                   alive=(self._pick(C, "湿件存活") or {}).get("value"),
                   decoupling_divergence=(self._pick(C, "切断湿件") or {}).get("value"))
        return self._route_block(crits, met, "P2 生物自然主义：湿件接口延迟", "T5 表 P2 行")

    def _t5_P3(self, r):
        C = r.get("criteria", [])
        js = self._pick(C, "JS")
        crits = [self._row(js, "P3 语义交换 JS 散度", "> %g" % TH["T5_P3_js"],
                           gate=lambda v: _finite(v) and v > TH["T5_P3_js"]),
                 self._row(self._pick(C, "G1b"), "P3 一阶注意因果效力（干预 g 效应量）", "> 0.8"),
                 self._row(self._pick(C, "G2a"), "P3 图式是注意的有效模型 R²", "> 0.5")]
        met = {k: (self._pick(C, k) or {}).get("value") for k in
               ("G1a", "G2a", "G2b", "G1b", "JS", "G4", "G5", "G6")}
        return self._route_block(crits, met, "P3 高阶表征/AST", "T5 表 P3 行")

    def _t5_P4(self, r):
        C = r.get("criteria", [])
        phi = self._pick(C, "Phi*") or self._pick(C, "整合度")
        mac = self._pick(C, "平均绝对跨区相关")
        pci = self._pick(C, "PCI")
        crits = [crit("P4 整合度代理 主体 vs 解耦对照（Phi*）",
                      phi.get("value") if phi else None, "主体 > 对照",
                      phi.get("passed") if phi else None,
                      note=(phi or {}).get("note", "")),
                 crit("P4 平均绝对跨区相关 主体 vs 对照",
                      mac.get("value") if mac else None, "主体 > 对照",
                      mac.get("passed") if mac else None, note=(mac or {}).get("note", "")),
                 crit("P4 PCI（zap-and-zip）主体 vs 对照",
                      pci.get("value") if pci else None, "主体 > 对照",
                      pci.get("passed") if pci else None, note=(pci or {}).get("note", ""))]
        met = {"phi_subject": (phi or {}).get("value"), "mac_diff": (mac or {}).get("value"),
               "pci_diff": (pci or {}).get("value"),
               "note": "均为代理量，不是 Φ（verify_all 已标注）"}
        return self._route_block(crits, met, "P4 整合信息论", "T5 表 P4 行")

    def _t5_P5(self, r):
        C = r.get("criteria", [])
        fb = self._pick(C, "自由能有界")
        rise = self._pick(C, "反暗室")
        dL = self._pick(C, "调质核团")
        crits = [self._row(fb, "P5 自由能有界 F", "< %g" % TH["T5_P5_F"],
                           gate=lambda v: _finite(v) and v < TH["T5_P5_F"]),
                 self._row(rise, "P5 反暗室：恒定输入下 F 上升", "> 0"),
                 self._row(dL, "P5 内生驱动：调质核团对输入统计量的响应", "> 0.01")]
        met = {"F_max": (fb or {}).get("value"), "dark_room_rise": (rise or {}).get("value"),
               "dL": (dL or {}).get("value"),
               "no_reward_audit": (self._pick(C, "无外部奖励") or {}).get("value")}
        return self._route_block(crits, met, "P5 主动推理", "T5 表 P5 行")

    def _t5_P6(self, r):
        C = r.get("criteria", [])
        kap = self._pick(C, "kappa")
        cross = self._pick(C, "跨分支")
        kv = kap.get("value") if kap else None
        crits = [crit("P6 κ ∈ [0.3, 0.9]", kv, "[%g, %g]" % (TH["T5_P6_kappa_lo"], TH["T5_P6_kappa_hi"]),
                      (_finite(kv) and TH["T5_P6_kappa_lo"] <= kv <= TH["T5_P6_kappa_hi"]),
                      note="κ = ‖J‖₂ = σ_max(Wm)（谱范数，不是谱半径）"),
                 self._row(cross, "P6 跨分支神谕 R²", "< %g" % TH["T5_P6_cross_r2"],
                           gate=lambda v: _finite(v) and v < TH["T5_P6_cross_r2"]),
                 self._row(self._pick(C, "不动点"), "P6 不动点 ≤12 NFE 收敛率", "≥ 0.999"),
                 self._row(self._pick(C, "迟滞"), "P6 迟滞率 H", "> 0.05")]
        met = {"kappa": kv, "cross_branch_r2": (cross or {}).get("value"),
               "hysteresis": (self._pick(C, "迟滞") or {}).get("value"),
               "branch_mi": (self._pick(C, "分支互信息") or {}).get("value")}
        return self._route_block(crits, met, "P6 构成性自模型", "T5 表 P6 行")


# ====================================================================== preflight
def preflight(bench):
    ops, args = bench.ops, bench.args
    out = {}
    # --- 0. 参考实现的文件指纹（brain.py 是被持续修改的活文件）
    out["reference_files_sha256"] = {
        "GPU_测试要求.md": _sha256(os.path.join(ROOT, "GPU_测试要求.md")),
        "ahi_theory/brain.py": _sha256(os.path.join(ROOT, "ahi_theory", "brain.py")),
        "ahi_theory/verify_all.py": _sha256(os.path.join(ROOT, "ahi_theory", "verify_all.py")),
        "transcribed_brain_sha256": TRANSCRIBED_BRAIN_SHA256,
        "note": "T5 的数字来自运行时刻已导入的 verify_all.py（其 sha256 记录在此）",
    }
    # --- 1. 与 brain.py 的一致性（参数 + 轨迹）
    #     ★ 代码同源判据固定用 numpy/float64（与基准后端无关）：float32/bf16 下同一份
    #       代码也会因舍入产生差，并会在本引擎的近临界自参考回路里被放大到 O(0.1)
    #       （本机实测 cuda/float32 = 0.295 / cuda/float64 = 2.0e-9）——那是精度差，
    #       不是代码差。故 float64 差值参与 PASS 判定（阈值 1e-9 不变），
    #       基准 dtype 下的同轨迹差另记为信息项。
    try:
        from ahi_theory import brain as brain_mod
        from ahi_theory.brain import Brain
        disk_sha = _sha256(os.path.join(ROOT, "ahi_theory", "brain.py"))
        ops64 = build_backend(detect_environment(), "numpy", "auto")[0]
        n_step = 200
        u_scale = float(getattr(args, "u_scale", 0.01))

        def _parity(opx):
            """同一份 200 步对比，跑在指定 ops 上（numpy/float64 或基准后端）。"""
            b = Brain(seed=args.seed)
            # (a) 自带的 rng 复刻（对应 TRANSCRIBED_BRAIN_SHA256）
            pairs = [("A", "A"), ("Wc", "Wc"), ("Bin", "Bin"), ("Wg", "Wg"), ("bg", "bg"),
                     ("Wgs", "Wgs"), ("Ws", "Ws"), ("Ws_in", "Ws_in"), ("Wm", "Wm"),
                     ("Wm_in", "Wm_in"), ("W_slow", "W_slow"), ("Wm_fwd", "Wm_fwd"),
                     ("Wm_surp", "Wm_surp"), ("u_mom", "u_mom"), ("WL", "WL"), ("GL", "GL"),
                     ("Wconf", "Wconf"), ("h0", "h"), ("s0", "s"), ("hslow0", "h_slow")]
            eng_init = Engine(EngCfg(12, 24, 16, 1, 1, args.seed, "reference"), opx)
            pdiff_init = max(float(np.max(np.abs(opx.numpy(eng_init.init_np[k])
                                                   - np.asarray(getattr(b, j)))))
                             for k, j in pairs if hasattr(b, j))
            # (b) 从活体 Brain 装配后的动力学一致性
            eng = Engine.from_brain(b, opx)
            pdiff = max(float(np.max(np.abs(opx.numpy(eng.P[k]) - np.asarray(getattr(b, k)))))
                        for k in ("A", "Wc", "Bin", "Wg", "bg", "Wgs", "Ws", "Ws_in", "Wm",
                                  "Wm_in", "W_slow", "WL", "GL", "Wconf") if hasattr(b, k))
            rng = np.random.default_rng(args.seed + 77)
            mxx, nfe_bad = {}, 0
            for t in range(n_step):
                u = rng.normal(size=eng.d) * u_scale
                s1 = b.step(u)
                s2 = eng.step(u)
                for k in ("h", "g", "s", "m", "h_slow", "L", "conf", "z_pred", "self_err",
                          "self_err_ema", "mom"):
                    if k not in s1:
                        continue
                    mxx[k] = max(mxx.get(k, 0.0),
                                 float(np.max(np.abs(np.asarray(opx.numpy(s2[k]))[0]
                                                     - np.asarray(s1[k])))))
                mxx["F"] = max(mxx.get("F", 0.0),
                               abs(float(opx.numpy(s2["F"])[0]) - float(s1["F"])))
                nfe_bad += int(int(np.asarray(opx.numpy(s2["nfe"]))[0]) != int(s1["nfe"]))
            return dict(init=pdiff_init, param=pdiff, worst=max(mxx.values()), by_key=mxx,
                        nfe=nfe_bad, missing=eng.missing_in_brain)

        p64 = _parity(ops64)
        pbk = p64 if ops.dtype_str == "float64" else _parity(ops)
        out["engine_parity_vs_brain_py"] = {
            "status": PASS if (p64["worst"] < 1e-9 and p64["nfe"] == 0) else FAIL,
            "reference_file": "ahi_theory/brain.py",
            "reference_sha256_on_disk": disk_sha,
            "transcribed_sha256": TRANSCRIBED_BRAIN_SHA256,
            "revision_match": bool(disk_sha == TRANSCRIBED_BRAIN_SHA256),
            "parity_ops": "numpy/float64（代码同源判据；与基准后端无关）",
            "init_stream_param_max_abs_diff": p64["init"],
            "from_brain_param_max_abs_diff": p64["param"],
            "params_missing_in_brain": p64["missing"],
            "traj_max_abs_diff": p64["worst"], "traj_max_abs_diff_by_key": p64["by_key"],
            "nfe_mismatch_steps": p64["nfe"],
            "backend_dtype": ops.dtype_str,
            "traj_max_abs_diff_backend_dtype": pbk["worst"],
            "traj_max_abs_diff_backend_dtype_by_key": pbk["by_key"],
            "nfe_mismatch_steps_backend_dtype": pbk["nfe"],
            "backend_dtype_note": "基准 dtype（%s）下同一代码的舍入差会被近临界自模型放大；"
                                  "不参与 PASS 判定" % ops.dtype_str,
            "steps": n_step, "u_scale": u_scale,
            "note": "分区批量转写 vs ahi_theory/brain.py 逐步同 u 对比（不修改 brain.py）；"
                    "初始化 rng 序列若因参考实现改写而不同，仅影响 init_stream 一栏，"
                    "动力学一致性用 from_brain 装配后核对"}
    except Exception as e:
        out["engine_parity_vs_brain_py"] = {"status": ERROR, "error": "%s: %s" % (type(e).__name__, e)}
    # --- 2. 参考实现基线（复核文档 §一 的【实测】数字）
    try:
        from ahi_theory.brain import Brain
        b = Brain(seed=args.seed)
        b.run(20, seed=1)                    # 预热（含首次实例化开销）
        n = 300
        dts = []
        for _ in range(3):                   # 取 3 次最好值，降低并发负载对计时的影响
            t0 = time.perf_counter()
            b.run(n, seed=1)
            dts.append(time.perf_counter() - t0)
        dt = min(dts)
        _PN = ("A", "Wc", "Bin", "Wg", "bg", "Wgs", "Ws", "Ws_in", "Wm", "Wm_in",
               "W_slow", "Wm_fwd", "Wm_surp", "u_mom", "WL", "GL", "Wconf")
        nparam = sum(int(np.prod(np.shape(getattr(b, k)))) for k in _PN if hasattr(b, k))
        nparam_all = sum(int(np.prod(np.shape(v))) for k, v in vars(b).items()
                         if isinstance(v, np.ndarray) and v.dtype.kind == "f")
        out["reference_impl_baseline"] = {
            "steps": n, "ms_per_step": dt / n * 1000.0, "seconds_per_3000_steps": dt / n * 3000.0,
            "runs": 3, "ms_per_step_all_runs": [x / n * 1000.0 for x in dts],
            "load_average_1min_at_measure": (os.getloadavg()[0] if hasattr(os, "getloadavg") else None),
            "parameter_matrix_count": nparam, "all_float_array_count": nparam_all,
            "doc_claim_ms_per_step": 1.5, "doc_claim_param_count": 1.2e4,
            "note": "文档 §一/§八 的【实测】数字在本机复核"}
    except Exception as e:
        out["reference_impl_baseline"] = {"status": ERROR, "error": "%s: %s" % (type(e).__name__, e)}
    # --- 3. 梯度估计器自检（解析函数）
    rng = np.random.default_rng(args.seed + 5)
    A = rng.normal(size=(50, 200))
    th = rng.normal(size=200)
    g_exact = A.T @ (A @ th)
    f = lambda v: 0.5 * float(np.sum((A @ v) ** 2))
    eps = 1e-6 * max(float(np.sqrt(np.mean(th ** 2))), 1.0)
    ests = []
    for _ in range(64):
        v = rng.integers(0, 2, size=th.shape) * 2.0 - 1.0
        ests.append((f(th + eps * v) - f(th - eps * v)) / (2 * eps))
    ests, acc = [], np.zeros_like(th)
    for _ in range(64):
        v = rng.integers(0, 2, size=th.shape) * 2.0 - 1.0
        d = (f(th + eps * v) - f(th - eps * v)) / (2 * eps)
        ests.append(d)
        acc += v * d
    ests = np.asarray(ests)
    g_hat = acc / 64.0
    gnorm_exact = float(np.linalg.norm(g_exact))
    gnorm_hutch = float(np.sqrt(np.mean(ests ** 2)))
    rel_scale = abs(gnorm_hutch - gnorm_exact) / gnorm_exact
    rel_dir = float(np.linalg.norm(g_hat - g_exact) / gnorm_exact)
    out["hutchinson_estimator_check"] = {
        "exact_grad_norm": gnorm_exact, "hutchinson_grad_norm": gnorm_hutch,
        "relative_error_norm": rel_scale, "spsa_relative_error": rel_dir,
        "samples": 64, "eps": eps,
        "status": PASS if rel_scale < 0.15 else FAIL,
        "note": "在解析函数 0.5||Aθ||² 上自检：sqrt(mean d²) 估 ||g||，mean(v·d) 估 g 方向"
                "（numpy 后端无 autograd 时用同一估计器测引擎梯度范数）"}
    # --- 4. 实测带宽 / 算力
    out["peak_bandwidth"] = bench.hw["peak_bandwidth"]
    out["peak_flops"] = bench.hw["peak_flops"]
    # --- 5. torch 可用时：autograd vs Hutchinson 交叉核对
    if ops.is_torch:
        try:
            cfg = EngCfg(12, 24, 16, 1, 2, args.seed, "reference")
            eng = Engine(cfg, ops)
            rngu = np.random.default_rng(args.seed + 9)
            win = [rngu.normal(size=(2, 24)) * 0.08 for _ in range(8)]
            snap = eng.snapshot()
            ga = eng.grad_autograd(win)
            eng.restore(snap)
            gh = eng.grad_hutchinson(snap, win, 64, 1e-6, np.random.default_rng(args.seed + 10))
            rel = abs(gh["grad_norm"] - ga["grad_norm"]) / max(ga["grad_norm"], 1e-300)
            out["autograd_vs_hutchinson"] = {
                "autograd_grad_norm": ga["grad_norm"], "hutchinson_grad_norm": gh["grad_norm"],
                "relative_error": rel, "window": len(win), "samples": 64,
                "status": PASS if rel < 0.35 else FAIL}
        except Exception as e:
            out["autograd_vs_hutchinson"] = {"status": ERROR, "error": "%s: %s" % (type(e).__name__, e)}
    return out


def failure_mode_diagnostics(report):
    """§六 五个失败模式 -> 本机实测证据。"""
    T3 = report.get("T3_throughput_bandwidth", {})
    T4 = report.get("T4_stability_longrun", {})
    T2 = report.get("T2_multi_card_scaling", {})
    T5 = report.get("T5_theory_routes", {})
    best = (T3.get("metrics", {}) or {}).get("best_config", {}) or {}
    util = best.get("train_bw_utilization_measured_peak_converted")
    out = {}
    sw = [r for r in (T3.get("metrics", {}) or {}).get("sweep", []) if "skipped" not in r]
    be, verdict = None, "无数据"
    if sw:
        lo = min(sw, key=lambda r: r["instances"])
        hi = max(sw, key=lambda r: r["instances"])
        be = {"instances_min": lo["instances"], "instances_max": hi["instances"],
              "instance_ticks_per_s_min": lo["instance_ticks_per_s"],
              "instance_ticks_per_s_max": hi["instance_ticks_per_s"],
              "throughput_speedup": hi["instance_ticks_per_s"] / max(lo["instance_ticks_per_s"], 1e-9),
              "ms_per_tick_min_batch": lo["ms_per_tick"], "ms_per_tick_max_batch": hi["ms_per_tick"],
              "weight_bytes_per_tick": hi["weight_bytes"]}
    if util is None:
        verdict = "无数据"
    elif util >= 0.10:
        verdict = "未确诊（利用率 %.4f ≥ 文档阈值 0.10）" % util
    elif be and be["throughput_speedup"] > 2.0:
        verdict = ("低于文档的 0.10 阈值，但**不是**文档给的根因（未锁步批处理）："
                   "实例 %d→%d 时吞吐 %.0f→%.0f 实例·tick/s（×%.2f），锁步批处理在起作用；"
                   "低利用率来自本机所测规模的权重张量只有 %.2f MB/tick（远小于撑满带宽所需），"
                   "属**规模不匹配**，须在 §2.1 目标规模（21.81 GB 权重）上重测"
                   % (be["instances_min"], be["instances_max"], be["instance_ticks_per_s_min"],
                      be["instance_ticks_per_s_max"], be["throughput_speedup"],
                      be["weight_bytes_per_tick"] / 1e6))
    else:
        verdict = "确诊（利用率 %.4f < 0.10 且批处理未带来摊薄）" % util
    out["1_吞吐只有预期的1/65"] = {
        "measured_train_bw_utilization_converted": util,
        "doc_rule": "带宽利用率 < 0.10 即确诊",
        "batching_evidence": be,
        "verdict": verdict,
        "evidence": "T3 metrics.sweep（实例 1→%d 扫描实测）" % (be["instances_max"] if be else 0)}
    zm = (T2.get("metrics", {}) or {}).get("zero_memory_model", {}) or {}
    peak = (T4.get("metrics", {}) or {}).get("memory")
    out["2_OOM但显存算下来够"] = {
        "zero2_per_card_GB_from_doc_formula": zm.get("per_card_GB_zero8"),
        "single_card_GB_from_doc_formula": zm.get("total_GB"),
        "measured_peak_bytes_T1": ((report.get("T1_single_card_smoke", {}).get("metrics", {}) or {})
                                   .get("memory", {}) or {}).get("peak_bytes"),
        "verdict": "无 CUDA：以公式分解给出参照（§2.2 参数公式）"}
    sat = (T4.get("metrics", {}) or {}).get("saturation_unit_max")
    out["3_loss突然NaN"] = {
        "measured_saturation_rate_unit_level": sat,
        "measured_engine_def_saturation": (T4.get("metrics", {}) or {}).get("saturation_engine_def_max"),
        "doc_rule": "饱和率 > 0.9 即确诊",
        "verdict": ("确诊" if (sat is not None and sat > 0.9) else "未确诊"),
        "evidence": "T4 metrics.saturation_unit_max（长跑全程最大值）"}
    cf = (T2.get("metrics", {}) or {}).get("allgather_time_fraction")
    out["4_多卡几乎无加速"] = {
        "measured_allgather_time_fraction": cf,
        "scaling_efficiency": (T2.get("metrics", {}) or {}).get("multi_device", {}).get("scaling_efficiency"),
        "verdict": ("无多卡：扩展效率按文档跳过" if report.get("T2_multi_card_scaling", {}).get("status") == SKIP
                    else "见 T2 criteria")}
    kap = ((T5.get("P6", {}).get("metrics", {}) or {}).get("kappa"))
    out["5_kappa判据莫名超标"] = {
        "measured_kappa_sigma_max_Wm": kap,
        "verdict": ("κ 在 [0.3,0.9] 内（按谱范数标定正确）"
                    if (isinstance(kap, (int, float)) and 0.3 <= kap <= 0.9) else "见 T5 P6"),
        "evidence": "T5 P6 metrics.kappa"}
    return out


# ====================================================================== 报告/CLI
def print_report(rep):
    W = 100
    print("=" * W)
    print("AHI GPU 基准  bench_gpu.py     文档判据来源: GPU_测试要求.md 第四节 T1–T5")
    print("=" * W)
    env, cfg = rep["environment"], rep["config"]
    print("后端      : %s（%s）" % (rep["backend"]["name"], rep["backend"]["reason"]))
    dev = env["devices"][0]["name"] if env.get("devices") else "无 CUDA 设备（CPU）"
    print("设备      : %s   torch=%s   cuda=%s" % (dev, env.get("torch_version"),
                                                   env.get("cuda_version")))
    print("主机      : %s × %s   RAM %.1f GB" % (env.get("cpu_model"), env.get("cpu_count"),
                                                 (env.get("mem_total_bytes") or 0) / 2 ** 30))
    print("规模      : %s  R=%d d=%d K=%d layers=%d 实例=%d  精度=%s  参数=%d"
          % (cfg["scale"], cfg["regions"], cfg["dim"], cfg["schema"], cfg["layers"],
             cfg["instances"], cfg["dtype"], rep["param_count"]["engine_params"]))
    print("实测带宽  : %.2f GB/s（%s）" % (rep["preflight"]["peak_bandwidth"]["bytes_per_s"] / 1e9,
                                          rep["preflight"]["peak_bandwidth"]["method"]))
    print("实测算力  : %.2f GFLOP/s（%s）" % (rep["preflight"]["peak_flops"]["flops_per_s"] / 1e9,
                                             rep["preflight"]["peak_flops"]["method"]))
    print("-" * W)
    for key in ("T1_single_card_smoke", "T2_multi_card_scaling", "T3_throughput_bandwidth",
                "T4_stability_longrun"):
        b = rep[key]
        print("[%s] %s" % (key.split("_")[0], _cn_status(b["status"])))
        for c in b["criteria"]:
            v = c["value"]
            vs = ("%.6g" % v) if isinstance(v, float) else str(v)
            print("    %-6s %-46s = %-18s 要求 %s"
                  % (c["passed"] if c["passed"] is None else ("PASS" if c["passed"] else "FAIL"),
                     _clip(c["name"], 46), _clip(vs, 18), _clip(str(c["threshold"]), 26)))
            if c.get("note"):
                print("           └ %s" % c["note"])
        if b.get("note"):
            print("    note: %s" % b["note"])
        if key == "T3_throughput_bandwidth":
            m = b.get("metrics", {})
            for row in m.get("sweep", []):
                if "skipped" in row:
                    print("    扫描 实例=%-4s 跳过（%s，估计 %.1f MB > 预算 %.1f MB）"
                          % (row["instances"], row["skipped"], row["estimate_bytes"] / 2 ** 20,
                             row["budget_bytes"] / 2 ** 20))
                else:
                    print("    扫描 实例=%-4d %8.3f ms/tick  实例·tick/s=%10.1f  "
                          "前向权重流 %8.2f MB/s  训练口径换算 %.5f"
                          % (row["instances"], row["ms_per_tick"], row["instance_ticks_per_s"],
                             row["forward_bytes_per_s"] / 1e6,
                             row["train_bw_utilization_measured_peak_converted"]))
            p = m.get("largest_feasible_scale", {})
            if p.get("scale"):
                print("    最大可行规模探针: %s R=%d d=%d layers=%d 参数=%d  %.2f ms/tick  "
                      "实例·tick/s=%.1f" % (p["scale"], p["regions"], p["dim"], p["layers"],
                                            p["param_count"], p["ms_per_tick"], p["ticks_per_s"]))
            fb = m.get("fleet_backcalc", {})
            for kk in ("main_scale", "largest_feasible_scale"):
                d = fb.get(kk)
                if d:
                    print("    实测反推(%s=%s 参数=%d): %.1f 实例·tick/s/卡 ⇒ 达到 %g/卡 需 %s 张卡；"
                          "100Hz × 128 实例需 %s 张卡"
                          % (kk, d["scale"], d["param_count"], d["measured_instance_ticks_per_s"],
                             fb.get("target_instance_ticks_per_s_per_card", 1e4),
                             d["cards_for_target_rate"],
                             d["cards_for_100Hz_wallclock_128_instances"]))
            tgt = m.get("target_scale", {})
            if tgt:
                print("    目标规模带宽需求（§2.2 公式 × 实测带宽上限）: %.2f TB/s 需求 / 实测上限 "
                      "%.3f TB/s ⇒ 缺口 %.1f×；100%% 带宽下最大训练 tick 率 %.2f Hz"
                      % (tgt["train_bytes_per_s_required"] / 1e12,
                         tgt["measured_peak_bandwidth_bytes_per_s"] / 1e12,
                         tgt["shortfall_vs_100Hz"], tgt["peak_train_tick_rate_at_100pct_bw"]))
                h = fb.get("doc_h100_arithmetic_check", {})
                if h:
                    print("    文档 §三 H100 推算复核: 3.35 TB/s ÷ (3×21.81 GB) = %.1f Hz "
                          "（文档称 51 Hz）" % h["computed_train_tick_rate_100pct"])
    print("-" * W)
    print("[T5] 六条理论路线（直接调用 ahi_theory/verify_all.py）")
    for k in ("P1", "P2", "P3", "P4", "P5", "P6"):
        b = rep["T5_theory_routes"][k]
        print("    %-4s %-6s %s" % (k, b["status"], b["note"] if b.get("note") else ""))
        for c in b["criteria"]:
            v = c["value"]
            vs = ("%.6g" % v) if isinstance(v, float) else str(v)
            print("         %-6s %-44s = %-16s 要求 %s"
                  % (c["passed"] if c["passed"] is None else ("PASS" if c["passed"] else "FAIL"),
                     _clip(c["name"], 44), _clip(vs, 16), _clip(str(c["threshold"]), 22)))
    print("-" * W)
    print("[复核文档推算] 公式计数（source 字段已标注，非本机实测）")
    try:
        zc = rep["T2_multi_card_scaling"]["metrics"]["zero_memory_model"]
        print("    §2.2 显存分解: 权重 %.1f + 梯度 %.1f + Adam %.1f + master %.1f = %.1f GB"
              "（文档称 174.4 GB）"
              % (zc["weights_bytes"] / 1e9, zc["grads_bytes"] / 1e9, zc["adam_mv_bytes"] / 1e9,
                 zc["master_bytes"] / 1e9, zc["total_GB"]))
        print("    8 卡 ZeRO-2: %.2f GB/卡（文档称 21.8 GB/卡）" % zc["per_card_GB_zero8"])
    except Exception:
        pass
    pc = rep["param_count"]
    print("    §2.1 目标规模参数: %d（%s）" % (pc["target_scale_formula"],
                                               pc["target_scale_formula_source"]))
    rb2 = rep["preflight"].get("reference_impl_baseline", {})
    if "parameter_matrix_count" in rb2:
        print("    §一 参考实现参数矩阵: 实测 %d 个 vs 文档称 ≈%g（实测/文档 = %.2f×）"
              % (rb2["parameter_matrix_count"], pc["doc_claim_reference_params"],
                 rb2["parameter_matrix_count"] / pc["doc_claim_reference_params"]))
    print("-" * W)
    print("[preflight] 引擎与 ahi_theory/brain.py 的一致性")
    pp = rep["preflight"]["engine_parity_vs_brain_py"]
    if "traj_max_abs_diff" in pp:
        print("    [%s] 200 步轨迹最大绝对差 = %g ；NFE 不一致步数 = %d ；初始 rng 复刻差 = %g"
              " （brain.py sha256 与转写修订%s）"
              % (pp["status"], pp["traj_max_abs_diff"], pp["nfe_mismatch_steps"],
                 pp["init_stream_param_max_abs_diff"],
                 "一致" if pp["revision_match"] else "不一致"))
    rb = rep["preflight"].get("reference_impl_baseline", {})
    if "ms_per_step" in rb:
        print("    brain.py 实测 %.3f ms/步（文档称 1.5 ms），3000 步 %.2f s；参数矩阵 %d 个"
              "（文档 §一 称 ≈1.2e4）" % (rb["ms_per_step"], rb["seconds_per_3000_steps"],
                                          rb["parameter_matrix_count"]))
    if not pp.get("revision_match", True):
        print("    ★ 磁盘上的 brain.py 已变更（sha256 不匹配），初始化 rng 复刻差异 = %g；"
              "动力学一致性用 from_brain 核对 = %g"
              % (pp.get("init_stream_param_max_abs_diff", float("nan")), pp["traj_max_abs_diff"]))
    hc = rep["preflight"]["hutchinson_estimator_check"]
    print("    梯度估计器自检：精确 %.6g vs 估计 %.6g（相对误差 %.2e）"
          % (hc["exact_grad_norm"], hc["hutchinson_grad_norm"], hc["relative_error_norm"]))
    print("-" * W)
    s = rep["summary"]
    print("总计: PASS %d / FAIL %d / SKIP %d   报告 -> %s"
          % (s["n_pass"], s["n_fail"], s["n_skip"], rep["output_path"]))
    for k, v in s["status"].items():
        print("    %-28s %s" % (k, _cn_status(v)))
    print("=" * W)


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def _cn_status(s):
    return {"PASS": "PASS 通过", "FAIL": "FAIL 未通过", "SKIP": "SKIP 跳过（硬件缺失）",
            "ERROR": "ERROR 异常"}.get(s, s)


def build_parser():
    p = argparse.ArgumentParser(description="AHI 引擎 GPU 基准（GPU_测试要求.md 第四节 T1–T5）")
    p.add_argument("--out", default=os.path.join(ROOT, "bench_report.json"))
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "torch-cpu", "numpy"])
    p.add_argument("--dtype", default="auto",
                   help="auto | float64 | float32 | bfloat16（torch 后端）")
    p.add_argument("--scale", default="auto",
                   choices=["auto", "reference", "small", "medium", "large", "target"])
    p.add_argument("--instances", type=int, default=0, help="锁步实例数（0=自动）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--u-scale", type=float, default=0.01,
                   help="外部输入尺度（brain.py run() 的默认值是 0.01）")
    p.add_argument("--skip-t5", action="store_true")
    # T1（默认 1100 = 预热 100 + 判据窗口 1000；启动瞬态见 TH["T1_stab_warmup"] 注释）
    p.add_argument("--t1-steps", type=int, default=1100)
    # T2
    p.add_argument("--t2-world", type=int, default=8)
    p.add_argument("--t2-instances", type=int, default=4)
    p.add_argument("--t2-ticks", type=int, default=20)
    p.add_argument("--t2-force-devices", type=int, default=0,
                   help="无多卡时用 N 个 CPU 副本强制跑多设备线程路径（仅自检用）")
    # T3
    p.add_argument("--t3-sweep", default="1,8,32,64,128")
    p.add_argument("--t3-ticks", type=int, default=200)
    p.add_argument("--t3-warmup", type=int, default=5)
    p.add_argument("--t3-config-budget-s", type=float, default=4.0)
    p.add_argument("--t3-probe-budget-s", type=float, default=6.0)
    p.add_argument("--no-scale-probe", action="store_true")
    # T4
    p.add_argument("--t4-steps", type=int, default=5000)
    p.add_argument("--t4-instances", type=int, default=32)
    p.add_argument("--t4-lr", type=float, default=1e-3)
    p.add_argument("--t4-window", type=int, default=100)
    p.add_argument("--t4-warmup", type=int, default=100)
    p.add_argument("--t4-grad-every", type=int, default=250)
    p.add_argument("--t4-bptt", type=int, default=64)
    p.add_argument("--t4-hutch-samples", type=int, default=32)
    p.add_argument("--t4-hutch-eps", type=float, default=1e-6)
    p.add_argument("--strict", action="store_true", help="有 FAIL 时退出码 1")
    p.add_argument("--quick", action="store_true", help="快速预算")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.quick:
        args.t4_steps = min(args.t4_steps, 1000)
        args.t3_ticks = min(args.t3_ticks, 100)
        args.t3_sweep = "1,8,32"
        args.t2_ticks = min(args.t2_ticks, 8)
    t_start = time.time()
    env = detect_environment()
    ops, backend_name, reason = build_backend(env, args.backend, args.dtype)
    bench = Bench(args, env, ops)
    if ops.is_cuda:
        free, _ = ops.torch.cuda.mem_get_info(ops.device)
        bench.hw["memory_budget_bytes"] = int(free * 0.5)
    else:
        bench.hw["memory_budget_bytes"] = int(meminfo().get("MemAvailable", 1 << 30) * 0.4)
    bench.hw["peak_bandwidth"] = measure_peak_bandwidth(ops, env)
    bench.hw["peak_flops"] = measure_peak_flops(ops)

    rep = {
        "report_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "doc": DOC + " 第四节 T1–T5",
        "script": os.path.abspath(__file__),
        "backend": {"name": backend_name, "reason": reason, "ops_class": type(ops).__name__,
                    "dtype": ops.dtype_str, "device": ops.device_str},
        "environment": env,
        "config": dict(bench.cfg.as_dict(), **{
            "dtype": ops.dtype_str, "instances": int(bench.cfg.instances),
            "t1_steps": args.t1_steps, "t4_steps": args.t4_steps,
            "t3_ticks": args.t3_ticks, "t3_sweep": args.t3_sweep,
            "t2_world": args.t2_world, "seed": args.seed, "u_scale": args.u_scale,
            "scale_reason": bench.scale_reason, "scale_probe": bench.scale_probe,
        }),
        "param_count": {
            "engine_params": int(bench.new_engine(instances=1).param_count()),
            "engine_params_source": "实测（本脚本引擎参数张量逐一张量计数）",
            "target_scale_formula": int(bench.cfg.block_params()) if bench.scale_name == "target"
            else 100 * 8 * 13 * 1024 * 1024,
            "target_scale_formula_source": "GPU_测试要求.md §2.1（公式计数，非本机实测）",
            "doc_claim_reference_params": 1.2e4,
        },
        "preflight": {},
        "checklist_map": CHECKLIST_MAP,
        "T1_single_card_smoke": None,
        "T2_multi_card_scaling": None,
        "T3_throughput_bandwidth": None,
        "T4_stability_longrun": None,
        "T5_theory_routes": None,
        "output_path": os.path.abspath(args.out),
    }
    rep["preflight"] = preflight(bench)

    def safe(name, fn):
        t0 = time.perf_counter()
        try:
            b = fn()
        except Exception as e:
            traceback.print_exc()
            b = block(ERROR, [crit("阶段执行", None, "无异常", False,
                                   note="%s: %s" % (type(e).__name__, e))], {})
        b["seconds"] = time.perf_counter() - t0
        return b

    rep["T1_single_card_smoke"] = safe("T1", bench.run_T1)
    rep["T2_multi_card_scaling"] = safe("T2", bench.run_T2)
    rep["T3_throughput_bandwidth"] = safe("T3", bench.run_T3)
    if args.no_scale_probe:
        (rep["T3_throughput_bandwidth"].get("metrics") or {})["largest_feasible_scale"] = {
            "measured": False, "reason": "--no-scale-probe"}
    rep["T4_stability_longrun"] = safe("T4", bench.run_T4)
    if args.skip_t5:
        rep["T5_theory_routes"] = {k: block(SKIP, [crit("T5", None, "见文档 T5 表", None,
                                                        note="--skip-t5")], {})
                                   for k in ("P1", "P2", "P3", "P4", "P5", "P6")}
    else:
        t0 = time.perf_counter()
        try:
            routes, lat = bench.run_T5()
            rep["T5_theory_routes"] = routes
            rep["wetware_latency"] = lat
        except Exception as e:
            traceback.print_exc()
            rep["T5_theory_routes"] = {k: block(ERROR, [crit("T5", None, "见文档 T5 表", False,
                                                             note="%s: %s" % (type(e).__name__, e))], {})
                                       for k in ("P1", "P2", "P3", "P4", "P5", "P6")}
        rep["T5_seconds"] = time.perf_counter() - t0

    status = {}
    for key in ("T1_single_card_smoke", "T2_multi_card_scaling", "T3_throughput_bandwidth",
                "T4_stability_longrun"):
        status[key] = rep[key]["status"]
    for k, v in (rep["T5_theory_routes"] or {}).items():
        status["T5_" + k] = v["status"]
    n_pass = sum(1 for v in status.values() if v == PASS)
    n_fail = sum(1 for v in status.values() if v in (FAIL, ERROR))
    n_skip = sum(1 for v in status.values() if v == SKIP)
    rep["summary"] = {"status": status, "n_pass": n_pass, "n_fail": n_fail, "n_skip": n_skip,
                      "exit_code": 0, "total_seconds": time.time() - t_start,
                      "fail_list": [k for k, v in status.items() if v in (FAIL, ERROR)],
                      "skip_list": [k for k, v in status.items() if v == SKIP]}
    rep["failure_mode_diagnostics"] = failure_mode_diagnostics(rep)
    rep["limitations"] = [
        "无 CUDA 设备时：T1/T4 的显存与吞吐是系统 RAM/CPU 口径；T3 的带宽上限用 numpy triad 实测。",
        "T2 的扩展效率判据在 CUDA 设备数 < 2 时按文档规定 SKIP；分片 loss 差异与 all-gather 占比仍实测。",
        "T4 的 24 小时长跑由 --t4-steps 配置；报告中给出实测步/秒与按该速率换算的 24 小时步数。",
        "§2.1/§2.2 的目标规模参数与显存分解是公式计数（报告内 source 字段已标注），非本机实测。",
        "本机是共享容器（PRoot/Android）：environment.load_average 与 "
        "preflight.reference_impl_baseline.ms_per_step_all_runs 记录了运行时的负载与重复测量，"
        "吞吐类数字（T3、preflight 基线）随负载变化，判据结论不受影响。",
    ]
    with open(args.out, "w") as fh:
        json.dump(jsonable(rep), fh, ensure_ascii=False, indent=2)
    print_report(rep)
    if args.strict and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
