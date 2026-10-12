# -*- coding: utf-8 -*-
"""判据_语义码.py —— meeting43 · P1+P2 修复判据（可失败；主臂 ∧ sham 对照）。

判据（全部可失败）：
  J1 P2 非空操作    ：主臂 max|Δcos| > 1e-3（旧实现 = 0.000000，必 FAIL）
  J2 语义组回升     ：主臂语义组 Δ > sham 臂语义组 Δ + 0.05
  J3 无关组不被抬   ：主臂语义组 Δ > 主臂无关组 Δ + 0.05
  J4 确定性         ：同 seed 同输入两次精化 → 编码逐位一致
运行：python 判据_语义码.py   （exit 0 = 全部可运行；判据结果见打印与结论行）
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
M28 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "meeting28")
if M28 not in sys.path:
    sys.path.insert(0, M28)

import numpy as np                                                # noqa: E402
import lang_codec                                                 # noqa: E402
from lang_codec import 语言编解码器                                # noqa: E402


def cos(a, b):
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-9)


def _载旧():
    """载入本次修复前的备份 lang_codec（C14 逐位对照用）。"""
    import importlib.util
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "备份_迁移前", "备份_20261011_语言链修复", "meeting28", "lang_codec.py")
    spec = importlib.util.spec_from_file_location("lang_codec_旧", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


对 = [("高兴", "开心"), ("苹果", "水果"), ("猫", "宠物"), ("喜爱", "喜欢")]
无关 = [("高兴", "苹果"), ("开心", "桌子"), ("喜欢", "石头"), ("猫", "桌子")]

教学 = ["苹果就是我的水果", "水果里有苹果", "苹果和水果都是吃的",
        "高兴就是开心", "开心的时候我很高兴", "高兴和开心是一个意思",
        "猫就是我的宠物", "宠物里有猫", "我喜欢猫也喜欢宠物",
        "喜爱就是喜欢", "喜欢和喜爱差不多", "我喜爱我的猫"]

sham语料 = ["苹果桌子石头", "高兴水果吃", "猫喜欢桌子石头", "宠物苹果开心",
            "我我我我我", "的的是是了了", "石头石头石头", "桌子桌子桌子",
            "苹果桌子", "高兴石头", "猫桌子", "喜爱石头"]


def 建(语料):
    c = 语言编解码器(seed=216)
    for t in 语料:
        c.注册段(t)
        c.共现(t)
    return c


def 测(语料):
    c = 建(语料)
    前对 = {k: cos(c.编码(a), c.编码(b)) for k, (a, b) in enumerate(对)}
    前无 = {k: cos(c.编码(a), c.编码(b)) for k, (a, b) in enumerate(无关)}
    前字符 = {ch: c._字符向量(ch).copy() for ch in "苹水高开猫宠喜"}
    c.精化(α=0.3, 轮数=2)
    后对 = {k: cos(c.编码(a), c.编码(b)) for k, (a, b) in enumerate(对)}
    后无 = {k: cos(c.编码(a), c.编码(b)) for k, (a, b) in enumerate(无关)}
    翻转 = sum(int(np.sum(c._字符向量(ch) != 前字符[ch])) for ch in 前字符)
    语义Δ = float(np.mean([后对[k] - 前对[k] for k in range(len(对))]))
    无关Δ = float(np.mean([后无[k] - 前无[k] for k in range(len(无关))]))
    Δmax = max(abs(后对[k] - 前对[k]) for k in range(len(对)))
    return dict(语义Δ=语义Δ, 无关Δ=无关Δ, Δmax=Δmax, 翻转=翻转, 后对=后对, 前对=前对)


def _尺度安全():
    """J5：大异构语料（标准库 json 源码）上，精化不得使段间相似度全局坍缩、也不得让无关查询越过阈值。

    ★这是本次实测漏洞的回归守卫：早期版本在小语料上「通过」，但在 1046 行异构语料上把所有字符
    拉向同一平均场方向 ⇒ 段间均值 cos 0.31→1.00、无关查询越阈 ⇒ 系统瞎猜。"""
    d = os.path.join(os.path.dirname(os.__file__), "json")
    语料 = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py"):
            for ln in open(os.path.join(d, fn), encoding="utf-8", errors="replace"):
                s = ln.strip()
                if 6 <= len(s) <= 120:
                    语料.append(s)
    c = 语言编解码器(seed=216)
    for t in 语料:
        c.注册段(t)
        c.共现(t)
    for _ in range(3):
        c.精化(α=0.3, 轮数=2)
    rng = np.random.default_rng(7)
    样本 = [语料[i] for i in rng.choice(len(语料), 40, replace=False)]
    M = np.stack([c._字袋(t) for t in 样本], 0)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    段间 = float((Mn @ Mn.T)[np.triu_indices(len(样本), 1)].mean())
    qb = c._字袋("番茄炒蛋应该怎么做得好吃？")
    qmax = max(float(qb @ v) / (np.linalg.norm(qb) * np.linalg.norm(v) + 1e-9) for v in M)
    return dict(通过=bool((段间 < 0.75) and (qmax < 0.30)), 段间=段间, 无关max=qmax)


def _加速等价():
    """J6：精化「按句+总−自身项」加速后，与加速前（备份）**符号级等价**，且确有加速。

    数学上 CTX_c = Σ_{ch∈L,ch≠c} v = (Σ_L v) − (c 的项) 等价；仅浮点求和次序变（~1e-16）。
    实测（1046 行异构语料、3 次调用）：字符生效码 0 位翻转、80 样本编码 0/81920 位翻转。"""
    import importlib.util
    import time
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "备份_迁移前", "备份_20261011_语言链修复", "meeting28", "lang_codec.阶段加速前.py")
    spec = importlib.util.spec_from_file_location("lang_codec_加速前", p)
    旧 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(旧)
    d = os.path.join(os.path.dirname(os.__file__), "json")
    语料 = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py"):
            for ln in open(os.path.join(d, fn), encoding="utf-8", errors="replace"):
                s = ln.strip()
                if 6 <= len(s) <= 120:
                    语料.append(s)

    def 跑(mod):
        c = mod.语言编解码器(seed=216)
        for t in 语料:
            c.注册段(t)
            c.共现(t)
        t0 = time.perf_counter()
        for _ in range(3):
            c.精化(α=0.3, 轮数=2)
        return c, time.perf_counter() - t0

    c新, t新 = 跑(lang_codec)
    c旧, t旧 = 跑(旧)
    字符 = sorted(set(c新._语义) & set(c旧._语义))
    码同 = all(np.array_equal(c新._字符向量(ch), c旧._字符向量(ch)) for ch in 字符)
    样本 = [语料[i] for i in np.random.default_rng(3).choice(len(语料), 80, replace=False)]
    编码同 = all(np.array_equal(c新.编码(t), c旧.编码(t)) for t in 样本)
    字袋同 = all(np.array_equal(c新._字袋(t), c旧._字袋(t)) for t in 样本)
    加速 = (t旧 / t新) if t新 > 0 else 0.0
    return dict(码同=码同, 编码同=编码同, 字袋同=字袋同, 加速=加速,
                通过=bool(码同 and 编码同 and 字袋同 and 加速 >= 2.0), 旧秒=t旧, 新秒=t新)


def main() -> int:
    主 = 测(教学)
    sham = 测(sham语料)

    print("\n[主臂] 语义对 Δcos：")
    for k, (a, b) in enumerate(对):
        print("  %s/%s  %.4f -> %.4f  Δ=%+.6f" % (a, b, 主["前对"][k], 主["后对"][k],
                                              主["后对"][k] - 主["前对"][k]))
    print("[主臂] 语义组均值Δ=%+.4f  无关组均值Δ=%+.4f  max|Δcos|=%.6f  符号翻转=%d"
          % (主["语义Δ"], 主["无关Δ"], 主["Δmax"], 主["翻转"]))
    print("[sham] 语义组均值Δ=%+.4f  无关组均值Δ=%+.4f  max|Δcos|=%.6f  符号翻转=%d"
          % (sham["语义Δ"], sham["无关Δ"], sham["Δmax"], sham["翻转"]))

    c1 = 建(教学); c1.精化(α=0.3, 轮数=2)
    c2 = 建(教学); c2.精化(α=0.3, 轮数=2)
    同 = all(np.array_equal(c1.编码(t), c2.编码(t)) for t in 教学)

    # C14：精化**前**必须与旧实现（备份件）逐位一致
    旧 = _载旧()
    c新 = 建(教学)
    c旧 = 旧.语言编解码器(seed=216)
    for t in 教学:
        c旧.注册段(t); c旧.共现(t)
    逐位同旧 = all(np.array_equal(c新.编码(t), c旧.编码(t)) for t in 教学)

    J1 = 主["Δmax"] > 1e-3
    J2 = 主["语义Δ"] > sham["语义Δ"] + 0.05
    J3 = 主["语义Δ"] > 主["无关Δ"] + 0.05
    J4 = bool(同 and 逐位同旧)

    # ── J5 尺度安全（大异构语料）：防"全局坍缩"——本次实测漏洞的回归守卫 ──
    尺 = _尺度安全()
    # ── J6 精化加速：与加速前符号级等价 + 加速比 ──
    加 = _加速等价()

    print("\n=== 判据 ===")
    print("J1 P2 非空操作 (max|Δcos|>1e-3)        :", "PASS" if J1 else "FAIL", "(%.6f)" % 主["Δmax"])
    print("J2 主臂语义回升 > sham + 0.05          :", "PASS" if J2 else "FAIL",
          "(%.4f vs %.4f)" % (主["语义Δ"], sham["语义Δ"]))
    print("J3 主臂语义回升 > 主臂无关 + 0.05      :", "PASS" if J3 else "FAIL",
          "(%.4f vs %.4f)" % (主["语义Δ"], 主["无关Δ"]))
    print("J4 确定性 ∧ 精化前逐位同旧实现(C14)    :", "PASS" if J4 else "FAIL",
          "(同=%s, 同旧=%s)" % (同, 逐位同旧))
    print("J5 大语料不坍缩/不瞎猜                 :", "PASS" if 尺["通过"] else "FAIL",
          "(段间均值=%.4f<0.75, 无关查询max=%.4f<0.30)" % (尺["段间"], 尺["无关max"]))
    print("J6 精化加速(符号级等价 ∧ ≥2×)          :", "PASS" if 加["通过"] else "FAIL",
          "(码同=%s 编码同=%s 字袋同=%s 加速=%.2fx)" % (加["码同"], 加["编码同"], 加["字袋同"], 加["加速"]))
    总 = J1 and J2 and J3 and J4 and 尺["通过"] and 加["通过"]
    print("总判据:", "PASS" if 总 else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
