# -*- coding: utf-8 -*-
"""判据_接口保距.py —— meeting43 · P5 判据（可失败；真投影 ∧ sham 对照 ∧ 默认关逐位恒等）。

J1 保距：真随机投影，投影前后两两余弦 Pearson 相关 > 0.8
J2 sham：零阵投影 ⇒ 相关 < 0.2（判不过 = 度量能失败）
J3 默认关逐位恒等旧路径：音频特征/向量/文本 与备份旧实现逐位一致（保幅度 默认 False）
运行：python 判据_接口保距.py （exit 0）
"""
from __future__ import annotations

import importlib.util
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))                       # meeting43/实施
AHI3 = os.path.dirname(os.path.dirname(HERE))                           # AHI3
M38 = os.path.join(AHI3, "meeting38", "实施")
M28 = os.path.join(AHI3, "meeting28")
for p in (M28, M38, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import lang_codec                                                      # noqa: E402
from lang_codec import 语言编解码器                                     # noqa: E402
import 通用接口 as GI                                                   # noqa: E402
from 通用接口 import 统一输入, 保幅度                                    # noqa: E402


def _载旧接口():
    p = os.path.join(AHI3, "备份_迁移前", "备份_20261011_语言链修复",
                     "meeting38", "实施", "通用接口.py")
    spec = importlib.util.spec_from_file_location("通用接口_旧", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    codec = 语言编解码器(seed=216)
    for t in ["苹果就是我的水果", "高兴就是开心"]:
        codec.注册段(t)

    ui = 统一输入(codec, seed=0)
    真 = ui.保距诊断("特征", f=64, 样本=128, seed=1234)

    # sham：零阵投影（把 (特征,64) 投影缓存替换为全零）
    ui2 = 统一输入(codec, seed=0)
    ui2._投影[("特征", 64)] = np.zeros((64, GI.当前D()), dtype=np.float32)
    sham = ui2.保距诊断("特征", f=64, 样本=128, seed=1234)

    # 默认关逐位恒等：与备份旧实现逐位对照
    旧 = _载旧接口()
    ui旧 = 旧.统一输入(codec, seed=0)
    rng = np.random.default_rng(5150)
    feats = [rng.standard_normal(37).astype(np.float32) for _ in range(5)]
    vecs = [rng.standard_normal(GI.当前D()).astype(np.float32) for _ in range(3)]
    同特征 = all(np.array_equal(ui.音频特征(f)["向量"], ui旧.音频特征(f)["向量"]) for f in feats)
    同向量 = all(np.array_equal(ui.向量(v)["向量"], ui旧.向量(v)["向量"]) for v in vecs)
    同文本 = all(np.array_equal(ui.文本(t)["向量"], ui旧.文本(t)["向量"]) for t in ["苹果", "高兴就是开心"])
    保幅度默认关 = (保幅度 is False)

    J1 = 真["相关"] > 0.8
    J2 = sham["相关"] < 0.2
    J3 = bool(同特征 and 同向量 and 同文本 and 保幅度默认关)

    print("真投影保距：", 真)
    print("零阵 sham ：", sham)
    print("默认关逐位恒等旧实现：特征=%s 向量=%s 文本=%s 保幅度=False=%s"
          % (同特征, 同向量, 同文本, 保幅度默认关))
    print("\n=== 判据 ===")
    print("J1 真投影保距(相关>0.8)        :", "PASS" if J1 else "FAIL", "(%.4f)" % 真["相关"])
    print("J2 sham 判不过(相关<0.2)       :", "PASS" if J2 else "FAIL", "(%.4f)" % sham["相关"])
    print("J3 默认关逐位恒等旧路径(C14)   :", "PASS" if J3 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
