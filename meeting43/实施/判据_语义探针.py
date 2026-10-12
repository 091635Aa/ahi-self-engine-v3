# -*- coding: utf-8 -*-
"""判据_语义探针.py —— meeting43 · P1 探针策略（默认关 + 增强项）。

J1 C14 默认关逐位恒等：未设 AHI_语义探针（或设 AHI_旧探针=1）⇒ 探针**逐位 == 旧随机探针**。
J2 开启后确为数据自定：探针行空间 ≈ 已注册段向量的前 16 主方向（子空间重叠 > 0.95），且 ≠ 随机探针。
J3 默认下「意仍进引擎」：精化后 cos(感知(苹果),感知(水果)) > 0.6 且显著高于精化前（语义经 精化→编码→JL 投影入引擎）。
J4 开启后区分度更强：无关对 cos(感知(苹果),感知(高兴)) 在开态显著低于默认态（探针的增量价值）。
运行：python 判据_语义探针.py （exit 0）
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(AHI3, "meeting31"), os.path.join(AHI3, "meeting30"),
          os.path.join(AHI3, "meeting29"), os.path.join(AHI3, "meeting28"), AHI3):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import lang_codec                                                      # noqa: E402
from lang_codec import 语言编解码器                                     # noqa: E402
from dialogue_bridge import 对话桥                                      # noqa: E402

教学 = ["苹果就是我的水果", "水果里有苹果", "苹果和水果都是吃的",
        "高兴就是开心", "开心的时候我很高兴", "高兴和开心是一个意思"]
额外 = ["今天风很轻", "桌子擦干净了", "窗外有一棵树", "水杯放在手边",
        "慢慢翻了一页书", "灯一直亮着", "他坐在对面", "雨点敲着玻璃",
        "天空是蓝色的", "云在慢慢移动", "街道很安静", "远处有钟声",
        "咖啡已经凉了", "笔尖停在纸上", "影子被拉得很长", "页脚卷了起来",
        "纸边有些发黄", "椅背抵着墙", "门缝透进一线光", "走廊没有人"]
对 = [("苹果", "水果"), ("高兴", "开心")]
无关 = ("苹果", "高兴")


def cos(a, b):
    return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-9)


def 建桥(开: bool):
    if 开:
        os.environ["AHI_语义探针"] = "1"
    else:
        os.environ.pop("AHI_语义探针", None)
    os.environ.pop("AHI_旧探针", None)
    桥 = 对话桥(seed=216)
    for t in 教学 + 额外:                    # 注册足够段（≥16）→ 探针 16 行可全来自数据
        桥.codec.注册段(t)
    for t in 教学:                          # 共现语料仍只含语义教学句（不影响额外句的语义）
        桥.codec.共现(t)
    return 桥


def 主方向(段向量s):
    X = np.stack([np.asarray(v, np.float64) for v in 段向量s])
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:16]


def main() -> int:
    # ── J1：默认关 ⇒ 逐位 == 旧随机探针 ──
    桥关 = 建桥(False)
    桥关.感知向量(桥关.codec.编码("苹果"))                     # 触发惰性重建
    逐位同旧 = np.array_equal(桥关.探针, 桥关._随机探针)

    # ── J2：开启 ⇒ 数据自定（子空间重叠）+ ≠ 随机 ──
    桥开 = 建桥(True)
    桥开.感知向量(桥开.codec.编码("苹果"))
    Vt = 主方向(桥开.codec._段向量)
    P = np.asarray(桥开.探针, np.float64)
    重叠 = float(np.linalg.norm(P @ Vt.T, ord="fro") ** 2 / 16.0)
    异于随机 = not np.allclose(P, 桥开._随机探针)

    # ── J3/J4：语义是否进 16 维感觉；区分度 ──
    def 感知cos(桥, a, b):
        return cos(桥.感知向量(桥.codec.编码(a)), 桥.感知向量(桥.codec.编码(b)))

    # 默认态：精化前后
    桥0 = 建桥(False)
    前 = {k: 感知cos(桥0, a, b) for k, (a, b) in enumerate(对)}
    桥0.codec.精化(α=0.3, 轮数=2)
    后 = {k: 感知cos(桥0, a, b) for k, (a, b) in enumerate(对)}
    无默 = 感知cos(桥0, *无关)

    桥1 = 建桥(True)
    桥1.codec.精化(α=0.3, 轮数=2)
    无开 = 感知cos(桥1, *无关)

    J1 = bool(逐位同旧)
    J2 = bool((重叠 > 0.95) and 异于随机)
    J3 = bool(后[0] > 0.6 and 后[0] > 前[0] + 0.05 and 后[1] > 前[1] + 0.05)
    J4 = bool(无开 < 无默 - 0.2)

    print("J1 默认探针逐位同随机 =", 逐位同旧)
    print("J2 开态子空间重叠 = %.4f ｜ 异于随机 = %s" % (重叠, 异于随机))
    print("J3 默认态 感知cos：苹果/水果 %.4f→%.4f ｜ 高兴/开心 %.4f→%.4f"
          % (前[0], 后[0], 前[1], 后[1]))
    print("J4 无关对（苹果/高兴）感知cos：默认 %.4f ｜ 开态 %.4f" % (无默, 无开))
    print("\n=== 判据 ===")
    print("J1 C14 默认关逐位恒等随机探针 :", "PASS" if J1 else "FAIL")
    print("J2 开态确为数据自定（主方向） :", "PASS" if J2 else "FAIL")
    print("J3 默认下意仍进引擎           :", "PASS" if J3 else "FAIL")
    print("J4 开态区分度更强             :", "PASS" if J4 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3 and J4) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
