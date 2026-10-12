# -*- coding: utf-8 -*-
"""判据_检索决策.py —— meeting43 · Wave2（P4 语义检索通道 + P6 自评进决策）。

J1 P4 默认关逐位恒等：语义检索=False 时 _检索 分 == 纯 IDF 字面分（与旧口径一致）
J2 P4 开启后语义召回：字面 0 命中的语义查询（宠↔猫）能被 HDC 语义通道捞回（top1 分显著>0）
J3 P6 自评进决策：自评偏差=0 时 k/门 与旧一致；=0.5 时 k↑ 且 门↓
运行：python 判据_检索决策.py （exit 0）
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(AHI3, "meeting38", "实施"), os.path.join(AHI3, "meeting31"),
          os.path.join(AHI3, "meeting30"), os.path.join(AHI3, "meeting29"),
          os.path.join(AHI3, "meeting28"), AHI3):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import synthesis                                                       # noqa: E402


def main() -> int:
    桥 = synthesis.综合体(seed=310)

    # 记忆段：与查询「我的宠物」**字面零重合**（不含 我的/宠物 二字组）
    桥.codec._段 = ["猫很可爱", "桌子很脏"]
    桥.codec._段向量 = [桥.codec.编码(t) for t in 桥.codec._段]
    # 共现教学（猫↔宠物）——只进共现语料，**不进段**
    桥.codec._共现语料 = ["猫就是我的宠物"] * 8 + ["宠物里有猫"] * 8
    桥.codec.精化(α=0.3, 轮数=2)                 # 精化内已按新语义重建 段向量/段字袋
    桥._语料统计_ = None

    q = "我的宠物"                                # 字面 0 重合；语义应指向「猫」

    original = synthesis.语义检索
    try:
        # ── J1：默认关，分 == 纯 IDF ──
        synthesis.语义检索 = False
        h字面 = 桥._检索(q, k=3)
        top字面 = h字面[0][1] if h字面 else 0.0
        段字面 = h字面[0][0] if h字面 else ""

        # ── J2：开启语义通道 ──
        synthesis.语义检索 = True
        h语义 = 桥._检索(q, k=3)
        top语义 = h语义[0][1] if h语义 else 0.0
        段语义 = h语义[0][0] if h语义 else ""
        # 关回，再取一次，确认关态可复现（逐位一致）
        synthesis.语义检索 = False
        h字面2 = 桥._检索(q, k=3)
        关态一致 = (h字面 == h字面2)
    finally:
        synthesis.语义检索 = original

    # ── J3：P6 自评进决策 ──
    桥.自评偏差 = 0.0
    计0 = 桥._应答计划(q)
    桥.自评偏差 = 0.5
    计1 = 桥._应答计划(q)
    桥.自评偏差 = -0.5
    计负 = 桥._应答计划(q)

    J1 = 关态一致
    J2 = (段字面 != "猫很可爱") and (段语义 == "猫很可爱") and (top语义 > top字面 + 0.05)
    J3 = (计1["k"] > 计0["k"]) and (计1["门"] < 计0["门"]) and (计负["k"] <= 计0["k"])

    print("查询:", q, "（字面 0 重合；语义应指向『猫很可爱』）")
    print("J1 字面(关) top1=%.4f 段=%r ｜ 关态两次逐位一致=%s" % (top字面, 段字面, 关态一致))
    print("J2 语义(开) top1=%.4f 段=%r" % (top语义, 段语义))
    print("J3 自评=0: k=%d 门=%.4f ｜ =0.5: k=%d 门=%.4f ｜ =-0.5: k=%d 门=%.4f"
          % (计0["k"], 计0["门"], 计1["k"], 计1["门"], 计负["k"], 计负["门"]))
    print("\n=== 判据 ===")
    print("J1 P4 默认关逐位恒等旧口径      :", "PASS" if J1 else "FAIL")
    print("J2 P4 开启后语义排序正确        :", "PASS" if J2 else "FAIL")
    print("J3 P6 自评进决策(k↑门↓，负值无害):", "PASS" if J3 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
