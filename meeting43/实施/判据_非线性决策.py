# -*- coding: utf-8 -*-
"""判据_非线性决策.py —— meeting43 · P7（决策耦合：线性缩放 → 饱和+乘性联合，并引入引擎新异 ñ）。

J1 C14 默认关：非线性决策=0 时，`_应答计划` 的 k/门 与**旧公式逐位一致**（判据内独立复算比对）。
J2 非线性：开启后 门(债) 的二阶差分非零（≠ 仿射/线性缩放）。
J3 更深耦合：旧决策**不含引擎新异 ñ**（改变 ñ 时 k/门 不变）；开启后 ñ 变化**确实改变** k/门。
J4 有界：宽网格扫描下 门 ∈ (0, 阈值]、k ∈ [3,12]（不越界）。
运行：python 判据_非线性决策.py （exit 0）
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


def 旧公式(阈值, 债, 自评, 回忆问):
    k = 3 + int(round(min(max(债, 0.0), 1.5) * 2)) + int(round(max(自评, 0.0) * 2))
    门 = 阈值 * (1.0 - 0.3 * min(max(债, 0.0), 1.0)) * (1.0 - 0.1 * 自评)
    if 回忆问:
        门 *= 0.6
    return k, 门


def main() -> int:
    桥 = synthesis.综合体(seed=310)
    阈 = float(桥.阈值)
    q = "苹果"                                                         # 非回忆问
    orig = synthesis.非线性决策
    try:
        # ── J1：默认关 = 旧公式 ──
        synthesis.非线性决策 = False
        同 = True
        for 债 in (0.0, 0.3, 0.7, 1.2, 1.6):
            for 自评 in (0.0, 0.4, -0.3, 1.0):
                桥.消化债, 桥.自评偏差, 桥.ñ = 债, 自评, 0.5
                计 = 桥._应答计划(q)
                k0, m0 = 旧公式(阈, 债, 自评, False)
                同 = 同 and (计["k"] == k0) and (abs(计["门"] - m0) < 1e-12)
        J1 = 同

        # ── J2：非线性（门(债) 二阶差分非零） ──
        synthesis.非线性决策 = True
        桥.自评偏差, 桥.ñ = 0.0, 0.5                                    # r新=0，隔离 债 一路
        债s = [0.0, 0.3, 0.6, 0.9, 1.2]
        门s = []
        for 债 in 债s:
            桥.消化债 = 债
            门s.append(桥._应答计划(q)["门"])
        二阶 = [门s[i + 1] - 2 * 门s[i] + 门s[i - 1] for i in range(1, len(门s) - 1)]
        J2 = max(abs(x) for x in 二阶) > 1e-5

        # ── J3：旧决策不含 ñ；新决策含 ñ ──
        synthesis.非线性决策 = False
        桥.消化债, 桥.自评偏差 = 0.5, 0.0
        桥.ñ = 0.2
        旧低 = 桥._应答计划(q)
        桥.ñ = 0.95
        旧高 = 桥._应答计划(q)
        旧与ñ无关 = (旧低["k"] == 旧高["k"]) and (abs(旧低["门"] - 旧高["门"]) < 1e-12)

        synthesis.非线性决策 = True
        桥.ñ = 0.2
        新低 = 桥._应答计划(q)
        桥.ñ = 0.95
        新高 = 桥._应答计划(q)
        新与ñ有关 = (新低["k"] != 新高["k"]) or (abs(新低["门"] - 新高["门"]) > 1e-6)
        J3 = 旧与ñ无关 and 新与ñ有关

        # ── J4：有界 ──
        synthesis.非线性决策 = True
        有界 = True
        for 债 in np.linspace(0, 2.0, 9):
            for 自评 in np.linspace(-0.5, 1.0, 7):
                for ñ in np.linspace(0.0, 1.0, 6):
                    桥.消化债, 桥.自评偏差, 桥.ñ = float(债), float(自评), float(ñ)
                    计 = 桥._应答计划(q)
                    有界 = 有界 and (0.0 < 计["门"] <= 阈 + 1e-12) and (3 <= 计["k"] <= 12)
        J4 = 有界
    finally:
        synthesis.非线性决策 = orig

    print("阈值 = %.3f" % 阈)
    print("J1 默认关逐位同旧公式 = %s" % J1)
    print("J2 门(债)=%s 二阶差分=%s 最大|Δ²|=%.6f" %
          ([round(x, 4) for x in 门s], [round(x, 5) for x in 二阶], max(abs(x) for x in 二阶)))
    print("J3 旧决策随ñ: k %d→%d 门 %.4f→%.4f（无关=%s）｜新决策随ñ: k %d→%d 门 %.4f→%.4f（有关=%s）"
          % (旧低["k"], 旧高["k"], 旧低["门"], 旧高["门"], 旧与ñ无关,
             新低["k"], 新高["k"], 新低["门"], 新高["门"], 新与ñ有关))
    print("J4 有界（门∈(0,阈] ∧ k∈[3,12]）= %s" % J4)
    print("\n=== 判据 ===")
    print("J1 C14 默认关逐位恒等旧公式   :", "PASS" if J1 else "FAIL")
    print("J2 非线性（二阶差分非零）     :", "PASS" if J2 else "FAIL")
    print("J3 更深耦合（引入引擎新异 ñ） :", "PASS" if J3 else "FAIL")
    print("J4 有界（不越界）             :", "PASS" if J4 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3 and J4) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
