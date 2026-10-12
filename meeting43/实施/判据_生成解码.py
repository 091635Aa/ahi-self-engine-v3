# -*- coding: utf-8 -*-
"""判据_生成解码.py —— meeting43 · P3（`解码` 接进活动链）+ P4（HDC 现场生成路）。

J1 C14 默认关：解码通道/生成路 未开时，`诊断_解码调用`==0，且 `_检索`/`_答问` 两次逐位一致。
J2 P3 解码真在活动链上：解码通道开 ⇒ `诊断_解码调用`>0，且「码→文本」命中的语义段被抬到 top1。
J3 P4 生成非检索：生成路开 ⇒ 回复比关闭时**多出一句**，该句**不属于任何注册记忆段**（=生成而非检索）。
J4 P4 守卫可失败：未训练（空 lm_d）⇒ `_生成应答` 返回 None（不硬编、不编造）。
运行：python 判据_生成解码.py （exit 0）
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

LM语料 = ["我的宠物是一只猫。", "我的宠物很可爱。", "我的宠物喜欢吃鱼。",
          "我的宠物爱睡觉。", "宠物很可爱。", "猫很可爱。", "猫喜欢吃鱼。",
          "猫爱睡觉。", "宠物需要照顾。", "我喜欢我的宠物。"]


def 建语料(段s, 共现):
    桥 = synthesis.综合体(seed=310)
    桥.codec._段 = list(段s)
    桥.codec._段向量 = [桥.codec.编码(t) for t in 段s]
    桥.codec._共现语料 = list(共现)
    桥.codec.精化(α=0.3, 轮数=2)
    桥._语料统计_ = None
    return 桥


def main() -> int:
    # ── J1：默认关（C14） ──
    桥1 = 建语料(["我的宠物很可爱", "桌子很脏"], [])
    for _ in range(3):
        桥1.lm_d.学句("我的宠物很可爱。")
    桥1._语料统计_ = None
    q3 = "我的宠物"
    r1 = 桥1._检索(q3, k=3)
    r2 = 桥1._检索(q3, k=3)
    a_off1 = 桥1._答问(q3)
    a_off2 = 桥1._答问(q3)
    J1 = (桥1.诊断_解码调用 == 0) and (r1 == r2) and (a_off1 == a_off2)

    # ── J2：P3 解码接进活动链 ──
    桥2 = 建语料(["猫很可爱", "桌子很脏"], ["猫就是我的宠物"] * 8 + ["宠物里有猫"] * 8)
    q2 = "我的宠物"
    原 = synthesis.语义检索, synthesis.解码通道
    try:
        synthesis.语义检索 = False
        synthesis.解码通道 = False
        h关 = 桥2._检索(q2, k=3)
        c关 = 桥2.诊断_解码调用
        synthesis.解码通道 = True
        h开 = 桥2._检索(q2, k=3)
        c开 = 桥2.诊断_解码调用
    finally:
        synthesis.语义检索, synthesis.解码通道 = 原
    J2 = (c关 == 0) and (c开 >= 1) and (h开[0][0] == "猫很可爱") and (h开[0][1] > h关[0][1])

    # ── J3：P4 生成路产出「非检索」新句 ──
    桥3 = 建语料(["我的宠物很可爱", "桌子很脏"], [])
    for s in LM语料:
        桥3.lm_d.学句(s)
    桥3._语料统计_ = None
    原2 = synthesis.生成路
    try:
        synthesis.生成路 = False
        a无 = 桥3._答问(q3)
        synthesis.生成路 = True
        a有 = 桥3._答问(q3)
        补 = 桥3._生成应答("我的宠物很可爱", q3)
    finally:
        synthesis.生成路 = 原2
    补新颖 = (补 is not None) and (补 not in 桥3.codec._段) and (补.strip() not in 桥3.lm_d.句集)
    J3 = (a有 != a无) and 补新颖 and (补 in a有)

    # ── J4：守卫可失败（未训练） ──
    空 = 建语料(["猫很可爱"], [])
    空.lm_d.计数1.clear()
    空.lm_d.计数2.clear()
    空.lm_d.计数3.clear()
    空.lm_d.计数4.clear()
    J4 = (空._生成应答("猫很可爱", "我的宠物") is None)

    print("J1 默认关：诊断_解码调用=%d 检索两次一致=%s 答问两次一致=%s"
          % (桥1.诊断_解码调用, r1 == r2, a_off1 == a_off2))
    print("J2 解码：关调用=%d 开调用=%d ｜ 关top1=%r 开top1=%r" %
          (c关, c开, h关[0][0], h开[0][0]))
    print("J3 生成：关答复=%r" % a无)
    print("       开答复=%r" % a有)
    print("       生成句=%r 新颖(∉段集∧∉句集)=%s" % (补, 补新颖))
    print("J4 守卫：空模型→%r 为 None=%s" % (空._生成应答("猫很可爱", "我的宠物"), J4))

    print("\n=== 判据 ===")
    print("J1 C14 默认关逐位恒等        :", "PASS" if J1 else "FAIL")
    print("J2 P3 解码真在活动链上        :", "PASS" if J2 else "FAIL")
    print("J3 P4 生成非检索（新颖句）    :", "PASS" if J3 else "FAIL")
    print("J4 P4 守卫可失败（不编造）    :", "PASS" if J4 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3 and J4) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
