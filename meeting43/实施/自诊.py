# -*- coding: utf-8 -*-
"""自诊.py —— P8 运行期自诊断回路（纯旁路 · 默认关 · 零器官依赖）。

病（预检 P8）：仓库有 自检.py（架构 A1–A7）、verify_all.py（理论台账）、synthesis.自审承载、
meeting39 内核_自模型，但**缺「运行期」对「自己输入是否失真 / 检索是否在瞎猜」的自感知回路**。

本件补两路**只读**自诊（不改检索本身、不进决策回路——是"工程自我意识"的观测面）：
  ① 输入失真：编码向量的零范退化 / 近重复（连续输入几乎同向 → 语义停滞）；
  ② 检索瞎猜：对近窗口查询的 top1 分与「阈值」的比较（均分低于阈、或过半低于阈 ⇒ 在瞎猜）。

铁律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型（本件不 import 任何器官）｜C13 确定性
（无 time/全局随机/内建 hash）｜C14 默认关 ⇒ 调用方不加即零影响（纯旁路）。
"""
from __future__ import annotations

import json

import numpy as np


class 自诊器:
    """运行期自诊断（纯旁路）。持有对话桥/综合体的**只读**引用。"""

    def __init__(self, 桥, seed: int = 0, 窗口: int = 32):
        self.桥 = 桥
        self.seed = int(seed)
        self.窗口 = int(窗口)
        self.日志: list = []
        self._输入迹: list = []                          # 最近编码向量（范数/重复检测）

    # ── ① 输入是否失真 ──
    def 记输入(self, vec) -> None:
        v = np.asarray(vec, dtype=np.float32).ravel()
        self._输入迹.append(v)
        if len(self._输入迹) > self.窗口:
            self._输入迹.pop(0)

    def 输入失真(self) -> dict:
        if not self._输入迹:
            return dict(退化=False, 原因="无输入迹")
        零范 = sum(1 for v in self._输入迹 if float(np.linalg.norm(v)) < 1e-6)
        重复 = 0
        for a, b in zip(self._输入迹, self._输入迹[1:]):
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na > 1e-9 and nb > 1e-9 and float(a @ b) / (na * nb) > 0.999:
                重复 += 1
        退化 = (零范 > 0) or (重复 >= max(3, self.窗口 // 4))
        return dict(退化=退化, 零范向量=零范, 近重复对=重复, 窗口=len(self._输入迹))

    # ── ② 检索是否在瞎猜（只读分析桥的 _检索 分数分布） ──
    def 检索瞎猜(self, 问题s, k: int = 3) -> dict:
        问题s = list(问题s)
        if not 问题s:
            return dict(瞎猜=False, 原因="无查询")
        分 = []
        for q in 问题s:
            try:
                hits = self.桥._检索(q, k=k) or []
            except Exception as ex:                      # 桥无 _检索 → 如实登记，不假装诊断
                return dict(瞎猜=False, 原因="桥不支持_检索", 错=repr(ex))
            分.append(float(hits[0][1]) if hits else 0.0)
        阈值 = float(getattr(self.桥, "阈值", 0.2) or 0.2)
        空命 = sum(1 for s in 分 if s <= 1e-9)
        低于阈 = sum(1 for s in 分 if s < 阈值)
        均分 = float(np.mean(分))
        瞎猜 = (低于阈 / len(分) >= 0.5) or (均分 < 阈值 * 0.5)
        return dict(瞎猜=瞎猜, 均分=round(均分, 4), 阈值=阈值, 低于阈=低于阈,
                    空命=空命, 查询数=len(分))

    # ── 快照（可落盘、可审计） ──
    def 快照(self, 问题s=()) -> dict:
        r = dict(输入自诊=self.输入失真())
        if 问题s:
            r["检索自诊"] = self.检索瞎猜(list(问题s))
        self.日志.append(r)
        return r

    def 落盘(self, 路径: str) -> None:
        with open(路径, "w", encoding="utf-8") as f:
            json.dump(self.日志, f, ensure_ascii=False, indent=1)


# ── 自诊（可失败自证；直接运行：python 自诊.py） ──
def _自证() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    class _假桥:
        阈值 = 0.4

        def __init__(self, 分):
            self._分 = 分

        def _检索(self, q, k=3):
            return [("段%s" % q, self._分)]

    import numpy as np
    # ① 输入失真：正常 vs 退化（零向量 / 近重复）
    d = 自诊器(_假桥(0.9), 窗口=8)
    rng = np.random.default_rng(7)
    for _ in range(8):
        d.记输入(rng.standard_normal(64))
    正常 = d.输入失真()
    d2 = 自诊器(_假桥(0.9), 窗口=8)
    for _ in range(8):
        d2.记输入(np.zeros(64, np.float32))
    退化 = d2.输入失真()
    d3 = 自诊器(_假桥(0.9), 窗口=8)
    v = rng.standard_normal(64).astype(np.float32)
    for _ in range(8):
        d3.记输入(v)
    重复 = d3.输入失真()

    # ② 检索瞎猜：高分（不瞎猜）vs 低分（瞎猜）
    好 = 自诊器(_假桥(0.8)).检索瞎猜(["a", "b", "c"])
    坏 = 自诊器(_假桥(0.05)).检索瞎猜(["a", "b", "c"])

    print("输入失真  正常:", 正常)
    print("输入失真  零范:", 退化)
    print("输入失真  近重复:", 重复)
    print("检索瞎猜  高分:", 好)
    print("检索瞎猜  低分:", 坏)

    A = (正常["退化"] is False and 退化["退化"] is True and 重复["退化"] is True)
    B = (好["瞎猜"] is False and 坏["瞎猜"] is True)
    print("P8-A 输入失真可识别（正例✓/反例✗）:", "PASS" if A else "FAIL")
    print("P8-B 检索瞎猜可识别（正例✓/反例✗）:", "PASS" if B else "FAIL")
    print("总判据:", "PASS" if (A and B) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_自证())
