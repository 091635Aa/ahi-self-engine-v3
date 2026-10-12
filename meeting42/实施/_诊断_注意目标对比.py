# -*- coding: utf-8 -*-
"""临时诊断2：注意目标的可学性（离线对比多种目标 + 拷贝基线）。

记录一段运行的自态 S_t 与各种"注意"候选量，再用同一套 NLMS 读出离线比较可学性。
"""
import os
import sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(R, "meeting42", "实施"), os.path.join(R, "meeting38", "实施"),
          os.path.join(R, "meeting39", "实施"), os.path.join(R, "meeting31"), R):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from 内核_元自我 import 顶覆盖_综合体M, _符号, _余弦
from 内核_自我回路 import 来源标_自源

句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
      "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]

桥 = 顶覆盖_综合体M(seed=350, 元自我开关=True, 自我开关=True, 学习开关=True,
                    自模型开关=True, 思核开关=True, 顿悟开关=True, 档案seed=310)
桥.钉来源(来源标_自源)
Φ = np.asarray(桥.eng.hdc.Φ_act, dtype=np.float64)
N = 100
Ss, a星s, ês, Gs, uas = [], [], [], [], []
for i in range(N):
    rec = 桥._引擎步(桥.感知向量(桥.codec.编码(句s[i % len(句s)])))
    Ss.append(_符号((rec["模块"] or {}).get("hdc", {}).get("H双极")))
    a星s.append(np.asarray(rec["a星"], dtype=np.float64).ravel())
    ês.append(np.asarray(rec["候选ê"], dtype=np.float64).ravel())
    Gs.append(np.asarray(rec["候选G"], dtype=np.float64).ravel())
    uas.append(np.asarray(rec["候选u_a"], dtype=np.float64))
print("形状：S=%s a星=%s ê=%s G=%s u_a=%s" % (Ss[0].shape, a星s[0].shape, ês[0].shape,
                                              Gs[0].shape, np.asarray(uas[0]).shape))
print("ê 前3拍：", [np.round(e, 3).tolist() for e in ês[:3]])
print("G 前3拍：", [np.round(g, 3).tolist() for g in Gs[:3]])
print("u_a 前1拍范数：", float(np.linalg.norm(np.asarray(uas[0]))))


def 单位(x):
    x = np.asarray(x, dtype=np.float64).ravel()
    n = float(np.linalg.norm(x))
    return x / n if n > 1e-12 else x


def 离线ε(目标s, 键s, 读出="绝对", η=0.1, 窗=20):
    目标s = [np.asarray(t, dtype=np.float64).ravel() for t in 目标s]
    D = 目标s[0].shape[0]
    W = np.zeros((D, D))
    上键 = None
    εs = []
    for t in range(N):
        if 上键 is not None:
            raw = W @ 上键
            Â = _符号(raw) if 读出 == "绝对" else raw
            ε = 1.0 - _余弦(Â, 目标s[t])
            εs.append(ε)
            n2 = float(上键 @ 上键) + 1e-6
            W += (η * ε) * np.outer(目标s[t] - Â, 上键) / n2
        上键 = 键s[t]
    首 = float(np.mean(εs[:窗]))
    末 = float(np.mean(εs[-窗:]))
    return 首, 末


def 拷贝ε(目标s, 窗=20):
    目标s = [np.asarray(t, dtype=np.float64).ravel() for t in 目标s]
    cos = [_余弦(目标s[t], 目标s[t - 1]) for t in range(1, N)]
    return 1.0 - float(np.mean(cos[窗:]))


S键 = [s for s in Ss]
目标集 = {
    "离散注意码 sign(Φ@a星)": [_符号(Φ @ a) for a in a星s],
    "Φ@a星(连续)": [Φ @ a for a in a星s],
    "单位(Φ@a星)": [单位(Φ @ a) for a in a星s],
}
print("=" * 78)
print("%-30s %8s %8s %8s %8s %8s" % ("目标", "首窗ε", "末窗ε", "拷贝ε", "只自态末", "学了吗"))
for 名, 目标s in 目标集.items():
    键s = [Ss[t] * 单位(目标s[t]) for t in range(N)]      # 键 = 自态 ⊛ 当拍注意
    读出 = "绝对" if 名.startswith("离散") else "连续"
    try:
        首, 末 = 离线ε(目标s, 键s, 读出=读出)
        拷 = 拷贝ε(目标s)
        键2 = [Ss[t] for t in range(N)]
        _首2, 末2 = 离线ε(目标s, 键2, 读出=读出)
        print("%-30s %8.4f %8.4f %8.4f %8.4f %8s" % (名, 首, 末, 拷, 末2,
                                                  "是" if 末 < 拷 - 0.02 else "否/弱"))
    except Exception as ex:
        print("%-30s 错误 %r" % (名, ex))
print("=" * 78)