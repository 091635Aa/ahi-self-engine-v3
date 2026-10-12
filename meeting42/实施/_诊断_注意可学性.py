# -*- coding: utf-8 -*-
"""临时诊断：注意目标的可学性（a星离散码 vs ê 连续方向）+ W_self 回读的身份敏感度。"""
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

from 内核_元自我 import 顶覆盖_综合体M, _符号, _余弦, _双极
from 内核_自我回路 import 来源标_自源

句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
      "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]


def 跑(档案seed=310, 锚=None, n=80):
    桥 = 顶覆盖_综合体M(seed=350, 元自我开关=True, 自我开关=True, 学习开关=True,
                        自模型开关=True, 思核开关=True, 顿悟开关=True, 档案seed=档案seed)
    if 锚 is not None:
        桥.设身份锚(锚)
    桥.钉来源(来源标_自源)
    Φ = np.asarray(桥.eng.hdc.Φ_act, dtype=np.float64)
    A星码 = []
    Aê方向 = []
    序 = []
    for i in range(n):
        rec = 桥._引擎步(桥.感知向量(桥.codec.编码(句s[i % len(句s)])))
        序.append(int(rec["a星序号"]))
        A星码.append(_符号(Φ @ np.asarray(rec["a星"], dtype=np.float64)))
        ê = np.asarray(rec["候选ê"], dtype=np.float64).ravel()
        ua = np.asarray(rec["候选u_a"], dtype=np.float64)
        if ua.ndim == 1:
            ua = ua[None, :]
        w = np.maximum(0.0, float(ê.mean()) - ê)
        if float(w.sum()) <= 1e-12:
            v = np.asarray(rec["a星"], dtype=np.float64)
        else:
            v = (ua * w[:ua.shape[0], None]).sum(0)
        nv = float(np.linalg.norm(v))
        Aê方向.append(v / nv if nv > 1e-12 else np.zeros(v.shape[0]))
    return 桥, 序, A星码, Aê方向


def 分离度(序列, 跳=0):
    序列 = [s for s in 序列 if float(np.linalg.norm(s)) > 1e-9]
    if len(序列) < 2:
        return None
    cos = [_余弦(序列[i], 序列[i - 1]) for i in range(1, len(序列))]
    return dict(相邻cos均值=round(float(np.mean(cos)), 4),
                相邻cos中位=round(float(np.median(cos)), 4),
                相同方向率=round(float(np.mean([1.0 if c > 0.999 else 0.0 for c in cos])), 3))


for 名, 造 in (("离散码 sign(Φ@a星)", lambda r, a: a), ("ê连续方向", lambda r, a: r)):
    pass

桥, 序, 码s, 向s = 跑()
print("=" * 78)
print("诊断1：注意目标可学性（n=%d，自源）" % len(序))
print("a星序号 分布：", {k: 序.count(k) for k in sorted(set(序))})
print("动作更换率：%.3f" % (sum(1 for i in range(1, len(序)) if 序[i] != 序[i - 1]) / (len(序) - 1)))
print("离散注意码  相邻cos：", 分离度(码s))
print("ê连续注意方向 相邻cos：", 分离度(向s))
进 = 桥.注意图式
εs = 进.ε序列()
print("注意图式(离散) ε_att 首5=%s 末5=%s 均值=%.4f" % ([round(x, 3) for x in εs[:5]],
                                              [round(x, 3) for x in εs[-5:]], float(np.mean(εs))))
核 = 桥.元自我核
print("回读：读回次数=%d 空读=%d ε_σ最近=%.4f 均值=%.4f"
      % (核.回读次数, 核.空读次数, 核.ε_σ史[-1]["ε_σ"], float(np.mean(核.ε序列()))))
print("W_self 范数=%.4f" % float(np.linalg.norm(桥.自我核.W_self)))
print("=" * 78)

# 诊断2：换 a_seal 后 ε_σ 是否变
sealA = 桥.自我核.a_seal.copy()
rng = np.random.default_rng(99991)
sealB = _双极(rng, len(sealA))
桥.设身份锚(sealB)
εB = []
for i in range(30):
    rec = 桥._引擎步(桥.感知向量(桥.codec.编码(句s[i % len(句s)])))
    εB.append(桥.元自我核.ε_σ史[-1]["ε_σ"])
print("诊断2：换 a_seal 后 ε_σ 末5=%s（均值 %.4f）" % ([round(x, 3) for x in εB[-5:]],
                                              float(np.mean(εB))))