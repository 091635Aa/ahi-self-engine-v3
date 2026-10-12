# -*- coding: utf-8 -*-
"""临时诊断3：W_self 回读 ε_σ 的轨迹与身份敏感度（连续注意版）。"""
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


def 跑(锚=None, n=100, 档案seed=310):
    桥 = 顶覆盖_综合体M(seed=350, 元自我开关=True, 自我开关=True, 学习开关=True,
                        自模型开关=True, 思核开关=True, 顿悟开关=True, 档案seed=档案seed)
    if 锚 is not None:
        桥.设身份锚(锚)
    桥.钉来源(来源标_自源)
    εs, cosW, 序 = [], [], []
    Às = []
    for i in range(n):
        rec = 桥._引擎步(桥.感知向量(桥.codec.编码(句s[i % len(句s)])))
        序.append(int(rec["a星序号"]))
        À = _符号(桥.自我核.迹[-1]["À"])
        Às.append(À)
        εs.append(桥.元自我核.ε_σ史[-1]["ε_σ"])
        Ä = 桥.自我核.a_seal * À
        Ŝ = _符号(桥.自我核.W_self @ Ä)
        cosW.append(_余弦(À, Ŝ))
    return 桥, εs, cosW, 序, Às


桥, εs, cosW, 序, Às = 跑()
print("=" * 78)
print("诊断3：ε_σ 轨迹 / 身份敏感度（n=%d，自源）" % len(εs))
print("ε_σ 首8：", [round(x, 4) for x in εs[:8]])
print("ε_σ 末8：", [round(x, 4) for x in εs[-8:]])
print("ε_σ 均值=%.4f 标准差=%.4f 唯一值数=%d" % (float(np.mean(εs)), float(np.std(εs)),
                                            len(set(round(x, 4) for x in εs))))
print("cos(À,Ŝ) 均值=%.4f" % float(np.mean(cosW)))
# À 的跨拍相关性（看 À 是否几乎常量）
if len(Às) > 2:
    cc = [_余弦(Às[i], Às[i - 1]) for i in range(1, len(Às))]
    print("À 相邻 cos：均值=%.4f 中位=%.4f" % (float(np.mean(cc)), float(np.median(cc))))
    print("À 对首拍 cos：均值=%.4f" % float(np.mean([_余弦(Às[i], Às[0]) for i in range(len(Às))])))
print("a星序号分布：", {k: 序.count(k) for k in sorted(set(序))})
print("W_self 范数=%.4f" % float(np.linalg.norm(桥.自我核.W_self)))
print("=" * 78)

# 身份敏感度
桥2, εs2, cosW2, 序2, _ = 跑(锚=_双极(np.random.default_rng(99991), 1024))
print("换 a_seal：ε_σ 均值=%.4f（原 %.4f）｜ 动作序首12 原=%s 新=%s"
      % (float(np.mean(εs2)), float(np.mean(εs)), 序[:12], 序2[:12]))
print("动作分布 原=%s" % {k: 序.count(k) for k in sorted(set(序))})
print("动作分布 新=%s" % {k: 序2.count(k) for k in sorted(set(序2))})
tvd = 0.5 * sum(abs(序.count(k) / len(序) - 序2.count(k) / len(序2))
                for k in set(序) | set(序2))
print("TVD=%.4f" % tvd)