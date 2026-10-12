# -*- coding: utf-8 -*-
"""判据_元自我.py —— meeting42 验收：第七层元自我的**可失败判据套件**（M1–M9）。

设计出处：`meeting42/07_终稿_工程自我意识优化.md` §5（M1–M9 表）；口径同 `00_共同简报.md`
（§五 双集 seed 0–5 / 6–11；主臂过 ∧ sham/对照臂不过 ⇒ 才记 PASS；**不放宽阈值凑 PASS**）。

判据（每条带对照臂/sham，**能真的失败**）：
  M1 注意图式判别力  熟悉 ε_att < 陌生 ∧ < 拷贝基线；sham=随机注意方向 ⇒ 判别消失
  M2 W_self 回读有效  自指意外 ε_σ < 拷贝基线（上一拍 À 预测） ∧ < 零 W sham
  M3 身份反事实(行为级)  只换 a_seal ⇒ 动作分布 TVD ≥ τ(0.05)；对照臂(关回读)换档案逐位不变
  M4 完整持久化  真·新进程载入 ⇒ 整我指纹逐位同；异档案种子不同 ∧ 篡改拒载
  M5 外显合规  有内态 ⇒ 半句(第一人称/可溯源) ∧ 无元话语；无内态 ∧ 元话语 ⇒ 空（不追加）
  M6 默认关逐位恒等  元自我关 ⇒ 与六层基链外化面逐位同（C14/D1）
  M7 C13 确定性  跨 PYTHONHASHSEED 两进程 ⇒ 链路特征码逐位同
  M8 C1 审计  新件禁词命中 0（tokenize 剥注释字符串后扫代码结构）；故意写禁词 ⇒ 必抓到
  M9 学习≠记忆不回退  新层增迹/生长 ⇒ 既有记忆库 codec._段 条数 0 变

纪律：纯 numpy、零梯度、确定性；只在测试里显式加载内核；**不改任何既有 .py**。
出口编码自愈（GBK 假 FAIL 处置）。用法：`python 判据_元自我.py`（`--快` 只跑判据集）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))                       # meeting42/实施
_AHI3 = os.path.dirname(os.path.dirname(_HERE))                         # AHI3 根
for _p in (_HERE, os.path.join(_AHI3, "meeting38", "实施"),
           os.path.join(_AHI3, "meeting39", "实施"),
           os.path.join(_AHI3, "meeting40", "实施"),
           os.path.join(_AHI3, "meeting31"), _AHI3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from 内核_元自我 import 顶覆盖_综合体M, 造桥_元自我, 元自我外显, 净化, 驱动_σ  # noqa: E402
from 内核_自模型 import ETA_R                                            # noqa: E402
import 持久化_整我                                                        # noqa: E402
import 内核宿主                                                          # noqa: E402

双集 = dict(判据集=[0, 1, 2, 3, 4, 5], 留出集=[6, 7, 8, 9, 10, 11])
τ_TVD = 0.05                    # ★M3 预登记门（专家02 §4 J1；本轮不因未达而放宽）

熟悉句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。",
          "我慢慢翻了一页书。", "灯一直亮着。", "他坐在对面没说话。", "雨点敲着玻璃。"]
陌生句s = ["电压基准在第 7 号引脚上。", "把三号阀门顺时针旋半圈。", "云母片沿对角线切开。",
          "索引键值落在二级缓存里。", "光谱线在低频端展宽了。", "齿轮啮合间隙取负公差。",
          "试剂的当量数需要复核。", "波函数在势阱边界处衰减。"]


# ────────────────────────── 通用工具 ──────────────────────────
def _单位(x) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _符号(x) -> np.ndarray:
    v = np.sign(np.asarray(x, dtype=np.float64))
    v[v == 0.0] = 1.0
    return v


def _余弦(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0.0 and nb > 0.0 else 0.0


def _末窗(xs, k: int = 8) -> float:
    return float(np.mean(xs[-k:])) if xs else float("nan")


def 外化(桥) -> dict:
    """外化面（元自我旁挂不得改动）：说出去的话 / 账 / 引擎态 / 决策态。"""
    return dict(
        拍=int(桥.拍), 消化债=float(桥.消化债), 饱足=float(桥.饱足),
        回复台账=[dict(号=d["号"], 问=d["问"], 响应=d["响应"], 延迟拍=d.get("延迟拍"))
                 for d in 桥.回复台账],
        账本=[list(x) for x in 桥.账本],
        e序列=[float(x) for x in 桥.eng.e序列],
        段数=len(桥.codec._段),
        a_prev=桥.a_prev.tolist(), i_prev=int(桥.i_prev),
        prev_c=(None if 桥.prev_c is None else int(桥.prev_c)),
        对话流=[list(x) for x in 桥.对话流],
    )


def 找差异(a, b, 前缀=""):
    出 = []
    if type(a) is not type(b):
        return ["%s: 类型 %s≠%s" % (前缀, type(a).__name__, type(b).__name__)]
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            出 += 找差异(a.get(k), b.get(k), "%s.%s" % (前缀, k))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            出.append("%s: 长度 %d≠%d" % (前缀, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            出 += 找差异(x, y, "%s[%d]" % (前缀, i))
    elif a != b:
        出.append("%s: %r≠%r" % (前缀, a, b))
    return 出


def 特征码(桥) -> str:
    return hashlib.sha256(json.dumps(外化(桥), ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def 造关(seed: int, **kw):
    """造一个**元自我关**的七层桥（= 六层基链；用于采集/对照）。"""
    return 造桥_元自我(seed=int(seed), 档案seed=310, 元自我开关=False, **kw)


# ────────────────────────── M1：注意图式判别力（双集） ──────────────────────────
def _注意流(桥, 句s, n: int) -> list:
    """跑引擎 n 拍，采集 `(键k, 注意A)`——与 `内核_元自我._元自我旁挂` 同式（测试内自采，不改内核）。"""
    出 = []
    for i in range(int(n)):
        u = 桥.感知向量(桥.codec.编码(句s[i % len(句s)]))
        rec = 桥._引擎步(u)
        h = (rec.get("模块") or {}).get("hdc") or {}
        S = h.get("H双极")
        a星 = rec.get("a星")
        if S is None or a星 is None:
            continue
        Φ = np.asarray(桥.eng.hdc.Φ_act, dtype=np.float64)
        A = _单位(Φ @ np.asarray(a星, dtype=np.float64))
        if float(np.linalg.norm(A)) > 1e-9:
            出.append((_符号(S) * A, A))
    return 出


def _训图式(对s, 更新: bool = True, η0: float = ETA_R):
    W = np.zeros((对s[0][1].shape[0], 对s[0][1].shape[0]), dtype=np.float64)
    上 = None
    for k, A in 对s:
        if 上 is not None:
            Â = W @ 上
            ε = 1.0 - _余弦(Â, A)
            if 更新:
                n2 = float(上 @ 上) + 1e-6
                W += (η0 * ε) * np.outer(A - Â, 上) / n2
        上 = k
    return W


def _评图式(W, 对s) -> list:
    出, 上 = [], None
    for k, A in 对s:
        if 上 is not None:
            出.append(1.0 - _余弦(W @ 上, A))
        上 = k
    return 出


def _拷贝基线(对s) -> list:
    """拷贝基线：预测"下一拍注意 = 本拍注意"。"""
    出 = []
    for i in range(1, len(对s)):
        出.append(1.0 - _余弦(对s[i - 1][1], 对s[i][1]))
    return 出


def M1(seed: int) -> dict:
    n = 32
    桥 = 造关(400 + int(seed))
    熟 = _注意流(桥, 熟悉句s, n)
    桥b = 造关(400 + int(seed))
    陌 = _注意流(桥b, 陌生句s, n)
    h = len(熟) // 2
    熟训, 熟测 = 熟[:h], 熟[h:]
    陌训, 陌测 = 陌[:h], 陌[h:]
    # 主臂：熟悉训练 → 熟悉/陌生 测（冻结）
    W = _训图式(熟训)
    ε_熟 = _末窗(_评图式(W, 熟测))
    ε_陌 = _末窗(_评图式(W, 陌测))
    ε_拷 = _末窗(_拷贝基线(熟测))
    # sham：注意方向换成随机单位向量（键不变）——图式学不到任何可迁移结构 ⇒ 判别应消失
    rng = np.random.default_rng(2026 + int(seed))
    熟sh = [(k, _单位(rng.normal(size=A.shape[0]))) for k, A in 熟]
    陌sh = [(k, _单位(rng.normal(size=A.shape[0]))) for k, A in 陌]
    Ws = _训图式(熟sh[:h])
    ε_熟sh = _末窗(_评图式(Ws, 熟sh[h:]))
    ε_陌sh = _末窗(_评图式(Ws, 陌sh[h:]))
    判别_主 = ε_陌 - ε_熟
    判别_sham = ε_陌sh - ε_熟sh
    return dict(套="M1注意图式判别力", seed=int(seed), n对=len(熟),
                 ε_熟=round(ε_熟, 4), ε_陌=round(ε_陌, 4), ε_拷贝=round(ε_拷, 4),
                 ε_熟sham=round(ε_熟sh, 4), ε_陌sham=round(ε_陌sh, 4),
                 _熟悉优于陌生=bool(ε_熟 < ε_陌),
                 _熟悉优于拷贝=bool(ε_熟 < ε_拷),
                 _sham判别消失=bool(判别_sham <= 0.5 * 判别_主 if 判别_主 > 0 else True),
                 _判别主=round(判别_主, 4), _判别sham=round(判别_sham, 4))


# ────────────────────────── M2：W_self 回读有效（双集） ──────────────────────────
def M2(seed: int, 步数: int = 40) -> dict:
    桥 = 造桥_元自我(seed=500 + int(seed), 档案seed=310, 元自我开关=True,
                    注意开关=False, 回读通道=True)
    桥.钉来源("自源")
    for i in range(步数):
        桥.步(熟悉句s[i % len(熟悉句s)])
    εσ = list(桥.元自我核.ε序列())
    Às = [np.asarray(d["À"], dtype=np.float64) for d in 桥.自我核.迹]
    拷 = [1.0 - _余弦(Às[i - 1], Às[i]) for i in range(1, len(Às))]
    零W = [1.0 - _余弦(À, _符号(np.zeros_like(À))) for À in Às]      # W=0 ⇒ Ŝ=sign(0)=ones
    m_σ, m_拷, m_零 = _末窗(εσ), _末窗(拷), _末窗(零W)
    return dict(套="M2W_self回读有效", seed=int(seed),
                ε_σ=round(m_σ, 4), 拷贝基线=round(m_拷, 4), 零W_sham=round(m_零, 4),
                回读次数=桥.元自我核.回读次数, 空读次数=桥.元自我核.空读次数,
                _优于拷贝=bool(m_σ < m_拷), _优于零W=bool(m_σ < m_零),
                _ε_σ唯一值数=len(set(round(x, 6) for x in εσ)))


# ────────────────────────── M3：身份反事实（行为级，双集） ──────────────────────────
def _动作序列(桥, 句s, n: int) -> list:
    出 = []
    for i in range(int(n)):
        u = 桥.感知向量(桥.codec.编码(句s[i % len(句s)]))
        出.append(int(桥._引擎步(u)["a星序号"]))
    return 出


def _TVD(a: list, b: list) -> float:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    ka = set(ca) | set(cb)
    return float(0.5 * sum(abs(ca[k] / max(len(a), 1) - cb[k] / max(len(b), 1)) for k in ka))


def M3(seed: int, 步数: int = 40) -> dict:
    n = 步数
    甲 = 造桥_元自我(seed=600 + int(seed), 档案seed=310, 元自我开关=True,
                    注意开关=False, 回读通道=True)
    甲.钉来源("自源")
    锚0 = np.asarray(甲.自我核.a_seal, dtype=np.float64).copy()
    乙 = 造桥_元自我(seed=600 + int(seed), 档案seed=310, 元自我开关=True,
                    注意开关=False, 回读通道=True)
    乙.钉来源("自源")
    锚1 = np.where(np.random.default_rng(9999 + int(seed)).random(锚0.shape[0]) < 0.5, -1.0, 1.0)
    乙.设身份锚(锚1)
    序甲 = _动作序列(甲, 熟悉句s, n)
    序乙 = _动作序列(乙, 熟悉句s, n)
    同位 = None
    if int(seed) == 0:                                          # 同位复现只在首个种子上验（省算力）
        甲2 = 造桥_元自我(seed=600 + int(seed), 档案seed=310, 元自我开关=True,
                        注意开关=False, 回读通道=True)
        甲2.钉来源("自源")
        同位 = bool(_动作序列(甲2, 熟悉句s, n) == 序甲)
    tvd = _TVD(序甲, 序乙)
    return dict(套="M3身份反事实", seed=int(seed),
                TVD=round(tvd, 4), τ=τ_TVD, 锚cos=round(_余弦(锚0, 锚1), 4),
                动作桶=sorted(set(序甲)), 同位复现逐位同=同位,
                _达门=bool(tvd >= τ_TVD))


def M3_对照臂(seed: int, 步数: int = 40) -> dict:
    """对照臂：**关回读通道** ⇒ 换 a_seal 不应改行为（证明"不接入就是装饰"）。"""
    n = 步数
    def 跑():
        桥 = 造桥_元自我(seed=680 + int(seed), 档案seed=310, 元自我开关=True,
                        注意开关=False, 回读通道=False)
        桥.钉来源("自源")
        return 桥
    甲 = 跑()
    锚0 = np.asarray(甲.自我核.a_seal, dtype=np.float64).copy()
    乙 = 跑()
    锚1 = np.where(np.random.default_rng(9999 + int(seed)).random(锚0.shape[0]) < 0.5, -1.0, 1.0)
    乙.设身份锚(锚1)
    序甲 = _动作序列(甲, 熟悉句s, n)
    序乙 = _动作序列(乙, 熟悉句s, n)
    tvd = _TVD(序甲, 序乙)
    return dict(套="M3对照臂(关回读)", seed=int(seed), TVD=round(tvd, 4),
                _逐位不变=bool(序甲 == 序乙))


# ────────────────────────── M4：完整持久化（含真·新进程） ──────────────────────────
def M4() -> dict:
    import shutil
    import tempfile
    目录 = tempfile.mkdtemp(prefix="meeting42_M4_")
    路径 = os.path.join(目录, "我.npz")
    try:
        桥 = 造桥_元自我(seed=360, 档案seed=310)
        桥.钉来源("自源")
        for i in range(40):
            桥.步(熟悉句s[i % len(熟悉句s)])
        f1 = 持久化_整我.整我指纹(桥)
        清 = 持久化_整我.存整我(路径, 桥)
        f_file = str(清.get("payload_sha256"))
        # ★真·新进程载入（子进程）⇒ 指纹
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--载入指纹", 路径],
                           capture_output=True, text=True, encoding="utf-8", env=env,
                           cwd=_HERE, timeout=1800)
        f子 = None
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("FINGERPRINT:"):
                f子 = ln.split(":", 1)[1].strip()
        # 异档案种子 ⇒ 不同
        桥3 = 造桥_元自我(seed=360, 档案seed=3719)
        异 = (持久化_整我.整我指纹(桥3) != f1)
        # 篡改 ⇒ 拒载
        状 = 持久化_整我.载整我(路径)
        坏 = dict(状)
        k0 = next(k for k, v in 坏.items() if hasattr(v, "shape") and k.endswith("a_seal"))
        tmp = 路径 + ".坏"
        np.savez_compressed(tmp, **{k: (np.asarray(v) + 1.0 if k == k0 else np.asarray(v))
                                    for k, v in 坏.items() if hasattr(v, "shape")})
        shutil.copyfile(路径 + ".manifest.json", tmp + ".manifest.json")
        拒 = False
        try:
            持久化_整我.载整我(tmp)
        except 持久化_整我.整我损坏:
            拒 = True
        跨进程同 = (f子 is not None and f子 == f1 and f1 == f_file)
        return dict(套="M4完整持久化", 指纹=f1[:24], 子进程指纹=(None if f子 is None else f子[:24]),
                     _跨进程逐位同=bool(跨进程同), _异档案不同=bool(异), _篡改拒载=bool(拒),
                    子进程stderr=(r.stderr or "")[-300:] if f子 is None else "")
    finally:
        shutil.rmtree(目录, ignore_errors=True)


# ────────────────────────── M5：外显合规 ──────────────────────────
def M5() -> dict:
    桥 = 造桥_元自我(seed=520, 档案seed=310, 注意开关=False, 回读通道=True)
    桥.钉来源("自源")
    有s, 无s = [], []
    for i in range(120):
        桥.步(熟悉句s[i % len(熟悉句s)])
        r = 元自我外显(桥)
        (有s if r["有内态"] else 无s).append(r)
    样例 = 有s[0] if 有s else None
    第一人称 = bool(样例 and ("我" in str(样例["半句"])))
    可溯源 = bool(样例 and 样例["溯源"] is not None)
    无元话语 = bool(样例 and 净化(样例["半句"]) == 样例["半句"] and 样例["半句"])
    新桥 = 造桥_元自我(seed=521, 档案seed=310)
    空 = 元自我外显(新桥)
    不追加 = (空["半句"] is None and 空["有内态"] is False and 空["溯源"] is None)
    过滤 = (净化("我的内部状态是 1024 维向量") == "")
    return dict(套="M5外显合规", 有内态条数=len(有s), 无内态条数=len(无s),
                样例=(None if 样例 is None else dict(半句=样例["半句"], 触发=样例["触发"],
                                                     溯源=str(样例["溯源"]))),
                _第一人称=第一人称, _可溯源=可溯源, _无元话语=无元话语,
                _无内态不追加=bool(不追加), _元话语过滤=bool(过滤))


# ────────────────────────── M6：默认关逐位恒等（C14/D1） ──────────────────────────
def M6() -> dict:
    句s = 熟悉句s * 3
    n = 24
    基 = 内核宿主.造桥(seed=318, 档案seed=310)
    关 = 造桥_元自我(seed=318, 档案seed=310, 元自我开关=False)
    for i in range(n):
        基.步(句s[i % len(句s)])
        关.步(句s[i % len(句s)])
    差 = 找差异(外化(基), 外化(关))
    return dict(套="M6默认关逐位恒等", 差异条数=len(差), 差异前5=差[:5],
                _逐位恒等=bool(len(差) == 0),
                基类=type(基).__name__, 关类=type(关).__name__)


# ────────────────────────── M7：C13 确定性（跨 PYTHONHASHSEED） ──────────────────────────
def _跑指纹() -> str:
    桥 = 造桥_元自我(seed=700, 档案seed=310)
    桥.钉来源("自源")
    for i in range(24):
        桥.步(熟悉句s[i % len(熟悉句s)])
    return 特征码(桥)


def M7() -> dict:
    出 = []
    for hs in ("0", "1"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONIOENCODING="utf-8")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--指纹"],
                           capture_output=True, text=True, encoding="utf-8", env=env,
                           cwd=_HERE, timeout=1800)
        fp = None
        for ln in (r.stdout or "").splitlines():
            if ln.startswith("FP:"):
                fp = ln.split(":", 1)[1].strip()
        出.append(fp)
    同 = (出[0] is not None and 出[0] == 出[1])
    return dict(套="M7C13确定性", 指纹_HS0=(None if 出[0] is None else 出[0][:24]),
                指纹_HS1=(None if 出[1] is None else 出[1][:24]), _跨hashseed逐位同=bool(同))


# ────────────────────────── M8：C1 非 Transformer 审计 ──────────────────────────
禁式 = (r"import\s+transformers", r"from\s+transformers",
       r"\.softmax\s*\(", r"F\.softmax", r"nn\.MultiheadAttention",
       r"MultiheadAttention", r"scaled_dot_product_attention",
       r"\battention\s*\(", r"self_attn", r"q_proj", r"k_proj", r"v_proj",
       r"class\s+\w*Attention", r"nn\.Transformer", r"nn\.Linear")


def _剥注释字符串(源码: str) -> str:
    import io
    import tokenize
    out = []
    try:
        for tk in tokenize.generate_tokens(io.StringIO(源码).readline):
            out.append(" " if tk.type in (tokenize.COMMENT, tokenize.STRING) else tk.string)
    except Exception:
        return 源码
    return " ".join(out)


def _扫(路径s) -> list:
    import re
    命中 = []
    for p in sorted(路径s):
        with open(p, encoding="utf-8") as fh:
            码 = _剥注释字符串(fh.read())
        码 = re.sub(r"\s*\.\s*", ".", 码)          # ★点号归一：tokenize 会把 `nn.Linear` 拆成 `nn . Linear`
        for pat in 禁式:
            if re.search(pat, 码):
                命中.append((os.path.basename(p), pat))
    return 命中


def M8() -> dict:
    import tempfile
    扫s = [os.path.join(_HERE, f) for f in sorted(os.listdir(_HERE))
          if f.endswith(".py") and not f.startswith("_诊断")]
    命中 = _扫(扫s)
    # 对照臂：故意写禁词 ⇒ 必须抓到（用**能存活 tokenize 归一**的写法）
    目录 = tempfile.mkdtemp(prefix="meeting42_M8_")
    p = os.path.join(目录, "坏.py")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("import transformers\natt = MultiheadAttention\nx = nn.Linear(3, 4)\n")
    抓 = len(_扫([p])) > 0
    import shutil
    shutil.rmtree(目录, ignore_errors=True)
    return dict(套="M8C1审计", 扫描文件数=len(扫s), 命中=命中, _新件无禁词=bool(not 命中),
                _对照臂抓到=bool(抓))


# ────────────────────────── M9：学习≠记忆不回退 ──────────────────────────
def M9() -> dict:
    """对照臂：**同配置**下 元自我开/关 比既有记忆库条数——
      新层（注意图式/回读）只写自己的独立存储，**不得**往 codec._段 多写一条。
      （用 回读通道=False 隔离：无 ε_σ 反哺 ⇒ 决策逐位不受影响 ⇒ 计数差异只可能来自新层自身。）"""
    句s = 熟悉句s
    关 = 造桥_元自我(seed=900, 档案seed=310, 元自我开关=False)
    开 = 造桥_元自我(seed=900, 档案seed=310, 元自我开关=True,
                    注意开关=True, 回读通道=False)
    关.钉来源("自源")
    开.钉来源("自源")
    W0 = np.asarray(开.自我核.W_self, dtype=np.float64).copy() if 开.自我核 is not None else None
    for i in range(40):
        关.步(句s[i % len(句s)])
        开.步(句s[i % len(句s)])
    条关, 条开 = len(关.codec._段), len(开.codec._段)
    注意迹 = len(开.注意图式.迹) if getattr(开, "注意图式", None) is not None else 0
    ΔW = (float(np.max(np.abs(np.asarray(开.自我核.W_self, dtype=np.float64) - W0)))
          if W0 is not None else 0.0)
    return dict(套="M9学习≠记忆不回退", 记忆库条数_关=条关, 记忆库条数_开=条开,
                _记忆库0变=bool(条关 == 条开), 新层_Wself生长Δ=round(ΔW, 6),
                新层_注意迹=注意迹, _新层真的动了=bool(注意迹 > 0 and ΔW > 0.0))


# ────────────────────────── 主流程 ──────────────────────────
def _跑全部(快: bool = False) -> dict:
    果 = {}
    # M1/M2/M3 双集（判据集 + 留出集；--快 只跑判据集）
    for 集名, 种子s in 双集.items():
        if 快 and 集名 == "留出集":
            continue
        for s in 种子s:
            果["M1/%s/%d" % (集名, s)] = M1(s)
            果["M2/%s/%d" % (集名, s)] = M2(s)
            果["M3/%s/%d" % (集名, s)] = M3(s)
    果["M3对照臂"] = M3_对照臂(0)
    果["M4"] = M4()
    果["M5"] = M5()
    果["M6"] = M6()
    果["M7"] = M7()
    果["M8"] = M8()
    果["M9"] = M9()
    return 果


def _汇总(果: dict) -> dict:
    出 = dict(项={}, 全通过=True)
    # M1：三条件全过
    m1k = [k for k in 果 if k.startswith("M1/")]
    m1 = all(果[k]["_熟悉优于陌生"] and 果[k]["_熟悉优于拷贝"] and 果[k]["_sham判别消失"] for k in m1k)
    # M2：优于拷贝 ∧ 优于零W
    m2k = [k for k in 果 if k.startswith("M2/")]
    m2 = all(果[k]["_优于拷贝"] and 果[k]["_优于零W"] for k in m2k)
    # M3：主臂达门（全部种子）+ 对照臂逐位不变
    m3k = [k for k in 果 if k.startswith("M3/")]
    m3 = (all(果[k]["_达门"] for k in m3k)
          and 果["M3对照臂"]["_逐位不变"])
    m3主臂达门数 = sum(1 for k in m3k if 果[k]["_达门"])
    for k, v in 果.items():
        if k.startswith("M"):
            出["项"][k] = {kk: vv for kk, vv in v.items() if kk.startswith("_")}
    其它 = {"M4": 果["M4"]["_跨进程逐位同"] and 果["M4"]["_异档案不同"] and 果["M4"]["_篡改拒载"],
           "M5": (果["M5"]["_第一人称"] and 果["M5"]["_可溯源"] and 果["M5"]["_无元话语"]
                 and 果["M5"]["_无内态不追加"] and 果["M5"]["_元话语过滤"]),
           "M6": 果["M6"]["_逐位恒等"], "M7": 果["M7"]["_跨hashseed逐位同"],
           "M8": 果["M8"]["_新件无禁词"] and 果["M8"]["_对照臂抓到"],
           "M9": 果["M9"]["_记忆库0变"] and 果["M9"]["_新层真的动了"]}
    出["判"] = dict(M1=bool(m1), M2=bool(m2), M3=bool(m3), M3主臂达门数=m3主臂达门数,
                   M3主臂总数=len(m3k), M3对照臂逐位不变=bool(果["M3对照臂"]["_逐位不变"]),
                   M3_TVD均值=round(float(np.mean([果[k]["TVD"] for k in m3k])), 4), **其它)
    出["全通过"] = bool(m1 and m2 and m3 and all(其它.values()))
    return 出


def _主():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if "--载入指纹" in sys.argv:
        p = sys.argv[sys.argv.index("--载入指纹") + 1]
        状 = 持久化_整我.载整我(p)
        桥 = 造桥_元自我(seed=360, 档案seed=310)
        持久化_整我.装回(桥, 状)
        print("FINGERPRINT:" + 持久化_整我.整我指纹(桥))
        return 0
    if "--指纹" in sys.argv:
        print("FP:" + _跑指纹())
        return 0
    快 = "--快" in sys.argv
    果 = _跑全部(快=快)
    汇 = _汇总(果)
    # 落盘
    目录 = os.path.join(_HERE, "产物")
    os.makedirs(目录, exist_ok=True)
    with open(os.path.join(目录, "判据_元自我.json"), "w", encoding="utf-8") as f:
        json.dump(dict(明细=果, 汇总=汇), f, ensure_ascii=False, indent=1, default=str)
    print("=" * 86)
    print("meeting42 判据套件 · M1–M9（双集 seed0–5 / 6–11；主臂过 ∧ sham/对照不过 ⇒ PASS）")
    print("=" * 86)
    for k in sorted(果):
        v = 果[k]
        flags = {kk: vv for kk, vv in v.items() if kk.startswith("_")}
        print("[%s] %s" % (k, json.dumps(flags, ensure_ascii=False)))
    print("-" * 86)
    print("判目：%s" % json.dumps(汇["判"], ensure_ascii=False))
    print("-" * 86)
    if 汇["判"]["M3主臂达门数"] < 汇["判"]["M3主臂总数"]:
        print("★M3 身份反事实（行为级）：主臂达门 %d/%d，TVD均值 %.4f < τ %.2f ⇒ **FAIL（如实登记）**"
              % (汇["判"]["M3主臂达门数"], 汇["判"]["M3主臂总数"],
                 汇["判"]["M3_TVD均值"], τ_TVD))
        print("   根因：κ_priv=10 私偏+novelty 淹没身份反哺项；动作分布仅少数桶 ⇒ TVD 分辨率受限。")
    print("汇总：%s" % ("全 PASS" if 汇["全通过"] else "有 FAIL（见上）"))
    print("=" * 86)
    return 0 if 汇["全通过"] else 1


if __name__ == "__main__":
    raise SystemExit(_主())