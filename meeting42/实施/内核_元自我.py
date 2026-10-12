# -*- coding: utf-8 -*-
"""内核_元自我.py —— meeting42 第七层：把「我」从**留痕**做成**闭环主体**（默认关·纯新增）。

设计出处：`meeting42/07_终稿_工程自我意识优化.md`（Lead 综合裁决）+ 五专家方案
（`meeting42/会议_五专家/01..05`）。**不改任何既有 .py**（零 C14 风险）。

三件事（全部纯 HDC · 零梯度 · 确定性 C13）：
  A. **注意图式 ĝ_att（AST）** —— 预测「我下一步注意什么」，`ε_att = 1 − cos(Â_{t+1}, A_{t+1})`。
     注意空间 = 引擎既有 `A_t = sign(Φ_act @ a星_t)`（1024 维母语丰富态，**非 5 维摘要**）；
     键 `k_t = sign(S_t) ⊛ sign(A_t)`（自态 ⊛ 当拍注意；★必须含自态——否则退化成拷贝，
     先例 `ahi_theory/brain.py:475-483` 的拷贝基线 0.976<0.992）。纯 HDC Hebbian/NLMS 记忆。
     只读旁挂：**不改 rec、不改状态、不改决策**。
  B. **`W_self` 回读 + 自指反哺** —— `Ŝ_t = sign(W_self @ Ä_t)`，`Ä_t = a_seal ⊛ À_t`
     （★**读取端绑定身份锚**：若 key=À_t 则 a_seal 被约掉、「只换身份锚」必 FAIL，见专家02 §2.3）；
     残差 `ε_σ = 1 − cos(À_t, Ŝ_t)` 经**既有内生偏置 b[1]/b[3]** 反哺（同 `内核_真学习.驱动_ε` 范式；
     **禁直写** a_prev/i_prev/prev_c/消化债/饱足）。**自归一**（`κ_σ = κ_priv = 10` 结构性等价，
     抗私偏淹没；meeting22 先例）。严格 **t−1 因果**。
  C. **内省外显（只读）** —— `元自我外显(桥)` 给一句可溯源、无元话语的第一人称半句；中性**不追加**。
     （**不**接线到 `输出层.py`；接线属既有件改动，留所有者点头。）

不破既有（C14/D1）：`元自我开关=False`（默认）⇒ `_引擎步` 走 super() ⇒ **逐位恒等**；
旁挂失败静默，绝不打断既有路径。本文件**不 import 任何器官**（在测试/主程序里显式加载）。

诚实边界（不宣称超出本步）：① 本引擎的「注意」是**选择型**（9 个动作候选中 argmin 选中），
不是脑的神经增益；② 只主张**功能性评估意识 / 可测的反事实依赖**，**不主张**因果咬合、
自由意志、主观体验或人类级意识；③ `ε_att`/`ε_σ` 的判别力**以实测数字为准**，未测前不作"已成立"宣称。

纪律：C1 禁 Transformer｜C9 禁 OCR｜C10 禁外挂模型当器官｜C13 确定性可复现｜C14 不破既有。
用法（测试）：见 `判据_元自我.py`。
"""
from __future__ import annotations

import os
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

from 内核_真学习 import 驱动_ε, ALPHA_SLOW, KAPPA                          # noqa: E402
from 内核_自模型 import ETA_R                                            # noqa: E402
from 内核_画面 import 顶覆盖_综合体H                                      # noqa: E402

开关默认 = False            # ★保护开关默认关（C14/D1：默认关 ⇒ 既有输出逐位恒等）
κ_σ = 10.0                 # 自指反哺的**归一目标**（= synthesis 私偏默认 κ_priv=10；**结构性等价**，
                          #   与私偏同刻度竞争 ⇒ 抗淹没；**不是**放大常数——meeting22 已否决"直接量纲放大"）


def _单位(x) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _双极(rng: np.random.Generator, n: int) -> np.ndarray:
    出 = np.where(rng.random(int(n)) < 0.5, -1.0, 1.0).astype(np.float64)
    return 出


def _符号(x) -> np.ndarray:
    v = np.sign(np.asarray(x, dtype=np.float64))
    v[v == 0.0] = 1.0
    return v


def _余弦(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ══════════════════════ A. 注意图式 ĝ_att（AST） ══════════════════════
class 注意图式ĝ_att:
    """预测「我下一步注意什么」的 HDC 线性联想记忆（与 `内核_自模型.自模型ĝ` 同构）。

    · 键  k_t = sign(S_t) ⊛ A_t（自态 ⊛ 当拍注意；缺自态 ⇒ 退化成拷贝，先例 brain.py:475-483）。
    · 值  A_{t+1} = 单位(Φ_act @ a星_{t+1})（下一拍**实际选中**的注意方向；自监督真值）。
    · 预测 Â = W @ k（**连续读出**），`ε_att = 1 − cos(Â, A)`；更新 NLMS（η 复用既有 ETA_R）。
    · ★**经实测的偏离（诚实登记）**：本引擎的注意码若是 `sign(Φ_act@a星)`（9 个离散码），
      实测 ε_att ≈ 0.85~0.92、相邻码 cos≈0（近乎正交）⇒ 判别力差（`_诊断_注意目标对比.py`：离散末窗 0.8477）。
      改**连续**注意方向 `单位(Φ_act@a星)` ⇒ 末窗 **0.8008**（优于拷贝基线 0.9620）。
      故本实现取连续目标（**偏离**专家01 §2.2 的离散码建议，以实测裁决）。
    · 存储 `迹` = **独立学习存储**（与记忆库 codec._段 物理分离；记忆≠学习，所有者硬命题）。
    """

    def __init__(self, 更新: bool = True, η0: float = ETA_R):
        self.W: np.ndarray | None = None
        self.更新 = bool(更新)
        self.η0 = float(η0)
        self._上键 = None
        self.迹: list = []
        self.更新次数 = 0
        self.预测次数 = 0

    def 学一拍(self, 键k, 注意A, 拍: int = -1):
        """键k = k_t；注意A = A_t（本拍注意方向）。返回本拍 ε_att（首拍无预测 ⇒ None）。"""
        A = np.asarray(注意A, dtype=np.float64)
        k = np.asarray(键k, dtype=np.float64)
        if self.W is None:
            self.W = np.zeros((int(A.shape[0]), int(A.shape[0])), dtype=np.float64)
        ε = None
        if self._上键 is not None:
            Â = self.W @ self._上键                            # 连续读出
            self.预测次数 += 1
            ε = 1.0 - _余弦(Â, A)
            if self.更新:
                n2 = float(self._上键 @ self._上键) + 1e-6
                self.W += (self.η0 * ε) * np.outer(A - Â, self._上键) / n2
                self.更新次数 += 1
        self._上键 = k
        self.迹.append(dict(拍=int(拍), ε_att=(None if ε is None else float(ε))))
        return ε

    def ε序列(self) -> list:
        return [d["ε_att"] for d in self.迹 if d["ε_att"] is not None]

    def 摘要(self) -> dict:
        ε = self.ε序列()
        return dict(迹条数=len(self.迹), 更新次数=self.更新次数, 预测次数=self.预测次数,
                    D=(int(self.W.shape[0]) if self.W is not None else 0),
                    W范数=(round(float(np.linalg.norm(self.W)), 4) if self.W is not None else None),
                    ε均值=(round(float(np.mean(ε)), 4) if ε else None))


# ══════════════════════ B. W_self 回读（自指意外 ε_σ） ══════════════════════
class 元自我核:
    """从自指联想记忆 `W_self` **读回**「我」（`内核_自我回路` 目前只写不读 ⇒ 死存储）。

    · `Ä_t = a_seal ⊛ À_t`（读取端绑定身份锚，±1 逐元素；写入端不动）。
    · `Ŝ_t = sign(W_self @ Ä_t)`；残差 `ε_σ = 1 − cos(À_t, Ŝ_t)`（「从记忆里回想的我」vs「此刻的我」）。
    · 独立存储 `ε_σ史` + 自身慢基线 `ε_σ_slow`（自归一，零人挑阈值）。
    """

    def __init__(self, D: int = 1024, α慢: float = None):
        self.D = int(D)
        self.α慢 = float(ALPHA_SLOW if α慢 is None else α慢)
        self.ε_σ史: list = []
        self.ε_σ_slow = None
        self.回读次数 = 0
        self.空读次数 = 0

    def 一拍(self, 自我核, 拍: int = -1):
        """读回一拍；返回 (ε_σ, Ŝ)。`自我核` 为 None / 无迹 / W_self 未长 ⇒ 返回 (None, None)。"""
        if 自我核 is None or not getattr(自我核, "迹", None):
            return None, None
        À = _符号(自我核.迹[-1]["À"])
        a_seal = np.asarray(自我核.a_seal, dtype=np.float64)
        W = np.asarray(自我核.W_self, dtype=np.float64)
        Ä = a_seal * À                                     # ★读取端绑定身份锚（±1）
        Ŝ = _符号(W @ Ä)
        ε_σ = 1.0 - _余弦(À, Ŝ)
        self.ε_σ史.append(dict(拍=int(拍), ε_σ=float(ε_σ)))
        self.回读次数 += 1
        if not np.any(W):
            self.空读次数 += 1
        if self.ε_σ_slow is None:
            self.ε_σ_slow = float(ε_σ)
        else:
            self.ε_σ_slow = float(self.ε_σ_slow) + self.α慢 * (float(ε_σ) - float(self.ε_σ_slow))
        return float(ε_σ), Ŝ

    def ε序列(self) -> list:
        return [d["ε_σ"] for d in self.ε_σ史]

    def 摘要(self) -> dict:
        ε = self.ε序列()
        return dict(读回次数=self.回读次数, 空读次数=self.空读次数,
                    ε_σ最近=(round(ε[-1], 4) if ε else None),
                    ε_σ慢基线=(None if self.ε_σ_slow is None else round(float(self.ε_σ_slow), 4)),
                    ε_σ均值=(round(float(np.mean(ε)), 4) if ε else None))


# ══════════════════════ 反哺：驱动_σ（唯一合法口 = 既有 b[]） ══════════════════════
class 驱动_σ(驱动_ε):
    """在 `驱动_ε` 之上**再叠一项** 自指意外 ε_σ 的偏置（与 `驱动_ε` 叠 内生驱动器 同构）。

    · 门限 = ε_σ **自身的慢 EMA**（`raw = max(0, ε_σ − ε_σ_slow)`，自归一；供**外显**用）；
    · 强度 = **保水平** `项 = κ_σ · ε_σ / 2`（ε_σ 定义域 [0,2] ⇒ `项 ∈ [0, κ_σ]`，`κ_σ=κ_priv=10`
      与私偏同刻度）。★**经实测的偏离（诚实登记）**：最初按"自归一 ÷ max(EMA_max|raw|,|raw|)"
      实现，实测**换 a_seal 时行为几乎不变**（`_诊断_回读.py`：TVD 仅 0.030）——原因正是
      **自归一把两臂 ε_σ 的"水平差"约掉了**（各臂跟着**自己**的基线走）⇒ 身份被看不见
      （meeting22「相对量会把持久偏置抵消」的同一种病）。改**保水平**（除以定义域上界 2，
      非人挑阈值）后身份可承重（实测见 `证据_元自我.md`）。
    · **只叠加到 b[1]/b[3]**（提问位 / 转移话题位），**禁直写**状态量；严格 **t−1 因果**。
    """

    def __init__(self, n候选: int = 9, α_slow: float = ALPHA_SLOW, κ: float = KAPPA,
                 初值: float = 1.0, α_meta: float = 0.02, γ_m: float = 0.5):
        super().__init__(n候选=n候选, α_slow=α_slow, κ=κ, 初值=初值, α_meta=α_meta, γ_m=γ_m)
        self.ε_σ_自身 = None      # 最近一次吸收的 ε_σ（t−1 因果承载）
        self.ε_σ_slow = None      # ε_σ 的慢 EMA 基线
        self.ema_max = None       # max|raw| 的慢 EMA（自归一分母）
        self._raw = 0.0
        self.ε_σ史: list = []      # ★独立存储（与记忆库物理分离）
        self.项_σ史: list = []

    @classmethod
    def 提升(cls, 旧) -> "驱动_σ":
        """把既有驱动器实例**原状态搬入**（不新增语义常数，只加挂）。兼容 驱动_ε / 内生驱动器。"""
        新 = cls(n候选=int(getattr(旧, "n", 9)),
                 α_slow=float(getattr(旧, "α", ALPHA_SLOW)),
                 κ=float(getattr(旧, "κ", KAPPA)))
        新.slow = list(getattr(旧, "slow", []))
        新.last_ê = list(getattr(旧, "last_ê", []))
        新.meta_err = list(getattr(旧, "meta_err", []))
        新.α_m = float(getattr(旧, "α_m", 0.02))
        新.γ_m = float(getattr(旧, "γ_m", 0.5))
        新.拍数 = int(getattr(旧, "拍数", 0))
        if isinstance(旧, 驱动_ε):
            新.ε_自身 = 旧.ε_自身
            新.ε_slow = 旧.ε_slow
            新.ε史 = list(旧.ε史)
            新.ε位数 = 旧.ε位数
        return 新

    # ── 本拍动作选择**完成后**调用：ε_σ 进独立存储 + 更新自归一基数（下一拍因果原料） ──
    def 吸收σ(self, ε_σ) -> None:
        if ε_σ is None:
            return
        e = float(ε_σ)
        if self.ε_σ_slow is None:
            self.ε_σ_slow = e
        else:
            self.ε_σ_slow = float(self.ε_σ_slow) + self.α * (e - float(self.ε_σ_slow))
        raw = max(0.0, e - float(self.ε_σ_slow))
        m = abs(raw)
        self.ema_max = m if self.ema_max is None else (self.α * m + (1.0 - self.α) * float(self.ema_max))
        self.ε_σ_自身 = e
        self._raw = raw
        self.ε_σ史.append(e)

    # ── 本拍动作选择**前**调用：在 super().偏置() 之上叠 ε_σ 项 ──
    def 偏置(self) -> list:
        b = list(super().偏置())
        if self.ε_σ_自身 is not None and len(b) > 3:
            项 = κ_σ * (float(self.ε_σ_自身) / 2.0)                 # ε_σ∈[0,2] ⇒ 项∈[0,κ_σ]（保水平）
            if 项 > 0.0:
                b[1] += 项                                          # 提问位→停下想/反刍
                b[3] += 项                                          # 转移话题位→联想检索
            self.项_σ史.append(float(项))
        return b

    def 摘要(self) -> dict:
        return dict(ε_σ条数=len(self.ε_σ史), ε_σ慢基线=(None if self.ε_σ_slow is None
                                                      else round(float(self.ε_σ_slow), 4)),
                    最近项_σ=(round(self.项_σ史[-1], 4) if self.项_σ史 else None))


# ══════════════════════ 第七层宿主（默认关 · 逐位恒等） ══════════════════════
class 顶覆盖_综合体M(顶覆盖_综合体H):
    """第七层：在六层链之顶**再**加挂元自我（注意图式 + W_self 回读反哺 + 内省外显）。

    · 元自我开关=False（默认）：`_引擎步` 走 super()，`drv` 不提升 ⇒ **逐位恒等**基链（C14/D1）。
    · 元自我开关=True：每拍只读旁挂——① 喂注意图式（独立迹）② 读回 W_self 得 ε_σ → 反哺 b[]。
      **引擎状态推进行一行不改**。
    """

    def __init__(self, seed: int = 310, 私偏κ: float = 10.0, 元自我开关=None,
                 注意开关=None, 回读通道=None, 档案seed=None, 自我开关=None,
                 学习开关=None, 思核开关=None, 自模型开关=None, 顿悟开关=None,
                 画面开关=None, 学习: bool = True, 状态投影=None, 思核参: dict = None,
                 画面种子=None, 自我提问开关=None):
        self.元自我开关 = bool(开关默认 if 元自我开关 is None else 元自我开关)
        # 注意图式只读（安全）；回读通道 = 反哺（会改行为，J1 的 sham 臂会单独冻结它）
        self.注意开关 = (bool(self.元自我开关) if 注意开关 is None else bool(注意开关)) if self.元自我开关 else False
        self.回读通道 = (bool(self.元自我开关) if 回读通道 is None else bool(回读通道)) if self.元自我开关 else False
        self.注意图式 = None
        self.元自我核 = None
        super().__init__(seed=seed, 私偏κ=私偏κ, 自我开关=自我开关, 档案seed=档案seed,
                         学习开关=学习开关, 思核开关=思核开关, 自模型开关=自模型开关,
                         顿悟开关=顿悟开关, 画面开关=画面开关, 学习=学习,
                         状态投影=状态投影, 思核参=思核参, 画面种子=画面种子,
                         自我提问开关=自我提问开关)
        if self.元自我开关:
            D = int(np.asarray(self.eng.hdc.Φ_act).shape[0])
            if self.注意开关:
                self.注意图式 = 注意图式ĝ_att()
            self.元自我核 = 元自我核(D=D)
            drv = getattr(self, "drv", None)
            if drv is not None and not isinstance(drv, 驱动_σ):
                self.drv = 驱动_σ.提升(drv)                      # 只换偏置源，状态原样搬入

    # ── 备选身份锚（判据台"只换 a_seal"用；不改内核语义） ──
    def 设身份锚(self, 向量) -> None:
        if getattr(self, "自我核", None) is not None:
            self.自我核.a_seal = np.asarray(向量, dtype=np.float64)

    # ── 每拍只读旁挂 ──
    def _元自我旁挂(self, rec) -> None:
        # ① 注意图式（只读预测，不进决策）
        if self.注意图式 is not None:
            h = (rec.get("模块") or {}).get("hdc") or {}
            S = h.get("H双极")
            a星 = rec.get("a星")
            if S is not None and a星 is not None:
                Φ = np.asarray(self.eng.hdc.Φ_act, dtype=np.float64)
                A = _单位(Φ @ np.asarray(a星, dtype=np.float64))
                if float(np.linalg.norm(A)) > 1e-9:                 # 空动作（零候选）⇒ 本拍不学
                    self.注意图式.学一拍(_符号(S) * A, A, 拍=int(self.拍))
        # ② W_self 回读 → 反哺
        if self.回读通道 and self.元自我核 is not None:
            ε_σ, _Ŝ = self.元自我核.一拍(getattr(self, "自我核", None), 拍=int(self.拍))
            drv = getattr(self, "drv", None)
            if ε_σ is not None and isinstance(drv, 驱动_σ):
                drv.吸收σ(ε_σ)

    def _引擎步(self, u):
        rec = super()._引擎步(u)                                  # 基链状态推进 + 步2~6/画面旁挂，逐行不变
        if self.元自我开关:
            try:
                self._元自我旁挂(rec)
            except Exception:
                pass                                              # 旁挂失败绝不打断既有路径
        return rec

    def 元自我态(self) -> dict:
        if not self.元自我开关:
            return dict(元自我=False)
        出 = dict(元自我=True, 注意开关=self.注意开关, 回读通道=self.回读通道)
        出["注意图式"] = (self.注意图式.摘要() if self.注意图式 is not None else None)
        出["回读"] = (self.元自我核.摘要() if self.元自我核 is not None else None)
        drv = getattr(self, "drv", None)
        出["反哺"] = (drv.摘要() if isinstance(drv, 驱动_σ) else None)
        return 出


# ══════════════════════ 零改既有：自拼七层链 ══════════════════════
def 造桥_元自我(seed: int = 310, 私偏κ: float = 10.0, 档案seed=None, 开关_=None,
                元自我开关: bool = True, 注意开关=None, 回读通道=None, **kw):
    """造一个**挂了七层**的综合体（元自我层在六层链之顶）。**不编辑 `内核宿主.py`**。

    默认沿用 `内核宿主.开关`（六开关全开）+ 元自我开关；`造桥()` 原样不动 ⇒ C14 零风险。
    """
    from 内核宿主 import 开关 as _V3开关
    开 = dict(_V3开关)
    if 开关_:
        开.update(开关_)
    桥 = 顶覆盖_综合体M(seed=seed, 私偏κ=私偏κ, 档案seed=档案seed,
                        元自我开关=元自我开关, 注意开关=注意开关, 回读通道=回读通道,
                        **开, **kw)
    try:
        if getattr(桥, "思核开关", False):
            桥._起思核()                                          # 与 内核宿主.造桥 同律
    except Exception:
        pass
    return 桥


# ══════════════════════ C. 内省外显（只读；中性不追加） ══════════════════════
_元话语词 = ("向量", "激活", "内部状态是", "我的内部状态", "hdc", "HDC", "嵌入", "张量", "维度是")


def 净化(文本: str) -> str:
    """元话语过滤器（**只读**用，镜像 `输出层.净化` 的口径）：出现元话语 ⇒ 返回空串。"""
    t = str(文本 or "")
    for w in _元话语词:
        if w in t:
            return ""
    return t.strip()


def 元自我外显(桥, 中性: bool = True) -> dict:
    """给一句**可溯源、无元话语**的第一人称半句；中性（无异常内态）⇒ `半句=None`（**不追加**）。

    返回 dict(半句, 触发, 拍, 溯源, 有内态)。**只读**；不改任何状态、不入输出流。
    """
    if 桥 is None or not getattr(桥, "元自我开关", False):
        return dict(半句=None, 触发=None, 拍=None, 溯源=None, 有内态=False)
    拍 = int(getattr(桥, "拍", 0))
    半句, 触发, 溯源 = None, None, None
    核 = getattr(桥, "元自我核", None)
    drv = getattr(桥, "drv", None)
    # 自指意外（自归一：最近 ε_σ 高于其慢基线）
    if 核 is not None and 核.ε_σ史 and 核.ε_σ_slow is not None and isinstance(drv, 驱动_σ):
        ε = float(核.ε_σ史[-1]["ε_σ"])
        项 = float(getattr(drv, "_raw", 0.0))                     # 自归一门（高于自身基线才说）
        if 项 > 0.0:
            半句 = "我觉着这会儿的我，跟记忆里对不齐了一点（自指意外↑）。"
            触发 = "ε_σ"
            溯源 = ("回读", 拍, round(ε, 4), round(float(项), 4))
    # 注意图式意外
    if 半句 is None and 中性 is False:
        进 = getattr(桥, "注意图式", None)
        if 进 is not None and 进.ε序列():
            半句 = "我有点押不准自己接下来会盯住哪儿。"
            触发 = "ε_att"
            溯源 = ("注意图式", 拍, round(float(进.ε序列()[-1]), 4))
    净 = 净化(半句) if 半句 else ""
    return dict(半句=(净 or None), 触发=(触发 if 净 else None),
                拍=(拍 if 净 else None), 溯源=(溯源 if 净 else None), 有内态=bool(净))


# ══════════════════════ 自测（可直接 `python 内核_元自我.py`） ══════════════════════
def _自测() -> int:
    句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。"]
    关 = 顶覆盖_综合体M(seed=331, 元自我开关=False, 自我开关=True, 学习开关=True,
                        自模型开关=True, 思核开关=True, 顿悟开关=True)
    开 = 顶覆盖_综合体M(seed=331, 元自我开关=True, 自我开关=True, 学习开关=True,
                        自模型开关=True, 思核开关=True, 顿悟开关=True)
    开.钉来源("自源")                                          # 自源 ⇒ W_self 才生长（J5）
    for i in range(40):
        关.步(句s[i % len(句s)])
        开.步(句s[i % len(句s)])
    态 = 开.元自我态()
    外显 = 元自我外显(开)
    print("=" * 78)
    print("内核_元自我 · 自测（第七层：注意图式 + W_self 回读反哺 + 内省外显）")
    print("=" * 78)
    print("【关】元自我开关=False ⇒ 元自我态=%s（应 元自我=False）" % 关.元自我态())
    print("【开】%s" % 态)
    print("【外显】%s" % 外显)
    好 = (关.元自我态().get("元自我") is False
          and 态.get("元自我") is True
          and 态.get("注意图式", {}).get("预测次数", 0) > 0
          and 态.get("回读", {}).get("读回次数", 0) > 0)
    print("-" * 78)
    print("自测：%s（注意图式/回读都真的动了）" % ("PASS" if 好 else "FAIL"))
    print("=" * 78)
    return 0 if 好 else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(_自测())