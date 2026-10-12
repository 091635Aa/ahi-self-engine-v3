# -*- coding: utf-8 -*-
"""持久化_整我.py —— meeting42 件 D：把「整个我」跨重启存/载（**复用** `快照.py`，不改任何既有件）。

设计出处：`meeting42/会议_五专家/03_同一性与持久化专家.md`（A/B/C 三类清单）。
口径（理论③）：**同一性锚在"档案"，不锚"动力学"** —— 存的是**可复现的身份与学到的东西**，
不是某一拍的瞬时几何。

三类（专家03 判定）：
  · **A 类＝必须进档**（不可由 seed 重算、且构成"谁是它/它学会了什么"）：
      `自我核.a_seal`（身份锚）、`自我核.W_self`（自指联想）、`自我核.慢`（自归一基线）、
      `自模型.W / .上键 / .上态`（预测自己的那张表）、`注意图式.W / .上键`（预测自己注意的表）、
      `元自我核.ε_σ史`、`驱动_σ.ε_σ_slow / .ema_max`、`顿悟核.ε史/站史/命史/事件拍s`、
      `思核.迹`（独立思考轨迹，存在才存）。
  · **B 类＝由 `档案seed` 确定性重算**（`角色s / 码s / 方向表 / 位族`）⇒ **不进档**（进了是冗余）。
  · **C 类＝审计流不进档**（`自我核.迹 / 顿悟账本 / 各 迹` 审计副本、计数器）⇒ **不进档**。

复用的是 `meeting40/实施/快照.py`（npz + manifest + **规范载荷 sha256** + `temp+os.replace` 原子替换 +
`fail-closed` 载入）。**本文件不 import 任何器官**；只碰文件与桥状态。
本文件加一层 `档案schema版本`（分层校验：本件版本 + 快照件版本）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AHI3 = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_AHI3, "meeting40", "实施"),
           os.path.join(_AHI3, "meeting39", "实施"),
           os.path.join(_AHI3, "meeting38", "实施"), _AHI3):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import 快照                                                        # noqa: E402

档案schema版本 = 1


def _键(名, 子) -> str:
    return "%s/%s" % (名, 子)


def 打包(桥) -> dict:
    """把「整个我」的 A 类状态打包成**扁平** dict（numpy 数组 + JSON 标量）——确定性、可 sha256。"""
    出: dict = dict(schema版本=int(档案schema版本))
    出["档案seed"] = int(getattr(桥, "_档案seed", getattr(桥, "档案seed", -1)))
    # ★C 类：拍号 / 各计数器**不进档**（专家03 §1.2 C 类；进了会与"从零起的新桥"指纹不同）
    # 身份锚 + 自指联想 + 慢基线
    zk = getattr(桥, "自我核", None)
    if zk is not None:
        出[_键("自我核", "a_seal")] = np.asarray(zk.a_seal, dtype=np.float64)
        出[_键("自我核", "W_self")] = np.asarray(zk.W_self, dtype=np.float64)
        出[_键("自我核", "慢")] = {k: (None if v is None else float(v))
                                  for k, v in dict(getattr(zk, "慢", {})).items()}
    # 自模型 ĝ
    zm = getattr(桥, "自模型", None)
    if zm is not None and getattr(zm, "W", None) is not None:
        出[_键("自模型", "W")] = np.asarray(zm.W, dtype=np.float64)
        出[_键("自模型", "上键")] = np.asarray(zm._上键, dtype=np.float64)
        出[_键("自模型", "上态")] = np.asarray(zm._上态, dtype=np.float64)
    # 注意图式 ĝ_att（第七层）
    att = getattr(桥, "注意图式", None)
    if att is not None and getattr(att, "W", None) is not None:
        出[_键("注意图式", "W")] = np.asarray(att.W, dtype=np.float64)
        出[_键("注意图式", "上键")] = np.asarray(att._上键, dtype=np.float64)
    # 元自我核（回读史） + 驱动_σ（反哺基线）
    yz = getattr(桥, "元自我核", None)
    if yz is not None:
        出[_键("元自我核", "ε_σ史")] = np.asarray([d["ε_σ"] for d in yz.ε_σ史], dtype=np.float64)
        出[_键("元自我核", "ε_σ_slow")] = (None if yz.ε_σ_slow is None else float(yz.ε_σ_slow))
    drv = getattr(桥, "drv", None)
    if drv is not None and hasattr(drv, "ε_σ_slow"):
        出[_键("驱动_σ", "ε_σ_slow")] = (None if drv.ε_σ_slow is None else float(drv.ε_σ_slow))
        出[_键("驱动_σ", "ema_max")] = (None if drv.ema_max is None else float(drv.ema_max))
        出[_键("驱动_σ", "ε_σ_自身")] = (None if drv.ε_σ_自身 is None else float(drv.ε_σ_自身))
    # 顿悟核（独立学习存储）
    dw = getattr(桥, "顿悟核", None)
    if dw is not None:
        出[_键("顿悟核", "ε史")] = np.asarray(list(dw.ε史), dtype=np.float64)
        出[_键("顿悟核", "站史")] = np.asarray(list(dw.站史), dtype=np.int64)
        出[_键("顿悟核", "命史")] = np.asarray(list(dw.命史), dtype=np.float64)
        出[_键("顿悟核", "事件拍s")] = np.asarray(list(dw.事件拍s), dtype=np.int64)
    # 思核（独立思考轨迹；存在才存）
    sx = getattr(桥, "思核", None)
    if sx is not None and getattr(sx, "迹", None):
        try:
            出[_键("思核", "迹")] = np.stack([np.asarray(d["z"], dtype=np.float32) for d in sx.迹], 0)
        except Exception:
            pass
    return 出


def 装回(桥, 状态: dict) -> None:
    """把打包状态装回桥（同 schema 才装；缺哪件跳哪件——只读容错）。"""
    if int(状态.get("schema版本", -1)) != int(档案schema版本):
        raise 整我损坏("schema 版本不符：%r ≠ %d" % (状态.get("schema版本"), 档案schema版本))
    zk = getattr(桥, "自我核", None)
    if zk is not None and _键("自我核", "a_seal") in 状态:
        zk.a_seal = np.asarray(状态[_键("自我核", "a_seal")], dtype=np.float64)
        zk.W_self = np.asarray(状态[_键("自我核", "W_self")], dtype=np.float64)
        慢 = 状态.get(_键("自我核", "慢")) or {}
        for k, v in 慢.items():
            zk.慢[k] = (None if v is None else float(v))
    zm = getattr(桥, "自模型", None)
    if zm is not None and _键("自模型", "W") in 状态:
        zm.W = np.asarray(状态[_键("自模型", "W")], dtype=np.float64)
        zm._上键 = np.asarray(状态[_键("自模型", "上键")], dtype=np.float64)
        zm._上态 = np.asarray(状态[_键("自模型", "上态")], dtype=np.float64)
    att = getattr(桥, "注意图式", None)
    if att is not None and _键("注意图式", "W") in 状态:
        att.W = np.asarray(状态[_键("注意图式", "W")], dtype=np.float64)
        att._上键 = np.asarray(状态[_键("注意图式", "上键")], dtype=np.float64)
    yz = getattr(桥, "元自我核", None)
    if yz is not None and _键("元自我核", "ε_σ史") in 状态:
        arr = np.asarray(状态[_键("元自我核", "ε_σ史")], dtype=np.float64)
        yz.ε_σ史 = [dict(拍=-1, ε_σ=float(x)) for x in arr]
        yz.ε_σ_slow = 状态.get(_键("元自我核", "ε_σ_slow"))
    drv = getattr(桥, "drv", None)
    if drv is not None and hasattr(drv, "ε_σ_slow") and _键("驱动_σ", "ε_σ_slow") in 状态:
        drv.ε_σ_slow = 状态.get(_键("驱动_σ", "ε_σ_slow"))
        drv.ema_max = 状态.get(_键("驱动_σ", "ema_max"))
        drv.ε_σ_自身 = 状态.get(_键("驱动_σ", "ε_σ_自身"))
    dw = getattr(桥, "顿悟核", None)
    if dw is not None and _键("顿悟核", "ε史") in 状态:
        dw.ε史 = [float(x) for x in np.asarray(状态[_键("顿悟核", "ε史")]).ravel()]
        dw.站史 = [int(x) for x in np.asarray(状态[_键("顿悟核", "站史")]).ravel()]
        dw.命史 = [float(x) for x in np.asarray(状态[_键("顿悟核", "命史")]).ravel()]
        dw.事件拍s = [int(x) for x in np.asarray(状态[_键("顿悟核", "事件拍s")]).ravel()]
    sx = getattr(桥, "思核", None)
    if sx is not None and _键("思核", "迹") in 状态:
        arr = np.asarray(状态[_键("思核", "迹")])
        sx.迹 = [dict(拍=-1, z=arr[i].astype(np.float32)) for i in range(arr.shape[0])]


def 存整我(路径: str, 桥) -> dict:
    """把「整个我」存到 `路径`（npz+manifest，原子替换）。返回 manifest（含 `payload_sha256`=整我指纹）。"""
    return 快照.存(str(路径), 打包(桥), [])


def 载整我(路径: str, 校验: bool = True) -> dict:
    """载入「整个我」的**扁平状态**（不含事件）。**校验不过 ⇒ 抛 整我损坏（fail-closed）**。"""
    try:
        状态, _事 = 快照.载(str(路径), 校验=校验)
    except 快照.快照损坏 as ex:
        raise 整我损坏(str(ex))
    if int(状态.get("schema版本", -1)) != int(档案schema版本):
        raise 整我损坏("schema 版本不符：%r ≠ %d" % (状态.get("schema版本"), 档案schema版本))
    return 状态


def 整我指纹(桥) -> str:
    """当前「整个我」的规范载荷指纹（同 `快照` 口径，可直接跨进程比对逐位同）。"""
    状态 = 打包(桥)
    数组s = {k: np.asarray(v) for k, v in 状态.items() if hasattr(v, "shape") or hasattr(v, "dtype")}
    标量s = {k: v for k, v in 状态.items() if k not in 数组s}
    return 快照._载荷指纹(数组s, 标量s, [])                          # noqa: SLF001（复用既有口径）


class 整我损坏(Exception):
    """「整个我」不可用（缺件/版本不符/校验失败）。**fail-closed**。"""


# ─────────────────────────── 自测 ───────────────────────────
def _自测(临时目录: str = None) -> int:
    import shutil
    import tempfile
    from 内核_元自我 import 造桥_元自我
    目录 = 临时目录 or tempfile.mkdtemp(prefix="meeting42_整我_")
    路径 = os.path.join(目录, "我.npz")
    句s = ["今天的风很轻。", "我把桌子擦干净了。", "窗外有一棵树。", "水杯放在手边。"]
    try:
        桥 = 造桥_元自我(seed=360, 档案seed=310)
        桥.钉来源("自源")
        for i in range(40):
            桥.步(句s[i % len(句s)])
        f1 = 整我指纹(桥)
        清 = 存整我(路径, 桥)
        # 取到**新桥**（同配置，未跑）→ 装回 → 指纹须逐位同
        桥2 = 造桥_元自我(seed=360, 档案seed=310)
        状 = 载整我(路径)
        装回(桥2, 状)
        f2 = 整我指纹(桥2)
        同 = (f1 == f2 == str(清.get("payload_sha256")))
        # 篡改 ⇒ 拒载
        坏 = dict(状)
        k0 = next(k for k, v in 坏.items() if hasattr(v, "shape") and k.endswith("a_seal"))
        with open(路径, "wb") as fh:
            np.savez_compressed(fh, **{k: (np.asarray(v) + 1.0 if k == k0 else np.asarray(v))
                                       for k, v in 坏.items() if hasattr(v, "shape")})
        拒 = False
        try:
            载整我(路径)
        except 整我损坏:
            拒 = True
        # 异档案种子的指纹必须不同
        桥3 = 造桥_元自我(seed=360, 档案seed=3719)
        异 = (整我指纹(桥3) != f1)
        print("=" * 78)
        print("持久化_整我 · 自测（整个我：存/载/跨桥逐位同）")
        print("=" * 78)
        print("指纹 甲=%s" % f1[:24])
        print("指纹 装回后=%s（同=%s）" % (f2[:24], 同))
        print("篡改拒载=%s ｜ 异档案种子指纹不同=%s" % (拒, 异))
        好 = bool(同 and 拒 and 异)
        print("-" * 78)
        print("自测：%s" % ("PASS" if 好 else "FAIL"))
        print("=" * 78)
        return 0 if 好 else 1
    finally:
        if 临时目录 is None:
            shutil.rmtree(目录, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(_自测())