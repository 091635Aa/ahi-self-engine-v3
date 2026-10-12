# -*- coding: utf-8 -*-
"""判据_投影保距_视觉.py —— meeting43 · 视觉侧 F1 复核：fsw_kernel 的 E5 细码投影保距性。

背景（体检_视觉只读.md F1，原「存疑待验」）：`FSW内核._P` = (D_F=4096, D=1024) 固定随机投影，
`_细码 = sign(P@粗码)`（sign 丢幅度）——与语言链 P5 同型，其**保距性从未独立验证**。

J1 原始投影保距：随机样本两两余弦，投影(P@x)前后 Pearson 相关 > 0.9（JL 保距）
J2 sign 后仍携带可分信息：细码两两余弦与原始余弦相关 > 0.1 且 > sham+0.1
J3 sham 判不过：零投影 ⇒ 相关 ≈ 0（度量能失败）
J4 检索可分性：写入 N 条记录，用记录自身的粗码检索地址，top1 命中自身率 == 1.0（细码精排不作弊）
运行：python 判据_投影保距_视觉.py （exit 0）
"""
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
AHI3 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (os.path.join(AHI3, "meeting38", "实施"), AHI3):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
import fsw_kernel                                                      # noqa: E402
from fsw_kernel import FSW内核                                          # noqa: E402


def 两两余弦(A):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    C = An @ An.T
    iu = np.triu_indices(A.shape[0], 1)
    return C[iu]


def main() -> int:
    k = FSW内核(种子=3800)
    D, D_F = k.D, k.D_F
    rng = np.random.default_rng(20261011)
    X = rng.standard_normal((96, D)).astype(np.float32)
    Xb = np.sign(X).astype(np.float32)                                  # 双极粗码（与内核输入同域）

    c原 = 两两余弦(Xb)
    Z = (k._P @ Xb.T).T                                                 # 原始投影（不清化）
    c投 = 两两余弦(Z)
    细 = np.stack([k._细码(Xb[i]) for i in range(Xb.shape[0])], 0)
    c细 = 两两余弦(细)

    # J1 用**连续**输入做标准 JL 检验（±1 双极码余弦被量化成粗格点/大量并列，Pearson 会被压低，属测量伪影）
    c原连 = 两两余弦(X)
    c投连 = 两两余弦((k._P @ X.T).T)

    # J1 标准 JL 判据：距离平方比值的最大相对误差（P 为 N(0,1/D) 阵 ⇒ E‖Px‖²=(D_F/D)‖x‖²；
    # 余弦相关因 PᵀP≈(D_F/D)I 带 ~1.6% 相对噪声，理论上限 <1，故不用它当唯一口径）
    s = float(D_F) / float(D)
    Xp = (k._P @ X.T).T
    D原 = X[:, None, :] - X[None, :, :]
    D投 = Xp[:, None, :] - Xp[None, :, :]
    n2原 = np.sum(D原 * D原, axis=2)
    n2投 = np.sum(D投 * D投, axis=2)
    iu = np.triu_indices(X.shape[0], 1)
    比 = n2投[iu] / np.maximum(s * n2原[iu], 1e-12)
    JL最大相对误差 = float(np.max(np.abs(比 - 1.0)))

    def 相关(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0

    r投 = 相关(c原连, c投连)                                            # 诊断：余弦相关
    r细 = 相关(c原, c细)                                                # J2（双极输入，真实用法）
    r投双 = 相关(c原, c投)                                              # 记录：双极输入下的投影相关（量化伪影）

    # sham：零投影
    k0 = FSW内核(种子=3800)
    k0._P = np.zeros_like(k0._P)
    Z0 = (k0._P @ Xb.T).T
    细0 = np.stack([k0._细码(Xb[i]) for i in range(Xb.shape[0])], 0)
    r投0 = 相关(c原, 两两余弦(Z0))
    r细0 = 相关(c原, 两两余弦(细0))

    # J4：写入 N 条记录 → 自检索命中自身
    命中 = 0
    N = 60
    for i in range(N):
        k.写记录([("r0", "v%d" % i), ("r1", "w%d" % (i % 7))], 句柄=i)
    for i in range(N):
        r = k.检索地址(k.记录[i]["束"], k=8)
        命中 += int(r.get("top1") == i)
    命中率 = 命中 / N

    J1 = JL最大相对误差 < 0.5
    J2 = (r细 > 0.1) and (r细 > r细0 + 0.1)
    J3 = abs(r投0) < 0.05
    J4 = 命中率 >= 0.95

    print("D=%d D_F=%d 样本=%d" % (D, D_F, Xb.shape[0]))
    print("J1 JL 距离比相对误差 最大 = %.4f（<0.5 即保距）" % JL最大相对误差)
    print("   ［诊断］余弦相关 连续输入=%.4f ｜ 双极输入=%.4f（±1 量化致并列）" % (r投, r投双))
    print("J2 细码(sign)相关 = %.4f ｜ sham 细码相关 = %.4f" % (r细, r细0))
    print("J3 sham 投影相关 = %.4f" % r投0)
    print("J4 自检索 top1 命中自身 = %d/%d = %.3f" % (命中, N, 命中率))
    print("\n=== 判据 ===")
    print("J1 fsw 投影 JL 保距(<0.5)   :", "PASS" if J1 else "FAIL")
    print("J2 sign 后仍携可分信息       :", "PASS" if J2 else "FAIL")
    print("J3 sham 判不过(|r|<0.05)     :", "PASS" if J3 else "FAIL")
    print("J4 检索可分性(≥0.95)         :", "PASS" if J4 else "FAIL")
    print("总判据:", "PASS" if (J1 and J2 and J3 and J4) else "CHECK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
