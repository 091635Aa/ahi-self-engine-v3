# -*- coding: utf-8 -*-
"""运行_第七轮.py —— AHI V3 第七轮根启动器（薄壳）。

真正的总装在 meeting41/实施/主程序_第七轮.py；本文件把实施目录加进 sys.path 后按其 __main__ 运行。

用法（在 AHI3 目录下）：
    python 运行_第七轮.py               # 一条龙：能力核 → 训练 → 边想边做边说 → 判据
    python 运行_第七轮.py --能力         # 只看 L1 能力核（线性时间/长程依赖/任意训练）
    python 运行_第七轮.py --训练 6       # 只看 L2 训练 6 轮（15→10→5→3 秒台账）
    python 运行_第七轮.py --同时 80      # 只看 L3 跑 80 拍（思考⇄手⇄正文同拍）
    python 运行_第七轮.py --判据         # 只跑 L4 总判据（双集 + 反例臂）
    python 运行_第七轮.py --效率         # 只跑 L4b 效率判据（E1/E2/E3）
    python 运行_第七轮.py --真机演练     # 只跑 L5 真机探头演练档（不动鼠标键盘）
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.join(HERE, "meeting41", "实施")
for p in (IMPL, os.path.join(HERE, "meeting40", "实施"), os.path.join(HERE, "meeting38", "实施")):
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    runpy.run_path(os.path.join(IMPL, "主程序_第七轮.py"), run_name="__main__")
