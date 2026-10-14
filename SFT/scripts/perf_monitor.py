#!/usr/bin/env python3
"""
压测采样 → TensorBoard（与训练进程写同一个 event 目录，实时曲线可见）。

采样每张 XPU 的显存/利用率/功耗：
  - perf/mem_avg_MiB、perf/mem_max_MiB、perf/util_avg_pct、perf/power_avg_W：全卡聚合
  - perf/card_NN/mem_MiB、perf/card_NN/util_pct、perf/card_NN/power_W：每卡独立曲线
    （16 卡训练时在 TensorBoard 左侧按 card_00 ~ card_15 分组，可单独勾选对比）
x 轴为脚本启动后的秒数（与 loss 曲线的 step 轴相互独立，各自成图）。

用法: python perf_monitor.py <tensorboard_log_dir> [采样间隔秒，默认30]
"""

import re
import subprocess
import sys
import time


def sample_xpu():
    """返回 (mems_MiB, utils_pct, powers_W) 列表，失败返回空。"""
    try:
        out = subprocess.run(
            ["xpu-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except Exception:
        return [], [], []
    # xpu-smi 每个 XPU 两行，第二行格式形如:
    # | N/A 39C N/A 80W / 400W | 44930MiB / 98304MiB | 0% Default |
    pat = re.compile(r"(\d+)W / \d+W\s*\|\s*(\d+)MiB / \d+MiB\s*\|\s*(\d+)%")
    mems, utils, powers = [], [], []
    for m in pat.finditer(out):
        powers.append(int(m.group(1)))
        mems.append(int(m.group(2)))
        utils.append(int(m.group(3)))
    return mems, utils, powers


def main():
    tb_dir = sys.argv[1]
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=tb_dir)
    t0 = time.time()
    print(f"[perf_monitor] TensorBoard 压测曲线写入 {tb_dir}（每 {interval:.0f}s 采样）", flush=True)
    while True:
        mems, utils, powers = sample_xpu()
        if mems:
            el = int(time.time() - t0)
            # 全卡聚合
            writer.add_scalar("perf/mem_avg_MiB", sum(mems) / len(mems), el)
            writer.add_scalar("perf/mem_max_MiB", max(mems), el)
            writer.add_scalar("perf/util_avg_pct", sum(utils) / len(utils), el)
            writer.add_scalar("perf/power_avg_W", sum(powers) / len(powers), el)
            # 每卡独立曲线（卡不均衡/掉队 rank 一眼可见）
            for i, (mem, util, power) in enumerate(zip(mems, utils, powers)):
                writer.add_scalar(f"perf/card_{i:02d}/mem_MiB", mem, el)
                writer.add_scalar(f"perf/card_{i:02d}/util_pct", util, el)
                writer.add_scalar(f"perf/card_{i:02d}/power_W", power, el)
            writer.flush()
        time.sleep(interval)


if __name__ == "__main__":
    main()
