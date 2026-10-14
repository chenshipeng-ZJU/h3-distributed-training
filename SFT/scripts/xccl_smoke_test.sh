#!/bin/bash
# =============================================================================
# XCCL 通信冒烟测试：不加载模型，最小用例验证多卡通信是否正常
#
# 用法：
#   bash scripts/xccl_smoke_test.sh              # 默认测 8 卡（当前训练配置）
#   bash scripts/xccl_smoke_test.sh 16           # 测 16 卡
#   bash scripts/xccl_smoke_test.sh 8 29601      # 指定端口（默认 29600）
#
# 测试项与顺序（FSDP 训练真实路径优先，all_reduce 原生测试殿后）：
#   1. all_gather      FSDP 前向/反向参数聚合（每步都用）
#   2. reduce_scatter  FSDP 梯度同步（每步都用）
#   3. barrier(patched) checkpoint 保存路径（barrier_fix 补丁后的行为）
#   4. bare barrier    原生裸 barrier（默认测；TEST_BARE=0 关闭）
#   5. all_reduce      原生 allreduce（默认测；TEST_ALLREDUCE=0 关闭；
#                      16 卡死锁/拆分是 XCCL 9d385bf 已知 bug，FSDP 训练不走此算子）
#
# 判定标准：
#   PASS = 前三项全部正常（FSDP 训练可放心开训）
#   第 4/5 项失败不代表训练不可用，仅作为通信层健康状况的独立信号
#
# 适用场景：机器重启后、XCCL/驱动升级后、改 BKCL 配置后、训练莫名卡死/挂掉后，
#           先跑本测试定位是不是通信层的问题
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NPROC="${1:-8}"
PORT="${2:-29600}"
TIMEOUT="${3:-120}"
# 诊断开关：测原生 barrier / allreduce（默认全开；TEST_BARE=0/TEST_ALLREDUCE=0 关闭）
TEST_BARE="${TEST_BARE:-1}"
TEST_ALLREDUCE="${TEST_ALLREDUCE:-1}"
# 兼容旧开关名 XCCL_TEST_ALLREDUCE=1（未设置时不报错——脚本开了 set -u）
if [[ "${XCCL_TEST_ALLREDUCE:-}" == "1" ]]; then
    TEST_ALLREDUCE=1
fi

TEST_PY="$(mktemp /tmp/xccl_smoke_XXXX.py)"
trap 'rm -f "${TEST_PY}"; pkill -9 -f "${TEST_PY}" 2>/dev/null || true' EXIT

cat > "${TEST_PY}" <<'PYEOF'
import os, torch
import torch.distributed as dist
rank = int(os.environ["RANK"])
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dist.init_process_group()
ws = dist.get_world_size()

# --- 1. all_gather：FSDP 前向/反向参数聚合路径（训练关键路径） ---
g = torch.full((1,), float(rank + 1), device="cuda")
out = [torch.zeros_like(g) for _ in range(ws)]
dist.all_gather(out, g)
expect = ws * (ws + 1) / 2
got = sum(x.item() for x in out)
assert abs(got - expect) < 1e-3, f"rank {rank}: all_gather 结果错误 {got} != {expect}"
print(f"[rank {rank}/{ws}] all_gather CORRECT", flush=True)

# --- 2. reduce_scatter：FSDP 梯度同步路径（训练关键路径） ---
# 输入 input[j] = (j+1)*(rank+1)；rank i 收到 Σ_r (i+1)*(r+1) = (i+1)*ws*(ws+1)/2
big = torch.arange(1, ws + 1, device="cuda", dtype=torch.float32) * (rank + 1)
rs = torch.zeros(1, device="cuda")
dist.reduce_scatter_tensor(rs, big)
expect_rs = (rank + 1) * ws * (ws + 1) / 2
assert abs(rs.item() - expect_rs) < 1e-3, f"rank {rank}: reduce_scatter 结果错误 {rs.item()} != {expect_rs}"
print(f"[rank {rank}/{ws}] reduce_scatter CORRECT", flush=True)

# --- 3. barrier：checkpoint 保存 / accelerate wait_for_everyone 场景使用 ---
# barrier_fix 补丁已注入（与训练一致），这里走的是 all_gather 替代实现；
# 后台进程会打印补丁标记 /tmp/barrier_fix_<rank>.ok 便于确认
dist.barrier()
print(f"[rank {rank}/{ws}] barrier ok (patched)", flush=True)

# --- 4. 原生裸 barrier（诊断项，TEST_BARE=1 开启）---
# 默认 PYTHONPATH 不注入 barrier_fix，直接调用就是原生 mock_barrier 行为
if os.environ.get("TEST_BARE") == "1":
    try:
        pg = dist._get_default_group()
        pg.barrier()  # ProcessGroup.barrier，绕开 dist.barrier 的函数级替换（若有）
        print(f"[rank {rank}/{ws}] bare barrier ok（原生）", flush=True)
    except Exception as e:
        print(f"[rank {rank}/{ws}] bare barrier EXCEPTION: {type(e).__name__}", flush=True)

# --- 5. 原生 all_reduce（诊断项，TEST_ALLREDUCE=1 开启）---
# XCCL 9d385bf 已知问题：16 卡死锁或静默拆分（返回 8.0 而非 16.0）。
# 这一项就是"报昆仑之前的老复现用例"，保留原样。
if os.environ.get("TEST_ALLREDUCE") == "1":
    t = torch.ones(1, device="cuda")
    dist.all_reduce(t)
    if t.item() == ws:
        print(f"[rank {rank}/{ws}] all_reduce CORRECT -> {t.item()}", flush=True)
    else:
        print(f"[rank {rank}/{ws}] all_reduce WRONG ({t.item()} != {ws}，通信域被拆分)", flush=True)
dist.destroy_process_group()
PYEOF

echo "========================================"
echo " XCCL 通信冒烟测试"
echo "========================================"
echo "卡数: ${NPROC}   端口: ${PORT}   超时: ${TIMEOUT}s"
echo "诊断项: bare_barrier=${TEST_BARE}  all_reduce=${TEST_ALLREDUCE}"
echo "========================================"

# 与训练保持同一环境（conda activate minimax_h3_csp 后运行）
# 注入 barrier_fix 补丁（与 stage2 训练的 PYTHONPATH 完全一致）——
# 第 3 项 dist.barrier() 被补丁替换为 all_gather 实现，测的是"训练真实行为"；
# 第 4 项绕过函数级替换直调 pg.barrier()，测的仍是原生行为（pybind 不可改）
# 默认全项测试（1-5 全跑）：TEST_BARE=1 TEST_ALLREDUCE=1
# 只测 FSDP 三项（快速判定训练可用性）: TEST_BARE=0 TEST_ALLREDUCE=0 bash xccl_smoke_test.sh 16
export TEST_BARE="${TEST_BARE}"
export TEST_ALLREDUCE="${TEST_ALLREDUCE}"
export PYTHONPATH="${SCRIPT_DIR}/barrier_fix:${PYTHONPATH:-}"

timeout "${TIMEOUT}" torchrun \
    --nproc_per_node="${NPROC}" \
    --master_port="${PORT}" \
    "${TEST_PY}" 2>&1 | grep -vE "XCCL /|SYMBOL_REWRITE|OMP_NUM_THREADS|^W[0-9]{4} |\*\*\*|Setting OMP"

RC=${PIPESTATUS[0]}
pkill -9 -f "${TEST_PY}" 2>/dev/null || true

echo "----------------------------------------"
if [[ ${RC} -eq 0 ]]; then
    echo "结果: PASS ✓（${NPROC} 卡 FSDP 关键路径 all_gather/reduce_scatter/barrier 正常）"
else
    echo "结果: FAIL ✗（exit=${RC}：124=超时死锁，1=结果错误或报错）"
    echo "提示: 看死锁发生在哪一步——"
    echo "      - all_gather / reduce_scatter 之前就无输出: 基础通信/初始化问题，FSDP 训练无法进行"
    echo "      - 打印到 all_gather/reduce_scatter OK 后超时: 卡在 barrier/all_reduce，"
    echo "        16 卡下属 XCCL 9d385bf 已知问题；FSDP 训练用补丁规避，不受影响"
    echo "      - 出现『WRONG (x != N)』: 通信域被静默拆分，比死锁更危险"
fi
echo "只测 FSDP 三项（快速判定训练可用性）: TEST_BARE=0 TEST_ALLREDUCE=0 bash $0 ${NPROC}"
exit ${RC}
