#!/usr/bin/env bash
# run_2c
# P800 PCIe 16 卡机双 卡）创建脚本 ——
# 参考机房交付的 yibing01 脚本，仅改容器名与挂载路径，其余保持一致。
#   容器 A: h3_container_a  物理 XPU 0-7  (NUMA0, cpuset 0-63,128-191)
#   容器 B: h3_container_b  物理 XPU 8-15 → 容器内 0-7 (NUMA1, cpuset 64-127,192-255)
#
# 【背景 2026-09-29】机房确认：这台机的 16 卡互联已分区（两 8 卡域独立），
#   无法单（实测 错乱）。双 8 卡容器是当前机器的标准形态。
#   训练配置对应改 yaml: accelerate.num_processes: 8。
#
# 关键点:
#   - 非 --privileged（privileged 会无视 --device 限制，两个容器都看到 16 卡，无法按 8 卡分区）
#   - --device=/dev/infiniband: XCCL RDMA 需要 uverbs0-6 枚举 IB 设备（旧容器 privileged 自带；
#     非 privileged 必须显式挂，否则 "No IB dev found" / "not found in ib device list"）
#   - 数据面 NIC: XPU0-3→ens101np0(mlx5_1), XPU4-7→ens110np0(mlx5_2),
#                 XPU8-11→ens113np0(mlx5_5), XPU12-15→ens122np0(mlx5_6); 控制面 ens24np0(mlx5_0)
set -euo pipefail

IMG="xmlir_ubuntu_2004_x86_64:v0.42_0602_vision"  # 本地镜像; 远端镜像仓库 无 v0.42_0602 tag(404), 与 dev_container 等容器一致
NAME_A="h3_container_a"
NAME_B="h3_container_b"

common() {  # $1 = 容器名, 其余参数透传给 docker run
  local name="$1"; shift
  docker run -dti --name "$name" --net=host --security-opt=seccomp=unconfined \
    -e IMAGE_VERSION="$IMG" --cap-add=SYS_PTRACE --shm-size=512g --cap-add=SYS_ADMIN \
    --device /dev/fuse --device=/dev/infiniband --restart=always \
    --ulimit=memlock=-1 --ulimit=nofile=120000 --ulimit=stack=67108864 \
    -v /data:/workspace \
    "$@"
}

echo "[+] removing old containers (if any) ..."
docker rm -f "${NAME_A}" 2>/dev/null || true
docker rm -f "${NAME_B}" 2>/dev/null || true
# 顺带清理此前 FORM=16 试验建的同名容器（已确认该形态不可用）
docker rm -f h3_container 2>/dev/null || true

echo "[+] creating ${NAME_A} (physical XPU 0-7, NUMA0) ..."
common "${NAME_A}" --cpuset-cpus=0-63,128-191 \
  --device=/dev/xpu0:/dev/xpu0 --device=/dev/xpu1:/dev/xpu1 \
  --device=/dev/xpu2:/dev/xpu2 --device=/dev/xpu3:/dev/xpu3 \
  --device=/dev/xpu4:/dev/xpu4 --device=/dev/xpu5:/dev/xpu5 \
  --device=/dev/xpu6:/dev/xpu6 --device=/dev/xpu7:/dev/xpu7 \
  --device=/dev/xpuctrl "$IMG"

echo "[+] creating ${NAME_B} (physical XPU 8-15 → container 0-7, NUMA1) ..."
common "${NAME_B}" --cpuset-cpus=64-127,192-255 \
  --device=/dev/xpu8:/dev/xpu0 --device=/dev/xpu9:/dev/xpu1 \
  --device=/dev/xpu10:/dev/xpu2 --device=/dev/xpu11:/dev/xpu3 \
  --device=/dev/xpu12:/dev/xpu4 --device=/dev/xpu13:/dev/xpu5 \
  --device=/dev/xpu14:/dev/xpu6 --device=/dev/xpu15:/dev/xpu7 \
  --device=/dev/xpuctrl "$IMG"

echo "[+] verify:"
for C in "${NAME_A}" "${NAME_B}"; do
  echo "--- $C ---"
  docker exec "$C" sh -c 'echo "  xpu nodes: $(ls /dev/xpu[0-9]* 2>/dev/null | tr "\n" " ")"; echo "  xpuctrl:   $(ls /dev/xpuctrl 2>/dev/null)"; echo "  ib devs:   $(ls /dev/infiniband/ 2>/dev/null | tr "\n" " ")"'
done
echo "[+] DONE"
