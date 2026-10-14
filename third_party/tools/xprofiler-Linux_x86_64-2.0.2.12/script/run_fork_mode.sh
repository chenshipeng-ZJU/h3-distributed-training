#!/bin/bash

# 清理旧的 socket 文件
rm -f "${PWD}/xprofiler.sock"

# 获取脚本所在目录
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

# 屏蔽 pybind_xprofiler 调用
export XPROFILER_PYBIND_DISABLE_DEVICES=0

# 启动 XProfiler
${DIR}/../bin/xprofiler --report -r500 --xputx --cpu --gzip --socket-path=/tmp/ python3 ${DIR}/example.py
