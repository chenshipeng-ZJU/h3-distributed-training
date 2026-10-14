#!/bin/bash

# 清理旧的 socket 文件
rm -f "${PWD}/xprofiler.sock"

# 获取脚本所在目录
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

SOCKET_PATH="/tmp/"

# 屏蔽 pybind_xprofiler 调用
export XPROFILER_PYBIND_DISABLE_DEVICES=0

# 启动 XProfiler (使用 -t10 参数，10秒后自动退出)
${DIR}/../bin/xprofiler --report -r500 --xputx --cpu -t10 --gzip --socket-path=${SOCKET_PATH} &> xprofiler.log &
xprofiler_pid=$!

# 等待 XProfiler 启动
echo "等待 XProfiler 启动..."
max_wait=30
wait_interval=1
time_waited=0

while [ $time_waited -lt $max_wait ]; do
    if [ -S "${SOCKET_PATH}/xprofiler.sock" ]; then
        echo "XProfiler 已就绪 (等待了 ${time_waited} 秒)"
        break
    fi

    # 检查进程是否还在运行
    if ! ps -p $xprofiler_pid > /dev/null 2>&1; then
        echo "错误: XProfiler 进程异常退出"
        # 输出日志文件内容以便调试
        if [ -f "xprofiler.log" ]; then
            echo "XProfiler 日志内容："
            cat xprofiler.log
        fi
        exit 1
    fi

    echo "等待中... (${time_waited}/${max_wait} 秒)"
    sleep $wait_interval
    time_waited=$((time_waited + wait_interval))
done

# 检查是否超时
if [ ! -S "${SOCKET_PATH}/xprofiler.sock" ]; then
    echo "错误: XProfiler 启动超时，未在 ${max_wait} 秒内创建 xprofiler.sock 文件"
    # 输出日志文件内容以便调试
    if [ -f "xprofiler.log" ]; then
        echo "XProfiler 日志内容："
        cat xprofiler.log
    fi
    exit 1
fi

# 设置环境变量
export XPU_ENABLE_PROFILER_TRACING=1
export XPU_TRACING_OUTPUT_NAME=${SOCKET_PATH}/xprofiler.sock
export NVTX_INJECTION64_PATH=${DIR}/../so/libxpuToolsExt.so
export XPUTX_RUN_MODE=client
export XPUTX_LISTEN_ADDR=unix:${SOCKET_PATH}/xputx.sock

# 运行 Python 脚本
echo "开始运行 Python 脚本..."
python3 ${DIR}/example.py

# 等待 XProfiler 进程结束（因为使用了 -t10 参数，会自动退出）
echo "等待 XProfiler 完成..."
wait $xprofiler_pid
xprofiler_exit_code=$?

if [ $xprofiler_exit_code -eq 0 ]; then
    echo "XProfiler 已正常完成"
else
    echo "警告: XProfiler 退出码为 $xprofiler_exit_code"
    # 输出日志文件内容以便调试
    if [ -f "xprofiler.log" ]; then
        echo "XProfiler 日志内容："
        tail -20 xprofiler.log
    fi
fi