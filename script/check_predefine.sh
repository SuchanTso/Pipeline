#!/bin/bash

# 记录总的退出状态码，默认为0（成功）
total_exit_code=0

# 定义一个函数来运行Python脚本并检查退出状态
run_python_script() {
    python3 "$@"
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "Error: Python script $1 exited with code $exit_code"
        total_exit_code=$exit_code
        return $exit_code
    fi
}

echo "Checking predefine values in the $1..."

# 运行多个Python脚本并检查它们的退出状态
run_python_script src/dataset/type_check.py -f "$1" -s "节点" -c '节点类型' -m "data/predefine/node_epa.json" || exit $total_exit_code
run_python_script src/dataset/type_check.py -f "$1" -s "节点" -c '节点类型.1' -m "data/predefine/node_type_industry.json" || exit $total_exit_code
run_python_script src/dataset/type_check.py -f "$1" -s "节点" -c '节点类型.2' -m "data/predefine/node_type.json" || exit $total_exit_code
run_python_script src/dataset/type_check.py -f "$1" -s "管道" -c '初始状态' -m "data/predefine/initial_state.json" || exit $total_exit_code
run_python_script src/dataset/type_check.py -f "$1" -s "管道" -c '材质' -m "data/predefine/material.json" || exit $total_exit_code

if [ $total_exit_code -eq 0 ]; then
    echo "=============================check predefine done============================="
else
    echo "=============================check predefine failed============================="
    exit $total_exit_code
fi