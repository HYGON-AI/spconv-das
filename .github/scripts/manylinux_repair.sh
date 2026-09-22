#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

# 对构建产物执行 auditwheel repair, 生成 manylinux wheel (仅 release 流程使用)
# 用法: manylinux_repair.sh [输入目录=dist] [输出目录=dist-manylinux]

# DTK 运行环境 (HIP/ROCm 库路径); env.sh 引用了未初始化的变量,
# 必须在 set -u 之前加载, 否则报 unbound variable
if [ -f /opt/dtk/env.sh ]; then
  # shellcheck disable=SC1091
  source /opt/dtk/env.sh
fi

set -euo pipefail

IN_DIR="${1:-dist}"
OUT_DIR="${2:-dist-manylinux}"

pip install --no-cache-dir auditwheel patchelf

# 依赖库搜索路径, 供 auditwheel 解析 spconv/_C_hip*.so 的依赖。
# wheel 内只有这一个扩展, 不自带任何 .so 库目录, 故无需定位已安装包
TORCH_PATH=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
echo "TORCH_PATH=${TORCH_PATH}"
export LD_LIBRARY_PATH="${TORCH_PATH}/lib:${LD_LIBRARY_PATH:-}"
# auditwheel 用 --ldpaths 的调用路径只认这个专用变量, 一并导出
export AUDITWHEEL_LD_LIBRARY_PATH="${TORCH_PATH}/lib:${AUDITWHEEL_LD_LIBRARY_PATH:-}"

# glibc 版本 -> manylinux 平台标签, 如 manylinux_2_17_x86_64
GLIBC_VER=$(ldd --version | awk '{print $NF}' | head -n1 | tr -d '\n' | tr '.' '_')
PLAT="manylinux_${GLIBC_VER}_x86_64"

# HIP/ROCm 运行时由 DCU 驱动提供, torch 库由用户环境的 torch 提供, 均不打进 wheel
excludes=(
  libgalaxyhip.so.5
  libamdhip64.so.5
  libamdhip64.so.6
  libhsa-runtime64.so.1
  librocm_smi64.so.2
  libhipblas.so.0
  libhipblas.so.2
  libhipblaslt.so.0
  libhipsparse.so.0
  libhipsparse.so.1
  librocblas.so.0
  librocblas.so.4
  librocsparse.so.0
  librocsparse.so.1
  libc10.so
  libc10_hip.so
  libtorch.so
  libtorch_cpu.so
  libtorch_hip.so
  libtorch_python.so
  libtorch_global_deps.so
  libomp.so
)

exclude_args=()
for lib in "${excludes[@]}"; do
  exclude_args+=(--exclude "${lib}")
done

mkdir -p "${OUT_DIR}"
auditwheel repair --plat "${PLAT}" --strip \
  "${exclude_args[@]}" \
  -w "${OUT_DIR}" "${IN_DIR}"/*.whl
