# SpConv-ROCm: Sparse Convolution for AMD GPUs

Sparse convolution library for AMD ROCm GPUs, forked from [spconv](https://github.com/traveller59/spconv).
Replaces 33k lines of pccm/cumm C++ codegen with ~1.5k lines of Python + self-contained HIP kernels.

## Architecture

```
Original:  Python → pccm (codegen) → C++ → nvcc → GPU     (152 files, 33k lines)
ROCm:      Python → HIP kernel + cumm-rocm implicit GEMM → GPU    (~1.5k lines)
```

| Component | Original | ROCm |
|-----------|----------|------|
| GEMM (inference) | pccm C++ codegen (~20k lines) | cumm-rocm implicit GEMM (fused gather+GEMM+scatter) |
| GEMM (training) | pccm C++ codegen | C++ `at::mm` loop (native path) |
| indice pairs | CUDA hash table (pccm/tensorview) | Self-contained HIP kernel (Murmur3 + open-addressing) |
| indice_conv loop | C++ for-loop (pccm) | C++ for-loop (JIT compiled, ATen API) |
| Python binding | pccm → nvcc → .so | `torch.utils.cpp_extension.load()` JIT |

### GEMM Dispatch

`indice_conv` automatically selects the fastest path:

```
Inference (requires_grad=False):
  1. cumm-rocm implicit GEMM  (fused gather+GEMM+scatter, 1 kernel launch)
  2. C++ at::mm loop           (fallback if cumm-rocm unavailable)

Training (requires_grad=True):
  1. C++ at::mm loop           (autograd-compatible)
```

cumm-rocm implicit GEMM uses MFMA cross-KV fusion for small-medium channels
(2.5-3.7x faster) and falls through to native for large channels. See
[cumm-rocm README](https://github.com/ZJLi2013/cumm-rocm) for kernel details.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (ROCm build)
- [cumm-rocm](https://github.com/ZJLi2013/cumm-rocm) (implicit GEMM + FlyDSL GEMM; optional but recommended)
- ROCm 6.x+ with hipCUB

## Install

```bash
# Install cumm-rocm (recommended, enables implicit GEMM)
git clone https://github.com/ZJLi2013/cumm-rocm.git
cd cumm-rocm && git checkout rocm && pip install -e .

# Install spconv-rocm
git clone https://github.com/ZJLi2013/spconv_rocm.git
cd spconv_rocm && git checkout rocm && pip install -e .
```

HIP kernels are JIT-compiled on first use (~10s). Without cumm-rocm, spconv
falls back to `torch.mm` for all GEMM operations.

## Usage

```python
import spconv.pytorch as spconv
import torch

# Same API as original spconv
conv = spconv.SubMConv3d(16, 32, 3).cuda()
# SparseConv3d, SparseConvTranspose3d, SparseSequential, etc. all work
```

## Features

- [x] `SubMConv3d`, `SparseConv3d`, `SparseConvTranspose3d`
- [x] `SparseSequential`, `ToDense`
- [x] `SparseMaxPool3d`, `SparseAvgPool3d`
- [x] Forward + backward (training supported)
- [x] fp32, fp16, bf16, AMP
- [x] 1D, 2D, 3D sparse convolution
- [x] indice_key reuse, dilation, bias
- [x] Encoder-Decoder (UNet) architecture
- [x] Implicit GEMM (fused gather+GEMM+scatter via cumm-rocm)
- [x] 27 tests passing on MI308X (gfx942)

## Performance (MI308X gfx942)

### Single layer — implicit GEMM vs native (inference)

| Config | Native (ms) | Implicit GEMM (ms) | Speedup |
|--------|------------|-------------------|---------|
| SubM 3x3x3, N=5k, C=32 | 1.85 | **0.82** | **2.26x** |
| SubM 3x3x3, N=20k, C=32→64 | 1.89 | **1.02** | **1.85x** |
| SubM 3x3x3, N=50k, C=16→32 | 1.93 | **0.95** | **2.03x** |
| Conv 3x3x3 s=2, N=20k, C=64→128 | 2.30 | 2.44 | 0.94x |

### 4-layer backbone (typical 3D detection)

```
SubM(16→32) → Stride2(32→64) → SubM(64→64) → Stride2(64→128)
B=2, N=20k/batch, spatial=[200,200,10]
```

| Path | Forward |
|------|---------|
| Native (at::mm loop) | 10.84ms |
| **Implicit GEMM** | **8.87ms (1.22x)** |

### indice pairs: HIP kernel vs alternatives

| Implementation | SubM N=50k, 3x3x3 |
|---------------|-------------------|
| GPU PyTorch native (sort+searchsorted) | 15.2ms |
| **GPU HIP kernel (current)** | **0.6ms** |
| Original CUDA (reference) | ~2ms |

## Code Structure

```
spconv/
  pytorch/
    ops.py           — GEMM dispatch (implicit GEMM / at::mm), indice_conv, pooling
    conv.py          — SparseConv3d, SubMConv3d modules
    core.py          — SparseConvTensor
    modules.py       — SparseSequential, SparseBatchNorm, SparseReLU
    pool.py          — SparseMaxPool, SparseAvgPool, SparseGlobalPool
    identity.py      — Identity module
    tables.py        — AddTable, ConcatTable, JoinTable
    constants.py     — ALL_WEIGHT_IS_KRSC, etc.
  csrc_hip/
    hash_table.h           — GPU hash table (Murmur3 + open-addressing)
    indice_pairs_kernel.hip — SubM/Regular conv indice pairs HIP kernels
    indice_pairs_api.cpp    — PyTorch C++ extension binding + indice_conv C++ loop
  core.py            — ConvAlgo enum
  constants.py       — Package-level constants
  tools.py           — CUDAKernelTimer
  utils/             — Voxelization stubs (Point2Voxel API compatibility)
test/
  test_sparse_conv.py    — 27 correctness + numerical validation tests
  bench_e2e.py           — End-to-end benchmark (native path)
  bench_implicit_gemm.py — A/B benchmark: implicit GEMM vs native
  bench_indice_pairs.py  — indice pairs kernel benchmark
  check_dispatch.py      — Verify implicit GEMM dispatch is active
```

## Testing & Benchmarking

```bash
# Correctness tests
python -m pytest test/test_sparse_conv.py -v

# End-to-end benchmark
python test/bench_e2e.py

# Implicit GEMM A/B comparison
PYTHONPATH=/path/to/cumm-rocm python test/bench_implicit_gemm.py
```

## Known Limitations

- Implicit GEMM is inference-only (`requires_grad=False`); training uses native `at::mm` path.
- Large channels (C_in×C_out ≥ ~8K): implicit GEMM has ~6% overhead vs native due to
  crossk preprocessing cost; per-KV implicit GEMM is planned in cumm-rocm.
- FlyDSL `hgemm_splitk` has a BLOCK_K alignment requirement that may reject certain
  1×1 conv shapes in fp16 (`test_conv1x1_fp16` known issue).

## License

Apache 2.0
