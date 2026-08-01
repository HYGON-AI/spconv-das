# Spconv for Hygon

## Open Source Attribution

This repository is based on [spconv-rocm](https://github.com/ZJLi2013/spconv_rocm)
and contains Hygon-maintained changes for HCU/ROCm environments. `spconv-rocm`
is derived from the upstream [spconv](https://github.com/traveller59/spconv)
project.

- Direct upstream: [spconv-rocm](https://github.com/ZJLi2013/spconv_rocm)
- Upstream branch: `rocm`
- Upstream baseline commit: `36172b6ac3fbed8320692ff6783cb2df4dc8e66d`
- Upstream lineage: [spconv](https://github.com/traveller59/spconv)
- Upstream documentation: [spconv documentation](https://spconv.pytorch.org/)
- Upstream-derived historical reference: [README_ORIGIN.md](README_ORIGIN.md)
- Third-party inventory: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- License: Apache-2.0

`README_ORIGIN.md` is retained as an upstream-derived historical reference. It
contains historical cumm/FlyDSL descriptions and does not define the supported
routes of this HCU build. The upstream [LICENSE](LICENSE) and existing source
file copyright and attribution notices are retained.

## Modification Scope

Hygon changes adapt sparse convolution for HCU/ROCm. The implementation includes
HIP indice-pair generation, HIP fused convolution kernels, a native C++ fallback,
extension packaging, and functional, numerical, and performance validation.
Unless explicitly stated otherwise, upstream-originated files remain governed by
their inherited Apache-2.0 license terms.

Modified by Hygon Information Technology Co., Ltd.

## Build and Installation

### Requirements

- Python >= 3.8
- PyTorch built for the target HCU environment

cumm and FlyDSL are not build or runtime dependencies of the current HCU path.

### Build from Source

```bash
# Build the packaged HIP extension in place.
CXX=hipcc python3 setup.py build_ext --inplace
pip install -e .
```

`setup.py` uses `ROCM_PATH` when set and otherwise defaults to `/opt/dtk`.
The extension is packaged as `spconv/_C_hip*.so`; runtime does not require
first-use JIT compilation.

Build a distributable wheel with:

```bash
CXX=hipcc python3 setup.py bdist_wheel
```

## Public Interface Support

The Python interface remains compatible with the upstream `spconv.pytorch`
surface used by common sparse-convolution models.

| Area          | Current support                                                              |
| ------------- | ---------------------------------------------------------------------------- |
| Sparse tensor | `SparseConvTensor`, indice-key reuse, dense conversion                     |
| Convolution   | `SubMConv`, `SparseConv`, `SparseConvTranspose`, `SparseInverseConv` |
| Dimensions    | 1D, 2D, 3D, and 4D public module interfaces                                  |
| Modules       | `SparseSequential`, normalization/activation modules, tables, pooling      |
| Training      | Forward, dinput, and dfilter for FP32, FP16, and BF16                        |
| AMP           | FP16 and BF16 autocast paths                                                 |

The HIP fused convolution path currently targets eligible non-1x1 sparse
convolutions. 1x1 convolution bypasses indice kernels and uses the GEMM path.
Unsupported cases use the native C++ gather -> GEMM -> scatter route.

Set `SPCONV_FORCE_NATIVE=1` before importing `spconv.pytorch` to force the
non-1x1 native route. This disables fused forward, fused dinput, fused dfilter,
and the FP16 workspace dinput path. It is intended for debugging, numerical
comparison, and native-route performance baselines.

## Current and Future GEMM Routes

| Route              | Current HCU status | Usage                                                   |
| ------------------ | ------------------ | ------------------------------------------------------- |
| HIP fused          | Supported          | Default for eligible FP32, FP16, and BF16 non-1x1 cases |
| Native C++ loop    | Supported          | Fallback and`SPCONV_FORCE_NATIVE=1` baseline          |
| cumm implicit GEMM | Not supported      | Future optional integration only                        |
| FlyDSL             | Not supported      | Future optional integration only                        |

Future cumm/FlyDSL work must remain optional, preserve the fused/native routes
as fallbacks, and pass independent HCU correctness and performance validation
before it can be enabled by default.

## Testing and Accuracy

Run the public sparse-convolution test suite on an HCU device:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python3 -m pytest test/test_sparse_conv.py -v
```

Fused accuracy tests exercise public `SubMConv3d` and `SparseConv3d` interfaces
against a dense `torch.nn.functional.conv3d` reference. They cover forward,
dinput, and dfilter, including regular-sparsity, high-sparsity, and unique-index
dispatch cases.

| Data type | Acceptance criterion                                                              |
| --------- | --------------------------------------------------------------------------------- |
| FP32      | `torch.allclose(..., atol=1e-4, rtol=1e-4)` for forward, dinput, and dfilter    |
| FP16/BF16 | MaskImplicitGemm-style L2 check:`norm(diff) < 10 * max(1, max(Cin, Cout) / 16)` |

Low-precision tests use an FP32 dense reference because the final FP16/BF16
store has dtype quantization. The L2 criterion evaluates aggregate numerical
error while retaining coverage of each forward and backward output.

## Performance

For eligible non-1x1 shapes, HIP fused is the default route. It combines sparse
gather, matrix multiplication, and accumulation in HIP kernels, avoiding the
intermediate work of the native C++ loop. Native remains available for unsupported
shapes and as a controlled numerical baseline.

Do not use the cumm implicit-GEMM measurements in `README_ORIGIN.md` as current
HCU performance results. Performance depends on active-point count, sparsity,
kernel volume, channels, dtype, and the target HCU. Measure on the deployment
device with GPU events, at least 10 warmup iterations, and at least 50 timed
iterations; report forward and forward-plus-backward separately.

## References

- [Official spconv repository](https://github.com/traveller59/spconv)
- [spconv-rocm direct upstream](https://github.com/ZJLi2013/spconv_rocm)
- [spconv documentation](https://spconv.pytorch.org/)
- [Upstream-derived historical reference](README_ORIGIN.md)

## License

Apache 2.0.

Additional modifications copyright:
Copyright 2026 Hygon Information Technology Co., Ltd.
