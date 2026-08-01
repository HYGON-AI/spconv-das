# Third-Party Notices

This file records the third-party projects directly referenced by this source
tree. Their original copyright and license notices remain in effect.

## spconv-rocm

- Project: spconv-rocm
- Repository: https://github.com/ZJLi2013/spconv_rocm.git
- Branch: `rocm`
- Baseline commit: `36172b6ac3fbed8320692ff6783cb2df4dc8e66d`
- License: Apache-2.0
- Local scope: repository-wide upstream baseline
- Modifications: Hygon adaptations for HCU/ROCm, including HIP kernels,
  packaging, native fallback paths, and validation

The unmodified upstream README at the fixed baseline is retained in
[README_ORIGIN.md](README_ORIGIN.md). The upstream license text is retained in
[LICENSE](LICENSE).

## PyTorch

- Project: PyTorch
- Repository: https://github.com/pytorch/pytorch
- Version requirement: runtime/build environment provides a compatible PyTorch
  release; this repository does not vendor PyTorch source
- License: BSD-3-Clause
- Local scope: Python and C++ extension dependency
- Modifications: none; used through public Python, ATen, and extension APIs

## NumPy

- Project: NumPy
- Repository: https://github.com/numpy/numpy
- Version requirement: supplied by the Python environment; this repository
  does not vendor NumPy source
- License: BSD-3-Clause
- Local scope: Python runtime dependency
- Modifications: none

## cumm-rocm (optional)

- Project: cumm-rocm
- Repository: https://github.com/ZJLi2013/cumm-rocm
- Version requirement: optional `implicit-gemm` extra; no fixed version is
  currently declared by `setup.py`
- License: Apache-2.0
- Local scope: optional external dependency; source is not vendored here
- Modifications: none in this repository

The current documented HCU fused/native routes do not require cumm-rocm. A
release that enables the optional extra must pin and record an approved exact
version or commit before distribution.

## FlyDSL

- Project: FlyDSL
- Repository: not vendored by this repository
- Version requirement: none in the current supported HCU runtime path
- License: Apache-2.0
- Local scope: historical/future optional integration references only
- Modifications: none in this repository

FlyDSL is not declared as an install requirement. If it is enabled in a future
release, record its authoritative repository and exact approved version or
commit here before distribution.
