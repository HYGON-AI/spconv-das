# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

# spconv-rocm: Sparse convolution ops for ROCm
# - indice pairs: HIP kernel (primary) / PyTorch native ops (fallback)
# - GEMM: FlyDSL for fp16/bf16, torch.mm for fp32

import os
import importlib
import warnings
import torch
import numpy as np
from typing import List, Optional, Tuple
from torch.autograd import Function
from spconv.core import ConvAlgo
from spconv.constants import ALL_WEIGHT_IS_KRSC, AllocKeys

# HIP indice pairs kernel (prebuilt into the wheel / install tree)
_HIP_MODULE = None
_HIP_LOAD_ATTEMPTED = False

def _get_hip_module():
    """Lazy-load the prebuilt HIP indice pairs extension."""
    global _HIP_MODULE, _HIP_LOAD_ATTEMPTED
    if _HIP_LOAD_ATTEMPTED:
        return _HIP_MODULE
    _HIP_LOAD_ATTEMPTED = True
    try:
        _HIP_MODULE = importlib.import_module("spconv._C_hip")
    except Exception as e:
        warnings.warn(
            "Failed to load packaged HIP indice pairs extension, "
            f"falling back to PyTorch native ops: {e}"
        )
        _HIP_MODULE = None
    return _HIP_MODULE

# FlyDSL GEMM backend (via cumm-rocm)
_FLYDSL_TUNER = None

def _get_flydsl_tuner():
    """Lazy-init FlyDSLGemmTuner (only when FlyDSL is available)."""
    global _FLYDSL_TUNER
    if _FLYDSL_TUNER is None:
        try:
            from cumm.gemm_tuner import FlyDSLGemmTuner
            _FLYDSL_TUNER = FlyDSLGemmTuner()
        except ImportError:
            _FLYDSL_TUNER = False  # FlyDSL not available, fallback to torch.mm
    return _FLYDSL_TUNER


def _gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM dispatch: FlyDSL for fp16/bf16, torch.mm for fp32.
    
    a: [M, K], b: [K, N] → output: [M, N]
    """
    tuner = _get_flydsl_tuner()
    use_tuner = (
        tuner and
        a.dtype in (torch.float16, torch.bfloat16) and
        not a.requires_grad and
        not b.requires_grad
    )
    if use_tuner:
        m, k = a.shape
        n = b.shape[1]
        # FlyDSL interface: C = A @ B^T, so B must be [N, K]
        b_t = b.t().contiguous()  # [K, N] → [N, K]
        c = torch.zeros(m, n, dtype=a.dtype, device=a.device)
        tuner.matmul(c, a.contiguous(), b_t)
        return c
    else:
        return torch.mm(a, b)

_IMPLICIT_GEMM_AVAILABLE = False
_FORCE_NATIVE = os.environ.get("SPCONV_FORCE_NATIVE", "0") == "1"
_USE_EXPLICIT_BWD = _FORCE_NATIVE or os.environ.get("SPCONV_USE_EXPLICIT_BWD", "0") == "1"
_USE_FUSED_FORWARD = not _FORCE_NATIVE and os.environ.get("SPCONV_ENABLE_FUSED_FORWARD", "1") != "0"
# Prefer the fused FP32 backward path by default.  Set this to 0 for the
# strict Native gather -> GEMM -> scatter numerical route.
_ALLOW_INEXACT_FUSED_FP32 = (
    not _FORCE_NATIVE and
    os.environ.get("SPCONV_ALLOW_INEXACT_FUSED_FP32", "1") != "0"
)


def _custom_fwd_cuda(fwd):
    amp = getattr(torch, "amp", None)
    custom_fwd = getattr(amp, "custom_fwd", None)
    if custom_fwd is not None:
        try:
            return custom_fwd(fwd, device_type="cuda")
        except TypeError:
            pass
    return torch.cuda.amp.custom_fwd(fwd)


def _custom_bwd_cuda(bwd):
    amp = getattr(torch, "amp", None)
    custom_bwd = getattr(amp, "custom_bwd", None)
    if custom_bwd is not None:
        try:
            return custom_bwd(bwd, device_type="cuda")
        except TypeError:
            pass
    return torch.cuda.amp.custom_bwd(bwd)


def _get_cuda_autocast_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        pass
    try:
        return torch.get_autocast_gpu_dtype()
    except AttributeError:
        return None


def _should_use_amp_native_path(features: torch.Tensor,
                                filters: torch.Tensor) -> bool:
    if not features.is_cuda or not torch.is_autocast_enabled():
        return False
    if os.environ.get("SPCONV_USE_AMP_NATIVE_INPUTS", "0") != "1":
        return False
    amp_dtype = _get_cuda_autocast_dtype()
    if amp_dtype not in (torch.float16, torch.bfloat16):
        return False
    return features.dtype == torch.float32 and filters.dtype == torch.float32


def _should_cast_output_for_autocast(features: torch.Tensor,
                                     filters: torch.Tensor) -> bool:
    if not features.is_cuda or not torch.is_autocast_enabled():
        return False
    if os.environ.get("SPCONV_CAST_OUTPUT_FOR_AMP", "0") != "1":
        return False
    amp_dtype = _get_cuda_autocast_dtype()
    if amp_dtype not in (torch.float16, torch.bfloat16):
        return False
    return features.dtype == torch.float32 and filters.dtype == torch.float32


def _cast_output_for_autocast(out: torch.Tensor,
                              features: torch.Tensor,
                              filters: torch.Tensor) -> torch.Tensor:
    if not _should_cast_output_for_autocast(features, filters):
        return out
    amp_dtype = _get_cuda_autocast_dtype()
    if out.dtype == amp_dtype:
        return out
    return out.to(dtype=amp_dtype)


def _maybe_to_indice_conv_layout(features: torch.Tensor,
                                 filters: torch.Tensor) -> torch.Tensor:
    """Accept public KRSC filters while keeping internal filter layouts unchanged."""
    if filters.ndim < 2:
        return filters

    in_channels = features.shape[1]

    # Internal layouts consumed by indice_conv kernels.
    if filters.ndim == 2 and filters.shape[0] == in_channels:
        return filters
    if filters.ndim == 3 and filters.shape[1] == in_channels:
        return filters

    # Public spconv KRSC layout: [C_out, *kernel_size, C_in].
    if filters.ndim >= 3 and filters.shape[-1] == in_channels:
        out_channels = filters.shape[0]
        kernel_volume = int(np.prod(tuple(filters.shape[1:-1])))
        if kernel_volume == 1:
            return filters.reshape(out_channels, in_channels).t().contiguous()
        perm = list(range(1, filters.ndim - 1)) + [filters.ndim - 1, 0]
        return filters.permute(*perm).contiguous().view(
            kernel_volume, in_channels, out_channels)

    return filters


def _is_public_krsc_filter(features: torch.Tensor,
                           filters: torch.Tensor) -> bool:
    if filters.ndim < 3:
        return False
    in_channels = features.shape[1]
    if filters.shape[-1] != in_channels:
        return False
    if filters.ndim == 3 and filters.shape[1] == in_channels:
        return False
    if filters.ndim == 2 and filters.shape[0] == in_channels:
        return False
    return True


def _grad_to_public_filter_layout(grad_filters: torch.Tensor,
                                  ref_filters: torch.Tensor) -> torch.Tensor:
    if ref_filters.ndim < 3:
        return grad_filters
    if grad_filters.ndim == 2:
        return grad_filters.t().contiguous().view_as(ref_filters)
    kernel_dims = list(ref_filters.shape[1:-1])
    out_channels = ref_filters.shape[0]
    in_channels = ref_filters.shape[-1]
    return grad_filters.view(*kernel_dims, in_channels, out_channels).permute(
        len(kernel_dims) + 1, *range(len(kernel_dims)), len(kernel_dims)
    ).contiguous()


def coalesce_duplicate_indices(input_tensor,
                               features: torch.Tensor,
                               indices: torch.Tensor,
                               spatial_shape: List[int]) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Coalesce duplicate sparse coordinates by summing features.

    Sparse convolution is linear in input features, so duplicate coordinates can
    be summed once and processed as a unique-coordinate tensor. Integer metadata
    is cached on the SparseConvTensor; feature aggregation is recomputed every
    call so autograd sees the current feature tensor.
    """
    if os.environ.get("SPCONV_COALESCE_DUPLICATE_INDICES", "1") == "0":
        return features, indices, False
    if not (features.is_cuda and indices.is_cuda):
        return features, indices, False
    if indices.dtype != torch.int32 or indices.ndim != 2 or features.ndim != 2:
        return features, indices, False
    if features.shape[0] != indices.shape[0] or indices.shape[0] == 0:
        return features, indices, False
    min_n = int(os.environ.get("SPCONV_COALESCE_DUP_MIN_N", "4096"))
    if indices.shape[0] < min_n:
        return features, indices, False

    spatial_shape_tuple = tuple(int(v) for v in spatial_shape)
    if len(spatial_shape_tuple) != indices.shape[1] - 1:
        return features, indices, False

    # Avoid int64 overflow in the linearized coordinate key.
    vol = 1
    for dim in spatial_shape_tuple:
        vol *= dim
    if vol <= 0 or vol >= (1 << 62):
        return features, indices, False

    indices_ptr = indices.data_ptr()
    cache_valid = (
        getattr(input_tensor, "_coalesced_indices_ptr", None) == indices_ptr and
        getattr(input_tensor, "_coalesced_indices_numel", None) == indices.numel() and
        getattr(input_tensor, "_coalesced_spatial_shape", None) == spatial_shape_tuple
    )
    if cache_valid:
        if getattr(input_tensor, "_coalesced_is_unique", None) is True:
            return features, indices, True
        cached_indices = getattr(input_tensor, "_coalesced_indices_cache", None)
        cached_inverse = getattr(input_tensor, "_coalesced_inverse_cache", None)
        if (cached_indices is not None and cached_inverse is not None and
                cached_inverse.numel() == features.shape[0]):
            out = torch.zeros((cached_indices.shape[0], features.shape[1]),
                              dtype=features.dtype, device=features.device)
            out.index_add_(0, cached_inverse, features)
            return out, cached_indices, True

    keys = indices[:, 0].to(torch.int64)
    for axis, dim in enumerate(spatial_shape_tuple):
        keys = keys * dim + indices[:, axis + 1].to(torch.int64)
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    input_tensor._coalesced_indices_ptr = indices_ptr
    input_tensor._coalesced_indices_numel = indices.numel()
    input_tensor._coalesced_spatial_shape = spatial_shape_tuple

    if unique_keys.numel() == indices.shape[0]:
        input_tensor._coalesced_indices_cache = indices
        input_tensor._coalesced_inverse_cache = None
        input_tensor._coalesced_is_unique = True
        return features, indices, True

    decoded = torch.empty((unique_keys.numel(), indices.shape[1]),
                          dtype=indices.dtype, device=indices.device)
    rem = unique_keys
    for axis in range(len(spatial_shape_tuple) - 1, -1, -1):
        dim = spatial_shape_tuple[axis]
        decoded[:, axis + 1] = (rem % dim).to(indices.dtype)
        rem = rem // dim
    decoded[:, 0] = rem.to(indices.dtype)

    out = torch.zeros((unique_keys.numel(), features.shape[1]),
                      dtype=features.dtype, device=features.device)
    out.index_add_(0, inverse, features)

    input_tensor._coalesced_indices_cache = decoded
    input_tensor._coalesced_inverse_cache = inverse
    input_tensor._coalesced_is_unique = False
    return out, decoded, True

def _try_implicit_gemm(features, filters, indice_pairs, indice_pair_num, num_activate_out):
    """Try cumm-rocm implicit GEMM only when explicitly enabled.

    Default is disabled to avoid FlyDSL `kernels` dependency warnings in
    environments that only need native sparse conv path.
    Enable by setting SPCONV_ENABLE_IMPLICIT_GEMM=1.
    """
    global _IMPLICIT_GEMM_AVAILABLE
    if _IMPLICIT_GEMM_AVAILABLE is False:
        if os.environ.get("SPCONV_ENABLE_IMPLICIT_GEMM", "0") != "1":
            return None
        _IMPLICIT_GEMM_AVAILABLE = None
    try:
        from cumm.implicit_gemm import implicit_gemm_forward
        _IMPLICIT_GEMM_AVAILABLE = True
        return implicit_gemm_forward(features, filters, indice_pairs,
                                     indice_pair_num, num_activate_out)
    except ImportError:
        _IMPLICIT_GEMM_AVAILABLE = False
        return None
    except Exception:
        return None


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] *
                (kernel_size[i] - 1) - 1) // stride[i] + 1
        if size < 0:
            size = 0
        output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation,
                           output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + dilation[i] * (
            kernel_size[i] - 1) + output_padding[i] + 1
        output_size.append(size)
    return output_size


###############################################################################
# GPU indice pairs via PyTorch native ops (sort + searchsorted)
# Mirrors original CUDA algorithm without custom kernels.
###############################################################################

def _coord_to_key(coords: torch.Tensor, spatial_shape: List[int],
                  batch_size: int) -> torch.Tensor:
    """Encode (batch, z, y, x, ...) into a single int64 scalar key.

    Mimics the original LinearLayout / LayoutNPQ:
        key = batch * prod(spatial_shape) + z * (Y*X) + y * X + x

    Args:
        coords: [N, ndim+1] int tensor (batch, spatial_dims...)
        spatial_shape: [ndim] spatial extents
    Returns:
        keys: [N] int64 tensor
    """
    ndim = len(spatial_shape)
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * spatial_shape[d + 1]
    vol = strides[0] * spatial_shape[0]

    keys = coords[:, 0].to(torch.int64) * vol
    for d in range(ndim):
        keys = keys + coords[:, d + 1].to(torch.int64) * strides[d]
    return keys


def _build_hash_table(keys: torch.Tensor):
    """Build a sort-based lookup table from keys.

    Returns:
        sorted_keys: sorted unique keys
        sorted_vals: original indices corresponding to sorted_keys
    """
    sorted_keys, sorted_indices = torch.sort(keys)
    return sorted_keys, sorted_indices


def _hash_lookup(sorted_keys: torch.Tensor, sorted_vals: torch.Tensor,
                 query_keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lookup query_keys in a sorted hash table.

    Returns:
        found_vals: values for found keys (index into original array), -1 if not found
        found_mask: bool tensor, True where key was found
    """
    pos = torch.searchsorted(sorted_keys, query_keys)
    pos = pos.clamp(max=sorted_keys.shape[0] - 1)
    found_mask = sorted_keys[pos] == query_keys
    found_vals = torch.where(found_mask, sorted_vals[pos],
                             torch.tensor(-1, dtype=sorted_vals.dtype,
                                          device=sorted_vals.device))
    return found_vals, found_mask


def _get_kernel_offsets(ksize: List[int]) -> torch.Tensor:
    """Generate kernel offset tuples as a tensor.

    Returns: [kv, ndim] int tensor with all kernel offsets.
    """
    ranges = [torch.arange(k) for k in ksize]
    grids = torch.meshgrid(*ranges, indexing='ij')
    return torch.stack([g.reshape(-1) for g in grids], dim=1)  # [kv, ndim]


def _get_indice_pairs_gpu(indices: torch.Tensor,
                          batch_size: int,
                          spatial_shape: List[int],
                          ksize: List[int],
                          stride: List[int],
                          padding: List[int],
                          dilation: List[int],
                          out_padding: List[int],
                          subm: bool,
                          transposed: bool):
    """GPU implementation of indice pairs using sort + searchsorted.

    Mirrors the original two-stage CUDA algorithm but uses only PyTorch ops.
    """
    ndim = len(spatial_shape)
    kv = int(np.prod(ksize))
    device = indices.device
    N = indices.shape[0]

    offsets = _get_kernel_offsets(ksize).to(device)  # [kv, ndim]

    if subm:
        return _get_indice_pairs_subm_gpu(
            indices, batch_size, spatial_shape, ksize, dilation,
            offsets, device)
    else:
        return _get_indice_pairs_conv_gpu(
            indices, batch_size, spatial_shape, ksize, stride,
            padding, dilation, out_padding, offsets, transposed, device)


def _get_indice_pairs_subm_gpu(indices, batch_size, spatial_shape, ksize,
                                dilation, offsets, device):
    """SubM conv: input == output, build hash on input, query neighbors."""
    ndim = len(spatial_shape)
    kv = offsets.shape[0]
    N = indices.shape[0]

    in_keys = _coord_to_key(indices, spatial_shape, batch_size)
    sorted_keys, sorted_vals = _build_hash_table(in_keys)

    center = kv // 2
    half_ksize = torch.tensor([k // 2 for k in ksize], device=device)

    indice_pairs = torch.full((kv, 2, N), -1, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    # Center kernel position: identity mapping
    arange_n = torch.arange(N, dtype=torch.int32, device=device)
    indice_pairs[center, 0, :N] = arange_n
    indice_pairs[center, 1, :N] = arange_n
    indice_pair_num[center] = N

    in_spatial = indices[:, 1:]  # [N, ndim]

    for kid in range(kv):
        if kid == center:
            continue
        offset = offsets[kid]  # [ndim]
        neighbor_spatial = in_spatial + (offset - half_ksize) * torch.tensor(
            dilation, device=device)

        valid_bounds = (neighbor_spatial >= 0).all(dim=1)
        for d in range(ndim):
            valid_bounds = valid_bounds & (neighbor_spatial[:, d] < spatial_shape[d])

        if not valid_bounds.any():
            continue

        valid_idx = valid_bounds.nonzero(as_tuple=True)[0]
        neighbor_coords = torch.cat([
            indices[valid_idx, 0:1],
            neighbor_spatial[valid_idx]
        ], dim=1)

        query_keys = _coord_to_key(neighbor_coords, spatial_shape, batch_size)
        found_vals, found_mask = _hash_lookup(sorted_keys, sorted_vals, query_keys)

        actual_found = found_mask.nonzero(as_tuple=True)[0]
        n_found = actual_found.shape[0]
        if n_found == 0:
            continue

        out_indices = valid_idx[actual_found]  # output point indices
        in_indices = found_vals[actual_found]  # input point indices

        indice_pairs[kid, 0, :n_found] = in_indices.int()
        indice_pairs[kid, 1, :n_found] = out_indices.int()
        indice_pair_num[kid] = n_found

        # Exploit symmetry: mirror kernel position
        mirror_kid = kv - 1 - kid
        if mirror_kid != kid and mirror_kid != center:
            indice_pairs[mirror_kid, 0, :n_found] = out_indices.int()
            indice_pairs[mirror_kid, 1, :n_found] = in_indices.int()
            indice_pair_num[mirror_kid] = n_found

    return indices, indice_pairs, indice_pair_num


def _get_indice_pairs_conv_gpu(indices, batch_size, spatial_shape, ksize,
                                stride, padding, dilation, out_padding,
                                offsets, transposed, device):
    """Regular/transposed conv: 2-stage approach.

    Stage 1: for each input × offset → compute output coord, collect unique outputs.
    Stage 2: build hash on unique outputs, lookup to fill indice_pairs[1].
    """
    ndim = len(spatial_shape)
    kv = offsets.shape[0]
    N = indices.shape[0]

    if transposed:
        out_spatial_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                                    padding, dilation, out_padding)
    else:
        out_spatial_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                                  padding, dilation)

    padding_t = torch.tensor(padding, device=device)
    stride_t = torch.tensor(stride, device=device)
    dilation_t = torch.tensor(dilation, device=device)
    out_spatial_t = torch.tensor(out_spatial_shape, device=device)

    in_spatial = indices[:, 1:]  # [N, ndim]
    batch_ids = indices[:, 0]  # [N]

    all_in_idx = []
    all_out_keys = []
    all_kid = []

    for kid in range(kv):
        offset = offsets[kid]  # [ndim]
        if transposed:
            out_sp = (in_spatial + padding_t - offset * dilation_t) * stride_t + offset * dilation_t
        else:
            out_sp = (in_spatial + padding_t - offset * dilation_t).div(
                stride_t, rounding_mode='floor')
            remainder = (in_spatial + padding_t - offset * dilation_t) % stride_t
            stride_valid = (remainder == 0).all(dim=1)

        valid = torch.ones(N, dtype=torch.bool, device=device)
        if not transposed:
            valid = valid & stride_valid
        for d in range(ndim):
            valid = valid & (out_sp[:, d] >= 0) & (out_sp[:, d] < out_spatial_t[d])

        if not valid.any():
            continue

        valid_idx = valid.nonzero(as_tuple=True)[0]
        out_coords = torch.cat([
            batch_ids[valid_idx].unsqueeze(1),
            out_sp[valid_idx]
        ], dim=1).int()

        out_keys = _coord_to_key(out_coords, out_spatial_shape, batch_size)

        all_in_idx.append(valid_idx)
        all_out_keys.append(out_keys)
        all_kid.append(torch.full((valid_idx.shape[0],), kid,
                                  dtype=torch.int32, device=device))

    if len(all_in_idx) == 0:
        out_inds = torch.zeros((0, ndim + 1), dtype=indices.dtype, device=device)
        indice_pairs = torch.full((kv, 2, 1), -1, dtype=torch.int32, device=device)
        indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)
        return out_inds, indice_pairs, indice_pair_num

    cat_in_idx = torch.cat(all_in_idx)
    cat_out_keys = torch.cat(all_out_keys)
    cat_kid = torch.cat(all_kid)

    # Stage 1.5: unique output coords
    unique_out_keys, inverse = torch.unique(cat_out_keys, return_inverse=True)
    num_out = unique_out_keys.shape[0]

    # Decode unique keys back to coordinates
    out_inds = torch.zeros((num_out, ndim + 1), dtype=indices.dtype, device=device)
    vol_strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        vol_strides[d] = vol_strides[d + 1] * out_spatial_shape[d + 1]
    vol = vol_strides[0] * out_spatial_shape[0]

    remainder = unique_out_keys.clone()
    out_inds[:, 0] = (remainder // vol).to(indices.dtype)
    remainder = remainder % vol
    for d in range(ndim):
        out_inds[:, d + 1] = (remainder // vol_strides[d]).to(indices.dtype)
        remainder = remainder % vol_strides[d]

    # Stage 2: fill indice_pairs using inverse mapping for output indices
    out_idx_mapped = inverse.int()  # maps each pair to unique output index
    N_max = max(N, num_out)
    indice_pairs = torch.full((kv, 2, N_max), -1, dtype=torch.int32, device=device)
    indice_pair_num = torch.zeros(kv, dtype=torch.int32, device=device)

    for kid in range(kv):
        mask = cat_kid == kid
        if not mask.any():
            continue
        kid_in_idx = cat_in_idx[mask].int()
        kid_out_idx = out_idx_mapped[mask]
        n_pairs = kid_in_idx.shape[0]
        indice_pairs[kid, 0, :n_pairs] = kid_in_idx
        indice_pairs[kid, 1, :n_pairs] = kid_out_idx
        indice_pair_num[kid] = n_pairs

    return out_inds, indice_pairs, indice_pair_num


###############################################################################
# Public API
###############################################################################

def _get_indice_pairs_hip(indices: torch.Tensor,
                          batch_size: int,
                          spatial_shape: List[int],
                          ksize: List[int],
                          stride: List[int],
                          padding: List[int],
                          dilation: List[int],
                          out_padding: List[int],
                          subm: bool,
                          transposed: bool):
    """Dispatch to JIT-compiled HIP kernel for indice pairs.

    Returns same format as _get_indice_pairs_gpu:
        out_inds: [N_out, ndim+1]
        indice_pairs: [kv, 2, N_max]
        indice_pair_num: [kv]
    """
    hip = _get_hip_module()
    indices_i32 = indices.int().contiguous()

    if subm:
        # SubM: output == input
        ip, ip_num = hip.get_indice_pairs_subm(
            indices_i32, batch_size, spatial_shape, ksize, dilation)
        # HIP kernels already return contiguous [kv, 2, N].
        return indices, ip, ip_num
    else:
        # Regular / transposed conv
        if transposed:
            out_spatial = get_deconv_output_size(
                spatial_shape, ksize, stride, padding, dilation, out_padding)
        else:
            out_spatial = get_conv_output_size(
                spatial_shape, ksize, stride, padding, dilation)
        out_inds, ip, ip_num = hip.get_indice_pairs_conv(
            indices_i32, batch_size, spatial_shape, out_spatial,
            ksize, stride, padding, dilation, transposed)
        # HIP kernels already return contiguous [kv, 2, N].
        return out_inds, ip, ip_num


class _IndiceConvFunction(Function):
    @staticmethod
    @_custom_fwd_cuda
    def forward(ctx,
                features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool,
                subm: bool) -> torch.Tensor:
        hip = _get_hip_module()
        if hip is None or not features.is_cuda:
            raise RuntimeError("_IndiceConvFunction requires HIP extension on device")

        out = hip.indice_conv_forward(
            features, filters, indice_pairs, indice_pair_num,
            num_activate_out, inverse, subm)
        ctx.save_for_backward(features, filters, indice_pairs, indice_pair_num)
        ctx.inverse = inverse
        ctx.subm = subm
        return out

    @staticmethod
    @_custom_bwd_cuda
    def backward(ctx, out_bp: torch.Tensor):
        features, filters, indice_pairs, indice_pair_num = ctx.saved_tensors
        din = dfilters = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            hip = _get_hip_module()
            if hip is None:
                raise RuntimeError("_IndiceConvFunction backward requires HIP extension")
            din, dfilters = hip.indice_conv_backward(
                features, filters, out_bp.contiguous(),
                indice_pairs, indice_pair_num, ctx.inverse, ctx.subm)
            if not ctx.needs_input_grad[0]:
                din = None
            if not ctx.needs_input_grad[1]:
                dfilters = None
        return din, dfilters, None, None, None, None, None, None


class _IndiceConvFusedFunction(Function):
    @staticmethod
    @_custom_fwd_cuda
    def forward(ctx,
                features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool,
                subm: bool,
                assume_unique_indices: bool) -> torch.Tensor:
        hip = _get_hip_module()
        if hip is None or not features.is_cuda:
            raise RuntimeError("_IndiceConvFusedFunction requires HIP extension on device")

        if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_unique"):
            out = hip.indice_conv_forward_fused_unique(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        else:
            out = hip.indice_conv_forward_fused(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        ctx.save_for_backward(features, filters, indice_pairs, indice_pair_num)
        ctx.inverse = inverse
        ctx.subm = subm
        return out

    @staticmethod
    @_custom_bwd_cuda
    def backward(ctx, out_bp: torch.Tensor):
        features, filters, indice_pairs, indice_pair_num = ctx.saved_tensors
        din = dfilters = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            hip = _get_hip_module()
            if hip is None:
                raise RuntimeError("_IndiceConvFusedFunction backward requires HIP extension")
            din, dfilters = hip.indice_conv_backward(
                features, filters, out_bp.contiguous(),
                indice_pairs, indice_pair_num, ctx.inverse, ctx.subm)
            if not ctx.needs_input_grad[0]:
                din = None
            if not ctx.needs_input_grad[1]:
                dfilters = None
        return din, dfilters, None, None, None, None, None, None


def get_indice_pairs(indices: torch.Tensor,
                     batch_size: int,
                     spatial_shape: List[int],
                     algo: ConvAlgo,
                     ksize: List[int],
                     stride: List[int],
                     padding: List[int],
                     dilation: List[int],
                     out_padding: List[int],
                     subm: bool = False,
                     transposed: bool = False,
                     is_train: bool = True,
                     alloc=None,
                     timer=None):
    """Compute indice pairs for sparse convolution.

    Primary: HIP kernel (GPU hash table, single-launch 2D grid).
    Fallback: PyTorch native ops (sort + searchsorted).

    Returns:
        out_inds: output indices [N_out, ndim+1]
        indice_pairs: [kv, 2, N_max] — gather/scatter index pairs
        indice_pair_num: [kv] — number of active pairs per kernel position
    """
    if indices.is_cuda and _get_hip_module() is not None:
        return _get_indice_pairs_hip(
            indices, batch_size, spatial_shape, ksize, stride,
            padding, dilation, out_padding, subm, transposed)
    return _get_indice_pairs_gpu(
        indices, batch_size, spatial_shape, ksize, stride,
        padding, dilation, out_padding, subm, transposed)



class _IndiceConvFusedF16Function(Function):
    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm, assume_unique_indices):
        hip = _get_hip_module()
        if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_f16_unique"):
            out = hip.indice_conv_forward_fused_f16_unique(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        else:
            out = hip.indice_conv_forward_fused_f16(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        ctx.save_for_backward(features, filters, indice_pairs, indice_pair_num)
        ctx.inverse = inverse
        ctx.subm = subm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        features, filters, indice_pairs, indice_pair_num = ctx.saved_tensors
        hip = _get_hip_module()
        grad_output = grad_output.contiguous()
        din, dfilters = hip.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            ctx.inverse, ctx.subm)
        return din, dfilters, None, None, None, None, None, None


class _IndiceConvFusedBF16Function(Function):
    @staticmethod
    def forward(ctx, features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm, assume_unique_indices):
        hip = _get_hip_module()
        if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_bf16_unique"):
            out = hip.indice_conv_forward_fused_bf16_unique(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        else:
            out = hip.indice_conv_forward_fused_bf16(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        ctx.save_for_backward(features, filters, indice_pairs, indice_pair_num)
        ctx.inverse = inverse
        ctx.subm = subm
        return out

    @staticmethod
    def backward(ctx, grad_output):
        features, filters, indice_pairs, indice_pair_num = ctx.saved_tensors
        hip = _get_hip_module()
        grad_output = grad_output.contiguous()
        din, dfilters = hip.indice_conv_backward(
            features, filters, grad_output, indice_pairs, indice_pair_num,
            ctx.inverse, ctx.subm)
        return din, dfilters, None, None, None, None, None, None


def indice_conv(features: torch.Tensor,
                filters: torch.Tensor,
                indice_pairs: torch.Tensor,
                indice_pair_num: torch.Tensor,
                num_activate_out: int,
                inverse: bool = False,
                subm: bool = False,
                algo: ConvAlgo = ConvAlgo.Native,
                timer=None,
                assume_unique_indices: bool = False) -> torch.Tensor:
    """Sparse convolution dispatch: implicit GEMM → native gather+GEMM+scatter.

    Args:
        features: [N_in, C_in] input features
        filters: internal [kv, C_in, C_out] / [C_in, C_out] or public
            KRSC [C_out, *kernel_size, C_in]
        indice_pairs: [kv, 2, N_max] gather/scatter indices
        indice_pair_num: [kv] active pair count per kernel position
        num_activate_out: number of output points

    Returns:
        out_features: [N_out, C_out]
    """
    filters = _maybe_to_indice_conv_layout(features, filters)
    if (not inverse) and features.is_cuda and not features.requires_grad:
        out = _try_implicit_gemm(features, filters, indice_pairs,
                                 indice_pair_num, num_activate_out)
        if out is not None:
            return _cast_output_for_autocast(out, features, filters)

    hip = _get_hip_module()
    if _should_use_amp_native_path(features, filters):
        amp_dtype = _get_cuda_autocast_dtype()
        features_amp = features.to(dtype=amp_dtype)
        filters_amp = filters.to(dtype=amp_dtype)
        if hip is not None and features.is_cuda:
            if _USE_EXPLICIT_BWD and (features.requires_grad or filters.requires_grad):
                return _IndiceConvFunction.apply(
                    features_amp, filters_amp, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            return hip.indice_conv_forward(
                features_amp, filters_amp, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
        features = features_amp
        filters = filters_amp
    if hip is not None and features.is_cuda:
        if (_USE_FUSED_FORWARD and _ALLOW_INEXACT_FUSED_FP32 and features.dtype == torch.float32 and
                filters.dtype == torch.float32):
            if features.requires_grad or filters.requires_grad:
                out = _IndiceConvFusedFunction.apply(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm, assume_unique_indices)
                return _cast_output_for_autocast(out, features, filters)
            if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_unique"):
                out = hip.indice_conv_forward_fused_unique(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            else:
                out = hip.indice_conv_forward_fused(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            return _cast_output_for_autocast(out, features, filters)
        if (_USE_FUSED_FORWARD and features.dtype == torch.float16 and
                filters.dtype == torch.float16 and
                hasattr(hip, "indice_conv_forward_fused_f16")):
            if features.requires_grad or filters.requires_grad:
                out = _IndiceConvFusedF16Function.apply(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm, assume_unique_indices)
                return _cast_output_for_autocast(out, features, filters)
            if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_f16_unique"):
                out = hip.indice_conv_forward_fused_f16_unique(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            else:
                out = hip.indice_conv_forward_fused_f16(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            return _cast_output_for_autocast(out, features, filters)
        if (_USE_FUSED_FORWARD and features.dtype == torch.bfloat16 and
                filters.dtype == torch.bfloat16 and
                hasattr(hip, "indice_conv_forward_fused_bf16")):
            if features.requires_grad or filters.requires_grad:
                out = _IndiceConvFusedBF16Function.apply(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm, assume_unique_indices)
                return _cast_output_for_autocast(out, features, filters)
            if assume_unique_indices and hasattr(hip, "indice_conv_forward_fused_bf16_unique"):
                out = hip.indice_conv_forward_fused_bf16_unique(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            else:
                out = hip.indice_conv_forward_fused_bf16(
                    features, filters, indice_pairs, indice_pair_num,
                    num_activate_out, inverse, subm)
            return _cast_output_for_autocast(out, features, filters)
        if _USE_EXPLICIT_BWD and (features.requires_grad or filters.requires_grad):
            out = _IndiceConvFunction.apply(
                features, filters, indice_pairs, indice_pair_num,
                num_activate_out, inverse, subm)
            return _cast_output_for_autocast(out, features, filters)
        out = hip.indice_conv_forward(
            features, filters, indice_pairs, indice_pair_num,
            num_activate_out, inverse, subm)
        return _cast_output_for_autocast(out, features, filters)

    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    center = kv // 2
    device = features.device
    dtype = features.dtype

    out_features = torch.zeros(num_activate_out, c_out, dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue

        if subm and (not inverse) and i == center:
            gemm_out = _gemm(features, filters[i])
            if gemm_out.dtype != out_features.dtype:
                gemm_out = gemm_out.to(out_features.dtype)
            out_features = out_features + gemm_out
            continue

        if inverse:
            # Inverse conv reuses forward-conv indice pairs, but reverses
            # gather/scatter direction: current input corresponds to pair[1].
            inp_inds = indice_pairs[i, 1, :n_act].long()
            out_inds = indice_pairs[i, 0, :n_act].long()
        else:
            inp_inds = indice_pairs[i, 0, :n_act].long()
            out_inds = indice_pairs[i, 1, :n_act].long()

        # gather input features
        inp_gathered = features[inp_inds]  # [n_act, C_in]

        # GEMM: [n_act, C_in] @ [C_in, C_out] → [n_act, C_out]
        w = filters[i]  # [C_in, C_out]
        gemm_out = _gemm(inp_gathered, w)

        # scatter-add to output (cast if AMP autocast changed dtype)
        if gemm_out.dtype != out_features.dtype:
            gemm_out = gemm_out.to(out_features.dtype)
        out_features.index_add_(0, out_inds, gemm_out)

    return _cast_output_for_autocast(out_features, features, filters)


def indice_conv_backward(features: torch.Tensor,
                         filters: torch.Tensor,
                         out_bp: torch.Tensor,
                         indice_pairs: torch.Tensor,
                         indice_pair_num: torch.Tensor,
                         inverse: bool = False,
                         subm: bool = False,
                         algo: ConvAlgo = ConvAlgo.Native,
                         timer=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward pass for native sparse conv.
    
    Returns:
        din: gradient w.r.t. input features [N_in, C_in]
        dfilters: gradient w.r.t. filters [kv, C_in, C_out]
    """
    filters_were_public_krsc = _is_public_krsc_filter(features, filters)
    filters_ref = filters
    filters = _maybe_to_indice_conv_layout(features, filters)
    hip = _get_hip_module()
    if hip is not None and features.is_cuda:
        din, dfilters = hip.indice_conv_backward(
            features, filters, out_bp, indice_pairs, indice_pair_num,
            inverse, subm)
        if filters_were_public_krsc:
            dfilters = _grad_to_public_filter_layout(dfilters, filters_ref)
        return din, dfilters

    kv = filters.shape[0]
    c_in = features.shape[1]
    c_out = filters.shape[2]
    center = kv // 2
    n_in = features.shape[0]
    device = features.device
    dtype = features.dtype

    din = torch.zeros(n_in, c_in, dtype=dtype, device=device)
    dfilters = torch.zeros_like(filters)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue

        if subm and (not inverse) and i == center:
            dfilters[i] = _gemm(features.t(), out_bp)
            din = din + _gemm(out_bp, filters[i].t())
            continue

        if inverse:
            inp_inds = indice_pairs[i, 1, :n_act].long()
            out_inds = indice_pairs[i, 0, :n_act].long()
        else:
            inp_inds = indice_pairs[i, 0, :n_act].long()
            out_inds = indice_pairs[i, 1, :n_act].long()

        # gather
        inp_gathered = features[inp_inds]  # [n_act, C_in]
        out_bp_gathered = out_bp[out_inds]  # [n_act, C_out]

        w = filters[i]  # [C_in, C_out]

        # dfilters[i] = inp_gathered^T @ out_bp_gathered
        dfilters[i] = _gemm(inp_gathered.t(), out_bp_gathered)

        # din: scatter grad back to input
        # din_gathered = out_bp_gathered @ w^T
        din_gathered = _gemm(out_bp_gathered, w.t())
        din.index_add_(0, inp_inds, din_gathered)

    if filters_were_public_krsc:
        dfilters = _grad_to_public_filter_layout(dfilters, filters_ref)
    return din, dfilters


def indice_maxpool(features: torch.Tensor,
                   indice_pairs: torch.Tensor,
                   indice_pair_num: torch.Tensor,
                   num_activate_out: int) -> torch.Tensor:
    """Max pooling over sparse indices."""
    c = features.shape[1]
    kv = indice_pairs.shape[0]
    device = features.device
    dtype = features.dtype

    out_features = torch.full((num_activate_out, c), float('-inf'),
                              dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue
        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()
        inp_gathered = features[inp_inds]
        out_features[out_inds] = torch.maximum(out_features[out_inds], inp_gathered)

    # Replace -inf with 0 for positions with no input
    out_features = torch.where(out_features == float('-inf'),
                               torch.zeros_like(out_features), out_features)
    return out_features


def indice_maxpool_backward(features: torch.Tensor,
                            out_features: torch.Tensor,
                            out_bp: torch.Tensor,
                            indice_pairs: torch.Tensor,
                            indice_pair_num: torch.Tensor) -> torch.Tensor:
    """Backward for sparse max pooling."""
    n_in = features.shape[0]
    c = features.shape[1]
    kv = indice_pairs.shape[0]
    device = features.device
    dtype = features.dtype

    din = torch.zeros(n_in, c, dtype=dtype, device=device)

    for i in range(kv):
        n_act = int(indice_pair_num[i].item())
        if n_act == 0:
            continue
        inp_inds = indice_pairs[i, 0, :n_act].long()
        out_inds = indice_pairs[i, 1, :n_act].long()

        inp_gathered = features[inp_inds]
        out_gathered = out_features[out_inds]
        mask = (inp_gathered == out_gathered).float()
        din.index_add_(0, inp_inds, mask * out_bp[out_inds])

    return din


def global_pool_rearrange(coords: torch.Tensor, batch_size: int):
    """Rearrange coords by batch for global pooling."""
    batch_indices = coords[:, 0]
    counts = torch.zeros(batch_size, dtype=torch.int64, device=coords.device)
    for b in range(batch_size):
        counts[b] = (batch_indices == b).sum()
    return counts
