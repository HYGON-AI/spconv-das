# spconv-rocm: Sparse pooling layers (Native path only)

import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from spconv.core import ConvAlgo
from spconv.pytorch import ops
from spconv.pytorch.core import SparseConvTensor, IndiceData, expand_nd
from spconv.pytorch.modules import SparseModule


class SparseMaxPool(SparseModule):
    """Sparse max pooling — ROCm version (Native path)."""

    def __init__(self, ndim, kernel_size, stride=1, padding=0, dilation=1,
                 indice_key=None, algo=None, **kwargs):
        super().__init__()
        self.ndim = ndim
        self.kernel_size = expand_nd(ndim, kernel_size)
        self.stride = expand_nd(ndim, stride)
        self.padding = expand_nd(ndim, padding)
        self.dilation = expand_nd(ndim, dilation)
        self.indice_key = indice_key
        self.algo = ConvAlgo.Native

    def forward(self, input: SparseConvTensor) -> SparseConvTensor:
        indices = input.indices
        features = input.features
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size

        out_spatial_shape = ops.get_conv_output_size(
            spatial_shape, self.kernel_size, self.stride,
            self.padding, self.dilation)

        outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
            indices, batch_size, spatial_shape, self.algo,
            self.kernel_size, self.stride, self.padding,
            self.dilation, [0] * self.ndim, subm=False)

        indice_dict = input.indice_dict.copy()
        if self.indice_key is not None:
            if self.indice_key in indice_dict:
                raise ValueError(f"indice key {self.indice_key} exists")
            indice_dict[self.indice_key] = IndiceData(
                indices,
                indice_pairs,
                indice_pair_num,
                spatial_shape,
                False,
                self.algo,
                out_indices=outids,
                out_spatial_shape=out_spatial_shape,
                ksize=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        num_out = outids.shape[0]
        out_features = ops.indice_maxpool(features, indice_pairs,
                                          indice_pair_num, num_out)

        out = SparseConvTensor(out_features, outids, out_spatial_shape, batch_size)
        out.indice_dict = indice_dict
        return out


class SparseMaxPool1d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(1, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseMaxPool2d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(2, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseMaxPool3d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(3, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseMaxPool4d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(4, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseAvgPool1d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(1, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseAvgPool2d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(2, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseAvgPool3d(SparseMaxPool):
    def __init__(self, kernel_size, stride=1, padding=0, dilation=1, indice_key=None, **kwargs):
        super().__init__(3, kernel_size, stride, padding, dilation, indice_key, **kwargs)


class SparseGlobalMaxPool(SparseModule):
    """Global max pool: reduce all spatial dims."""
    def forward(self, input: SparseConvTensor) -> torch.Tensor:
        features = input.features
        indices = input.indices
        batch_size = input.batch_size
        out = torch.zeros(batch_size, features.shape[1],
                          dtype=features.dtype, device=features.device)
        for b in range(batch_size):
            mask = indices[:, 0] == b
            if mask.any():
                out[b] = features[mask].max(dim=0).values
        return out


class SparseGlobalAvgPool(SparseModule):
    """Global average pool: reduce all spatial dims."""
    def forward(self, input: SparseConvTensor) -> torch.Tensor:
        features = input.features
        indices = input.indices
        batch_size = input.batch_size
        out = torch.zeros(batch_size, features.shape[1],
                          dtype=features.dtype, device=features.device)
        for b in range(batch_size):
            mask = indices[:, 0] == b
            if mask.any():
                out[b] = features[mask].mean(dim=0)
        return out
