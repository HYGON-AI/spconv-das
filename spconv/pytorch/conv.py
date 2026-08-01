# spconv-rocm: Sparse convolution layers (ROCm, pure Python, Native path only)

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter

from spconv.core import ConvAlgo
from spconv.pytorch.core import SparseConvTensor, IndiceData, expand_nd
from spconv.pytorch import ops
from spconv.pytorch.ops import _gemm
from spconv.pytorch.modules import SparseModule
from spconv.constants import ALL_WEIGHT_IS_KRSC


class SparseConvolution(SparseModule):
    """Base sparse convolution layer — ROCm version (Native path only)."""

    def __init__(self,
                 ndim: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 subm: bool = False,
                 output_padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 transposed: bool = False,
                 inverse: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None,
                 **kwargs):
        super().__init__()
        assert groups == 1, "groups != 1 not supported"
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = expand_nd(ndim, kernel_size)
        self.stride = expand_nd(ndim, stride)
        self.dilation = expand_nd(ndim, dilation)
        self.padding = expand_nd(ndim, padding)
        self.output_padding = expand_nd(ndim, output_padding)
        self.subm = subm
        self.transposed = transposed
        self.inverse = inverse
        self.indice_key = indice_key
        self.algo = ConvAlgo.Native  # ROCm: always Native

        kv = int(np.prod(self.kernel_size))
        self.conv1x1 = (kv == 1) and (int(np.prod(self.stride)) == 1)

        # Weight layout: [kv, C_in, C_out] (KRSC-like, each kernel position has its own weight)
        if self.conv1x1:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.weight = Parameter(torch.Tensor(kv, in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * int(np.prod(self.kernel_size))
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: SparseConvTensor) -> SparseConvTensor:
        assert input.features.shape[1] == self.in_channels

        features = input.features
        indices = input.indices
        spatial_shape = input.spatial_shape
        batch_size = input.batch_size

        if self.conv1x1:
            out_features = _gemm(features, self.weight)
            if self.bias is not None:
                out_features = out_features + self.bias
            out_tensor = input.replace_feature(out_features)
            return out_tensor

        # Compute output spatial shape
        if self.transposed:
            out_spatial_shape = ops.get_deconv_output_size(
                spatial_shape, self.kernel_size, self.stride,
                self.padding, self.dilation, self.output_padding)
        elif self.subm:
            out_spatial_shape = spatial_shape
        else:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape, self.kernel_size, self.stride,
                self.padding, self.dilation)

        # Get or compute indice pairs
        datas = input.find_indice_pair(self.indice_key) if self.indice_key else None
        if self.inverse:
            assert datas is not None and self.indice_key is not None, (
                "inverse conv requires existing indice_key from coupled forward conv"
            )
            assert not datas.is_subm, (
                "inverse conv can only be used with standard conv/pool indice data"
            )
            if datas.ksize is not None:
                assert datas.ksize == self.kernel_size, (
                    "inverse conv must have same kernel size as its couple conv"
                )
            outids = datas.indices
            indice_pairs = datas.indice_pairs
            indice_pair_num = datas.indice_pair_num
            out_spatial_shape = datas.spatial_shape
        elif datas is not None:
            outids = datas.out_indices
            indice_pairs = datas.indice_pairs
            indice_pair_num = datas.indice_pair_num
        else:
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                indices, batch_size, spatial_shape, self.algo,
                self.kernel_size, self.stride, self.padding,
                self.dilation, self.output_padding,
                subm=self.subm, transposed=self.transposed)

            if self.indice_key is not None:
                indice_data = IndiceData(
                    indices,
                    indice_pairs,
                    indice_pair_num,
                    spatial_shape,
                    self.subm,
                    self.algo,
                    out_indices=outids,
                    out_spatial_shape=out_spatial_shape,
                    ksize=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
                input.indice_dict[self.indice_key] = indice_data

        num_out = outids.shape[0]

        # Native sparse conv: gather → GEMM → scatter
        out_features = ops.indice_conv(
            features, self.weight, indice_pairs, indice_pair_num,
            num_out, inverse=self.inverse, subm=self.subm)

        if self.bias is not None:
            out_features = out_features + self.bias

        out_tensor = SparseConvTensor(out_features, outids, out_spatial_shape, batch_size)
        out_tensor.indice_dict = input.indice_dict
        return out_tensor

    def extra_repr(self):
        return (f'{self.in_channels}, {self.out_channels}, '
                f'kernel_size={self.kernel_size}, stride={self.stride}, '
                f'padding={self.padding}, dilation={self.dilation}, '
                f'subm={self.subm}, algo=Native(ROCm)')


# Convenience classes matching original spconv API
class SparseConv1d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, indice_key=indice_key, algo=algo, **kwargs)


class SparseConv2d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, indice_key=indice_key, algo=algo, **kwargs)


class SparseConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(3, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, indice_key=indice_key, algo=algo, **kwargs)


class SparseConv4d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(4, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, indice_key=indice_key, algo=algo, **kwargs)


class SparseConvTranspose1d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, output_padding=0, **kwargs):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, transposed=True, output_padding=output_padding,
                         indice_key=indice_key, algo=algo, **kwargs)


class SparseConvTranspose2d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, output_padding=0, **kwargs):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, transposed=True, output_padding=output_padding,
                         indice_key=indice_key, algo=algo, **kwargs)


class SparseConvTranspose3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, output_padding=0, **kwargs):
        super().__init__(3, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, transposed=True, output_padding=output_padding,
                         indice_key=indice_key, algo=algo, **kwargs)


class SparseConvTranspose4d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, output_padding=0, **kwargs):
        super().__init__(4, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, transposed=True, output_padding=output_padding,
                         indice_key=indice_key, algo=algo, **kwargs)


class SparseInverseConv1d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, **kwargs):
        super().__init__(1, in_channels, out_channels, kernel_size, bias=bias,
                         inverse=True, indice_key=indice_key, algo=algo, **kwargs)


class SparseInverseConv2d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, **kwargs):
        super().__init__(2, in_channels, out_channels, kernel_size, bias=bias,
                         inverse=True, indice_key=indice_key, algo=algo, **kwargs)


class SparseInverseConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, **kwargs):
        super().__init__(3, in_channels, out_channels, kernel_size, bias=bias,
                         inverse=True, indice_key=indice_key, algo=algo, **kwargs)


class SparseInverseConv4d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key, bias=True, algo=None, **kwargs):
        super().__init__(4, in_channels, out_channels, kernel_size, bias=bias,
                         inverse=True, indice_key=indice_key, algo=algo, **kwargs)


class SubMConv1d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(1, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, subm=True, indice_key=indice_key, algo=algo, **kwargs)


class SubMConv2d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(2, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, subm=True, indice_key=indice_key, algo=algo, **kwargs)


class SubMConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(3, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, subm=True, indice_key=indice_key, algo=algo, **kwargs)


class SubMConv4d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, **kwargs):
        super().__init__(4, in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, subm=True, indice_key=indice_key, algo=algo, **kwargs)
