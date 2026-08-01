# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""Test spconv-rocm sparse convolution on ROCm GPU.

Run: python -m pytest test/test_sparse_conv.py -v
Requires: PyTorch with ROCm support, AMD/HCU GPU available.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_sparse_input(batch_size, num_points, in_channels, spatial_shape, device,
                       dtype=torch.float32):
    """Create a random SparseConvTensor with unique coordinates."""
    import spconv.pytorch as spconv

    ndim = len(spatial_shape)
    coords_set = set()
    indices = []

    for b in range(batch_size):
        count = 0
        while count < num_points:
            coord = tuple(np.random.randint(0, s) for s in spatial_shape)
            key = (b,) + coord
            if key not in coords_set:
                coords_set.add(key)
                indices.append([b] + list(coord))
                count += 1

    indices = torch.tensor(indices, dtype=torch.int32, device=device)
    features = torch.randn(len(indices), in_channels, dtype=dtype, device=device)

    return spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)


class _HipCallRecorder:
    """Proxy the extension so public-interface tests prove fused dispatch."""

    def __init__(self, module, recorded_methods):
        self._module = module
        self.calls = {name: 0 for name in recorded_methods}

    def __getattr__(self, name):
        target = getattr(self._module, name)
        if name not in self.calls:
            return target

        def wrapped(*args, **kwargs):
            self.calls[name] += 1
            return target(*args, **kwargs)

        return wrapped


def _gather_dense_features(dense_features, indices):
    batch, z, y, x = (indices[:, axis].long() for axis in range(4))
    return dense_features[batch, :, z, y, x]


def _sparse_features_to_dense(features, indices, spatial_shape, batch_size):
    dense = torch.zeros(
        (batch_size, features.shape[1], *spatial_shape),
        dtype=features.dtype,
        device=features.device,
    )
    batch, z, y, x = (indices[:, axis].long() for axis in range(4))
    dense[batch, :, z, y, x] = features
    return dense


def _assert_nv_low_precision_l2(actual, reference, in_channels, out_channels, name):
    """Match NV MaskImplicitGemm FP16 acceptance: L2(diff) < 10 * multiplier."""
    assert actual.shape == reference.shape, f"{name}: shape mismatch"
    multiplier = max(1.0, max(in_channels, out_channels) / 16.0)
    error_norm = torch.norm((actual.float() - reference.float()).reshape(-1), p=2)
    limit = 10.0 * multiplier
    assert error_norm.item() < limit, (
        f"{name}: L2 error {error_norm.item():.6g} exceeds NV limit {limit:.6g}; "
        f"max_abs={(actual.float() - reference.float()).abs().max().item():.6g}"
    )


def _assert_fp32_strict(actual, reference, name):
    assert torch.allclose(actual, reference, atol=1e-4, rtol=1e-4), (
        f"{name}: exceeds FP32 tolerance (atol=rtol=1e-4), max_abs="
        f"{(actual - reference).abs().max().item():.6g}"
    )


# ============================================================
# Basic Forward Tests
# ============================================================

class TestSparseConv3d:
    def test_subm_conv_forward(self, device):
        """SubManifold conv: output indices == input indices."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        batch_size = 2
        num_points = 50

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape == (batch_size * num_points, out_ch)
        assert out.spatial_shape == spatial_shape
        assert torch.equal(out.indices, x.indices)

    def test_sparse_conv_forward(self, device):
        """Standard sparse conv: output shape changes with stride."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        batch_size = 1
        num_points = 30

        conv = spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=True).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape[1] == out_ch
        assert out.spatial_shape == [5, 5, 5]

    def test_conv1x1(self, device):
        """1x1 conv is just a dense matmul."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 32, 64
        spatial_shape = [8, 8, 8]
        batch_size = 1
        num_points = 20

        conv = spconv.SparseConv3d(in_ch, out_ch, 1, bias=False).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape == (num_points, out_ch)
        ref = torch.mm(x.features, conv.weight.reshape(conv.weight.shape[0], conv.weight.shape[-1]).t().contiguous())
        assert torch.allclose(out.features, ref, atol=1e-5)

    def test_sequential(self, device):
        """SparseSequential with conv + BN + ReLU."""
        import spconv.pytorch as spconv

        in_ch, mid_ch, out_ch = 16, 32, 64
        spatial_shape = [8, 8, 8]
        batch_size = 1
        num_points = 25

        model = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, mid_ch, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(mid_ch),
            spconv.SparseReLU(),
            spconv.SubMConv3d(mid_ch, out_ch, 3, padding=1, bias=False),
        ).to(device)

        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        out = model(x)

        assert out.features.shape == (num_points, out_ch)
        assert (out.features >= 0).any()

    def test_to_dense(self, device):
        """Convert SparseConvTensor to dense."""
        import spconv.pytorch as spconv

        in_ch = 4
        spatial_shape = [5, 5, 5]
        batch_size = 1
        num_points = 10

        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)
        dense = x.dense()

        assert dense.shape == (batch_size, in_ch, 5, 5, 5)


# ============================================================
# Numerical Correctness
# ============================================================

class TestNumericalCorrectness:
    def test_subm_conv_vs_dense(self, device):
        """SubM conv result should match dense conv on sparse input positions."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 4, 8
        spatial_shape = [7, 7, 7]
        batch_size = 1
        num_points = 30

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False).to(device)
        x = _make_sparse_input(batch_size, num_points, in_ch, spatial_shape, device)

        # Run sparse conv
        out_sparse = conv(x)

        # Build dense input and run torch dense conv
        dense_input = x.dense()  # [B, C, D, H, W]
        # Manually compute dense conv for verification
        weight_3d = conv.weight.data  # [ C_out, k, k, k, C_in]
        weight_dense = weight_3d.permute(0, 4, 1, 2, 3)  # [C_out, C_in, 3, 3, 3]

        dense_conv = torch.nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False).to(device)
        dense_conv.weight.data = weight_dense.contiguous()

        dense_out = dense_conv(dense_input)  # [B, C_out, D, H, W]

        # Compare at sparse output positions
        for idx in range(out_sparse.indices.shape[0]):
            b = int(out_sparse.indices[idx, 0])
            z = int(out_sparse.indices[idx, 1])
            y = int(out_sparse.indices[idx, 2])
            x_coord = int(out_sparse.indices[idx, 3])
            sparse_val = out_sparse.features[idx]
            dense_val = dense_out[b, :, z, y, x_coord]
            assert torch.allclose(sparse_val, dense_val, atol=1e-4), \
                f"Mismatch at point {idx}: sparse={sparse_val[:3]} dense={dense_val[:3]}"

    def test_conv_bias(self, device):
        """Bias should be added to every output point."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 8, 16
        spatial_shape = [6, 6, 6]
        num_points = 15

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device)
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)

        # Set bias to known value
        conv.bias.data.fill_(42.0)
        conv.weight.data.zero_()

        out = conv(x)
        # With zero weight, output should be just bias
        assert torch.allclose(out.features, torch.full_like(out.features, 42.0), atol=1e-5)


# ============================================================
# Backward / Gradient Tests
# ============================================================

class TestBackward:
    def test_subm_conv_backward(self, device):
        """Verify gradients flow through SubM conv."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 8, 16
        spatial_shape = [6, 6, 6]
        num_points = 20

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device)
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)
        x.features.requires_grad_(True)

        out = conv(x)
        loss = out.features.sum()
        loss.backward()

        assert x.features.grad is not None
        assert x.features.grad.shape == (num_points, in_ch)
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    def test_sparse_conv_backward(self, device):
        """Verify gradients for strided sparse conv."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 8, 16
        spatial_shape = [8, 8, 8]
        num_points = 15

        conv = spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False).to(device)
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)
        x.features.requires_grad_(True)

        out = conv(x)
        loss = out.features.sum()
        loss.backward()

        assert x.features.grad is not None
        assert conv.weight.grad is not None

    def test_sequential_backward(self, device):
        """Gradient through SparseSequential pipeline."""
        import spconv.pytorch as spconv

        model = spconv.SparseSequential(
            spconv.SubMConv3d(8, 16, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(16),
            spconv.SparseReLU(),
            spconv.SubMConv3d(16, 32, 3, padding=1, bias=False),
        ).to(device)
        model.train()

        x = _make_sparse_input(1, 20, 8, [6, 6, 6], device)
        x.features.requires_grad_(True)

        out = model(x)
        loss = out.features.sum()
        loss.backward()

        assert x.features.grad is not None


# ============================================================
# Fused FP32 Accuracy
# ============================================================


class TestFusedFP32Accuracy:
    @pytest.mark.parametrize("spatial_shape,num_points", [
        pytest.param([7, 7, 7], 40, id="regular-sparsity"),
        pytest.param([32, 32, 32], 64, id="high-sparsity"),
    ])
    @pytest.mark.parametrize("conv_cls,conv_kwargs", [
        ("SubMConv3d", {"padding": 1}),
        ("SparseConv3d", {"stride": 2, "padding": 1}),
    ])
    def test_fused_fp32_matches_dense_forward_and_backward(
            self, device, monkeypatch, spatial_shape, num_points, conv_cls, conv_kwargs):
        """Check fused FP32 against dense reference with strict allclose tolerances."""
        if device.type != "cuda":
            pytest.skip("fused HIP accuracy test requires a GPU")

        import spconv.pytorch as spconv
        from spconv.pytorch import ops

        hip_module = ops._get_hip_module()
        if hip_module is None or not hasattr(hip_module, "indice_conv_forward_fused"):
            pytest.skip("packaged HIP extension does not provide fused FP32 convolution")

        torch.manual_seed(17)
        np.random.seed(17)
        recorder = _HipCallRecorder(
            hip_module, {"indice_conv_forward_fused", "indice_conv_backward"})
        monkeypatch.setattr(ops, "_get_hip_module", lambda: recorder)
        monkeypatch.setattr(ops, "_USE_FUSED_FORWARD", True)
        monkeypatch.setattr(ops, "_ALLOW_INEXACT_FUSED_FP32", True)
        monkeypatch.setenv("SPCONV_USE_FUSED_DINPUT", "1")
        monkeypatch.setenv("SPCONV_USE_FUSED_DFILTERS", "1")

        in_channels, out_channels = 16, 32
        conv = getattr(spconv, conv_cls)(
            in_channels, out_channels, 3, bias=False, **conv_kwargs).to(device)
        sparse_input = _make_sparse_input(
            1, num_points, in_channels, spatial_shape, device)
        sparse_input.features.requires_grad_(True)

        reference_features = sparse_input.features.detach().requires_grad_(True)
        reference_weight = conv.weight.detach().requires_grad_(True)
        dense_input = _sparse_features_to_dense(
            reference_features, sparse_input.indices, spatial_shape, sparse_input.batch_size)
        dense_weight = reference_weight.permute(0, 4, 1, 2, 3).contiguous()
        reference_dense = F.conv3d(
            dense_input,
            dense_weight,
            stride=conv_kwargs.get("stride", 1),
            padding=conv_kwargs.get("padding", 0),
        )

        sparse_output = conv(sparse_input)
        reference_output = _gather_dense_features(reference_dense, sparse_output.indices)
        grad_output = torch.randn_like(sparse_output.features)
        sparse_output.features.backward(grad_output)
        (reference_output * grad_output).sum().backward()

        for name, actual, reference in (
                ("forward", sparse_output.features, reference_output),
                ("dinput", sparse_input.features.grad, reference_features.grad),
                ("dfilter", conv.weight.grad, reference_weight.grad)):
            _assert_fp32_strict(actual, reference, name)
        assert recorder.calls["indice_conv_forward_fused"] > 0
        assert recorder.calls["indice_conv_backward"] > 0


# ============================================================
# Fused Low-Precision Accuracy
# ============================================================

class TestFusedLowPrecisionAccuracy:
    @pytest.mark.parametrize("dtype", [
        torch.float16,
        torch.bfloat16,
    ])
    @pytest.mark.parametrize("conv_cls,conv_kwargs", [
        ("SubMConv3d", {"padding": 1}),
        ("SparseConv3d", {"stride": 2, "padding": 1}),
    ])
    def test_fused_low_precision_matches_dense_forward_and_backward(
            self, device, monkeypatch, dtype, conv_cls, conv_kwargs):
        """Compare public sparse-conv fused results against FP32 dense reference."""
        if device.type != "cuda":
            pytest.skip("fused HIP accuracy test requires a GPU")

        import spconv.pytorch as spconv
        from spconv.pytorch import ops

        hip_module = ops._get_hip_module()
        if hip_module is None:
            pytest.skip("packaged HIP extension is unavailable")

        fused_name = (
            "indice_conv_forward_fused_f16"
            if dtype == torch.float16
            else "indice_conv_forward_fused_bf16"
        )
        if not hasattr(hip_module, fused_name):
            pytest.skip(f"HIP extension does not provide {fused_name}")

        torch.manual_seed(17)
        np.random.seed(17)
        recorder = _HipCallRecorder(
            hip_module, {fused_name, "indice_conv_backward"})
        monkeypatch.setattr(ops, "_get_hip_module", lambda: recorder)
        monkeypatch.setattr(ops, "_USE_FUSED_FORWARD", True)
        monkeypatch.setenv("SPCONV_USE_FUSED_DINPUT", "1")
        monkeypatch.setenv("SPCONV_USE_FUSED_DFILTERS", "1")

        spatial_shape = [7, 7, 7]
        in_channels, out_channels, num_points = 16, 32, 40
        conv = getattr(spconv, conv_cls)(
            in_channels, out_channels, 3, bias=False, **conv_kwargs).to(device).to(dtype)
        sparse_input = _make_sparse_input(
            1, num_points, in_channels, spatial_shape, device, dtype=dtype)
        sparse_input.features.requires_grad_(True)

        reference_features = sparse_input.features.detach().float().requires_grad_(True)
        reference_weight = conv.weight.detach().float().requires_grad_(True)
        dense_input = _sparse_features_to_dense(
            reference_features, sparse_input.indices, spatial_shape, sparse_input.batch_size)
        dense_weight = reference_weight.permute(0, 4, 1, 2, 3).contiguous()
        reference_dense = F.conv3d(
            dense_input,
            dense_weight,
            stride=conv_kwargs.get("stride", 1),
            padding=conv_kwargs.get("padding", 0),
        )

        sparse_output = conv(sparse_input)
        reference_output = _gather_dense_features(reference_dense, sparse_output.indices)
        _assert_nv_low_precision_l2(
            sparse_output.features, reference_output,
            in_channels, out_channels, "forward")

        grad_output = torch.randn_like(sparse_output.features)
        sparse_output.features.backward(grad_output)
        (reference_output * grad_output.float()).sum().backward()

        _assert_nv_low_precision_l2(
            sparse_input.features.grad, reference_features.grad,
            in_channels, out_channels, "dinput")
        _assert_nv_low_precision_l2(
            conv.weight.grad, reference_weight.grad,
            in_channels, out_channels, "dfilter")
        assert recorder.calls[fused_name] > 0
        assert recorder.calls["indice_conv_backward"] > 0


# ============================================================
# Fused Unique-Index Accuracy
# ============================================================

class TestFusedUniqueAccuracy:
    @pytest.mark.parametrize("dtype,fused_name", [
        (torch.float32, "indice_conv_forward_fused_unique"),
        (torch.float16, "indice_conv_forward_fused_f16_unique"),
        (torch.bfloat16, "indice_conv_forward_fused_bf16_unique"),
    ])
    def test_fused_unique_matches_dense_forward_and_backward(
            self, device, monkeypatch, dtype, fused_name):
        """Trigger the unique-index fused binding and compare against dense reference."""
        if device.type != "cuda":
            pytest.skip("fused HIP accuracy test requires a GPU")

        import spconv.pytorch as spconv
        from spconv.pytorch import ops

        hip_module = ops._get_hip_module()
        if hip_module is None or not hasattr(hip_module, fused_name):
            pytest.skip(f"packaged HIP extension does not provide {fused_name}")

        torch.manual_seed(23)
        np.random.seed(23)
        recorder = _HipCallRecorder(hip_module, {fused_name, "indice_conv_backward"})
        monkeypatch.setattr(ops, "_get_hip_module", lambda: recorder)
        monkeypatch.setattr(ops, "_USE_FUSED_FORWARD", True)
        monkeypatch.setattr(ops, "_ALLOW_INEXACT_FUSED_FP32", True)
        monkeypatch.setenv("SPCONV_COALESCE_DUP_MIN_N", "4096")
        monkeypatch.setenv("SPCONV_USE_FUSED_DINPUT", "1")
        monkeypatch.setenv("SPCONV_USE_FUSED_DFILTERS", "1")

        spatial_shape = [64, 64, 64]
        in_channels, out_channels, num_points = 16, 32, 4096
        conv = spconv.SubMConv3d(
            in_channels, out_channels, 3, padding=1, bias=False).to(device).to(dtype)
        sparse_input = _make_sparse_input(
            1, num_points, in_channels, spatial_shape, device, dtype=dtype)
        sparse_input.features.requires_grad_(True)

        reference_features = sparse_input.features.detach().float().requires_grad_(True)
        reference_weight = conv.weight.detach().float().requires_grad_(True)
        dense_input = _sparse_features_to_dense(
            reference_features, sparse_input.indices, spatial_shape, sparse_input.batch_size)
        dense_weight = reference_weight.permute(0, 4, 1, 2, 3).contiguous()
        reference_dense = F.conv3d(dense_input, dense_weight, padding=1)

        sparse_output = conv(sparse_input)
        reference_output = _gather_dense_features(reference_dense, sparse_output.indices)
        grad_output = torch.randn_like(sparse_output.features)
        sparse_output.features.backward(grad_output)
        (reference_output * grad_output.float()).sum().backward()

        comparisons = (
            ("forward", sparse_output.features, reference_output),
            ("dinput", sparse_input.features.grad, reference_features.grad),
            ("dfilter", conv.weight.grad, reference_weight.grad),
        )
        if dtype == torch.float32:
            for name, actual, reference in comparisons:
                _assert_fp32_strict(actual, reference, name)
        else:
            for name, actual, reference in comparisons:
                _assert_nv_low_precision_l2(
                    actual, reference, in_channels, out_channels, name)

        assert recorder.calls[fused_name] > 0
        assert recorder.calls["indice_conv_backward"] > 0


# ============================================================
# Pooling Tests
# ============================================================

class TestPooling:
    def test_max_pool_forward(self, device):
        """SparseMaxPool3d reduces spatial dims."""
        import spconv.pytorch as spconv

        in_ch = 16
        spatial_shape = [10, 10, 10]
        num_points = 40

        pool = spconv.SparseMaxPool3d(2, stride=2).to(device)
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)
        out = pool(x)

        assert out.features.shape[1] == in_ch
        assert out.spatial_shape == [5, 5, 5]
        assert out.features.shape[0] > 0

    def test_global_max_pool(self, device):
        """Global pool reduces to [B, C]."""
        import spconv.pytorch as spconv
        from spconv.pytorch.pool import SparseGlobalMaxPool

        in_ch = 8
        batch_size = 2
        num_points = 15

        pool = SparseGlobalMaxPool()
        x = _make_sparse_input(batch_size, num_points, in_ch, [6, 6, 6], device)
        out = pool(x)

        assert out.shape == (batch_size, in_ch)

    def test_global_avg_pool(self, device):
        """Global avg pool reduces to [B, C]."""
        import spconv.pytorch as spconv
        from spconv.pytorch.pool import SparseGlobalAvgPool

        in_ch = 8
        batch_size = 2
        num_points = 15

        pool = SparseGlobalAvgPool()
        x = _make_sparse_input(batch_size, num_points, in_ch, [6, 6, 6], device)
        out = pool(x)

        assert out.shape == (batch_size, in_ch)


# ============================================================
# Multi-dim / Edge Cases
# ============================================================

class TestEdgeCases:
    def test_2d_conv(self, device):
        """SparseConv2d works."""
        import spconv.pytorch as spconv

        conv = spconv.SparseConv2d(8, 16, 3, stride=1, padding=1, bias=True).to(device)
        x = _make_sparse_input(1, 20, 8, [12, 12], device)
        out = conv(x)

        assert out.features.shape[1] == 16
        assert out.spatial_shape == [12, 12]

    def test_1d_conv(self, device):
        """SparseConv1d works."""
        import spconv.pytorch as spconv

        conv = spconv.SparseConv1d(4, 8, 3, stride=1, padding=1, bias=True).to(device)
        x = _make_sparse_input(1, 10, 4, [20], device)
        out = conv(x)

        assert out.features.shape[1] == 8

    def test_large_batch(self, device):
        """Multiple batches."""
        import spconv.pytorch as spconv

        conv = spconv.SubMConv3d(8, 16, 3, padding=1).to(device)
        x = _make_sparse_input(4, 30, 8, [8, 8, 8], device)
        out = conv(x)

        assert out.features.shape == (4 * 30, 16)
        assert out.batch_size == 4

    def test_single_point(self, device):
        """Single point input."""
        import spconv.pytorch as spconv

        conv = spconv.SubMConv3d(4, 8, 3, padding=1).to(device)
        x = _make_sparse_input(1, 1, 4, [5, 5, 5], device)
        out = conv(x)

        assert out.features.shape == (1, 8)

    def test_dilation(self, device):
        """Dilated sparse conv."""
        import spconv.pytorch as spconv

        conv = spconv.SubMConv3d(8, 16, 3, padding=2, dilation=2).to(device)
        x = _make_sparse_input(1, 20, 8, [10, 10, 10], device)
        out = conv(x)

        assert out.features.shape[1] == 16
        assert out.spatial_shape == [10, 10, 10]

    def test_indice_key_reuse(self, device):
        """Two convs sharing indice_key should reuse computed indices."""
        import spconv.pytorch as spconv

        in_ch, mid_ch, out_ch = 8, 16, 32
        spatial_shape = [8, 8, 8]

        conv1 = spconv.SubMConv3d(in_ch, mid_ch, 3, padding=1, indice_key="subm1").to(device)
        conv2 = spconv.SubMConv3d(mid_ch, out_ch, 3, padding=1, indice_key="subm1").to(device)

        x = _make_sparse_input(1, 20, in_ch, spatial_shape, device)
        mid = conv1(x)
        out = conv2(mid)

        assert out.features.shape == (20, out_ch)
        assert "subm1" in out.indice_dict


# ============================================================
# Transpose Conv
# ============================================================

class TestTransposeConv:
    def test_sparse_conv_transpose(self, device):
        """SparseConvTranspose3d upsamples spatial shape."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 8
        spatial_shape = [5, 5, 5]
        num_points = 15

        conv = spconv.SparseConvTranspose3d(
            in_ch, out_ch, 3, stride=2, padding=1, output_padding=1
        ).to(device)
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)
        out = conv(x)

        assert out.features.shape[1] == out_ch
        assert out.spatial_shape == [10, 10, 10]


# ============================================================
# Encoder-Decoder Pattern (typical in 3D detection)
# ============================================================

class TestEncoderDecoder:
    def test_mini_unet(self, device):
        """Encoder-decoder pattern typical in 3D object detection."""
        import spconv.pytorch as spconv

        spatial_shape = [16, 16, 16]
        in_ch = 4

        encoder = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, 16, 3, padding=1, bias=False, indice_key="subm0"),
            spconv.SparseBatchNorm(16),
            spconv.SparseReLU(),
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False, indice_key="down1"),
            spconv.SparseBatchNorm(32),
            spconv.SparseReLU(),
        ).to(device)

        x = _make_sparse_input(1, 50, in_ch, spatial_shape, device)
        out = encoder(x)

        assert out.spatial_shape == [8, 8, 8]
        assert out.features.shape[1] == 32
        assert out.features.shape[0] > 0


# ============================================================
# FlyDSL GEMM Path (fp16/bf16)
# ============================================================

class TestFlyDSLGemm:
    """Tests that exercise the FlyDSL GEMM path (fp16/bf16 dtypes)."""

    def test_subm_conv_fp16(self, device):
        """SubM conv in fp16 — triggers FlyDSL hgemm_splitk."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [8, 8, 8]
        num_points = 30

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device).half()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.float16)
        out = conv(x)

        assert out.features.dtype == torch.float16
        assert out.features.shape == (num_points, out_ch)
        assert not torch.isnan(out.features).any()

    def test_subm_conv_bf16(self, device):
        """SubM conv in bf16 — triggers FlyDSL hgemm_splitk."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [8, 8, 8]
        num_points = 30

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device).bfloat16()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.bfloat16)
        out = conv(x)

        assert out.features.dtype == torch.bfloat16
        assert out.features.shape == (num_points, out_ch)
        assert not torch.isnan(out.features).any()

    def test_sparse_conv_fp16_stride(self, device):
        """Strided sparse conv in fp16."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        num_points = 40

        conv = spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False).to(device).half()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.float16)
        out = conv(x)

        assert out.features.dtype == torch.float16
        assert out.spatial_shape == [5, 5, 5]
        assert not torch.isnan(out.features).any()

    def test_conv1x1_fp16(self, device):
        """1x1 conv fp16 — direct _gemm call."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 32, 64
        spatial_shape = [8, 8, 8]
        num_points = 20

        conv = spconv.SparseConv3d(in_ch, out_ch, 1, bias=False).to(device).half()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.float16)
        out = conv(x)

        assert out.features.shape == (num_points, out_ch)
        assert out.features.dtype == torch.float16

    def test_fp16_backward(self, device):
        """Verify gradients flow through fp16 sparse conv (FlyDSL path)."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [6, 6, 6]
        num_points = 20

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device).half()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.float16)
        x.features.requires_grad_(True)

        out = conv(x)
        loss = out.features.sum()
        loss.backward()

        assert x.features.grad is not None
        assert x.features.grad.dtype == torch.float16
        assert conv.weight.grad is not None

    def test_amp_training(self, device):
        """Mixed precision training with torch.cuda.amp.autocast."""
        import spconv.pytorch as spconv

        if device.type == 'cpu':
            pytest.skip("AMP requires CUDA/ROCm")

        in_ch, out_ch = 16, 32
        spatial_shape = [8, 8, 8]
        num_points = 25

        model = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(out_ch),
            spconv.SparseReLU(),
        ).to(device)

        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device)

        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x)

        assert out.features.shape == (num_points, out_ch)

    def test_sequential_fp16_inference(self, device):
        """Full pipeline in fp16 inference mode."""
        import spconv.pytorch as spconv

        model = spconv.SparseSequential(
            spconv.SubMConv3d(8, 16, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(16),
            spconv.SparseReLU(),
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            spconv.SparseBatchNorm(32),
            spconv.SparseReLU(),
        ).to(device).half()
        model.eval()

        x = _make_sparse_input(1, 40, 8, [12, 12, 12], device, dtype=torch.float16)

        with torch.no_grad():
            out = model(x)

        assert out.features.dtype == torch.float16
        assert out.spatial_shape == [6, 6, 6]
        assert out.features.shape[1] == 32
        assert not torch.isnan(out.features).any()

    def test_sparse_conv_bf16_stride(self, device):
        """Strided sparse conv in bf16 — exercises fused bf16 kernel."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [10, 10, 10]
        num_points = 40

        conv = spconv.SparseConv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False).to(device).bfloat16()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.bfloat16)
        out = conv(x)

        assert out.features.dtype == torch.bfloat16
        assert out.spatial_shape == [5, 5, 5]
        assert not torch.isnan(out.features).any()

    def test_bf16_backward(self, device):
        """Verify gradients flow through bf16 sparse conv (fused bf16 kernel)."""
        import spconv.pytorch as spconv

        in_ch, out_ch = 16, 32
        spatial_shape = [6, 6, 6]
        num_points = 20

        conv = spconv.SubMConv3d(in_ch, out_ch, 3, padding=1, bias=True).to(device).bfloat16()
        x = _make_sparse_input(1, num_points, in_ch, spatial_shape, device, dtype=torch.bfloat16)
        x.features.requires_grad_(True)

        out = conv(x)
        loss = out.features.sum()
        loss.backward()

        assert x.features.grad is not None
        assert x.features.grad.dtype == torch.bfloat16
        assert conv.weight.grad is not None

    def test_sequential_bf16_inference(self, device):
        """Full pipeline in bf16 inference mode — exercises fused bf16 kernel."""
        import spconv.pytorch as spconv

        model = spconv.SparseSequential(
            spconv.SubMConv3d(8, 16, 3, padding=1, bias=False),
            spconv.SparseBatchNorm(16),
            spconv.SparseReLU(),
            spconv.SparseConv3d(16, 32, 3, stride=2, padding=1, bias=False),
            spconv.SparseBatchNorm(32),
            spconv.SparseReLU(),
        ).to(device).bfloat16()
        model.eval()

        x = _make_sparse_input(1, 40, 8, [12, 12, 12], device, dtype=torch.bfloat16)

        with torch.no_grad():
            out = model(x)

        assert out.features.dtype == torch.bfloat16
        assert out.spatial_shape == [6, 6, 6]
        assert out.features.shape[1] == 32
        assert not torch.isnan(out.features).any()
