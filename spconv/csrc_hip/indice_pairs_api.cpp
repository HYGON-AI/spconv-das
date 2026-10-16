// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Modified by Hygon Information Technology Co., Ltd., 2026.

#include <torch/extension.h>
#include <cstdlib>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <vector>
#include <algorithm>
#include <string>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/hip/HIPBlas.h>

namespace spconv_hip {

template <typename K>
void generate_subm_conv_inds(
    const int* indices_in, int* indice_pairs, int* indice_num_per_loc,
    int num_act_in, int batch_size,
    const int* input_dims_h, const int* ksize_h, const int* dilation_h,
    int ndim, hipStream_t stream);

template <typename K>
int generate_conv_inds(
    const int* indices_in, int* indice_pairs, int* indice_num_per_loc,
    int* out_indices, int num_act_in, int batch_size,
    const int* input_dims_h, const int* output_dims_h,
    const int* ksize_h, const int* stride_h, const int* padding_h,
    const int* dilation_h, int ndim, bool transposed, hipStream_t stream);

template <typename IndexT>
void build_indice_conv_lut(
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    int* lut,
    int* dup_slots,
    int* dup_inp,
    int* dup_count_dev,
    int dup_capacity,
    int num_out,
    int kv,
    int nmax,
    int gather_axis,
    int scatter_axis,
    bool skip_center,
    hipStream_t stream);

template <typename IndexT>
void build_indice_conv_lut_unique(
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    int* lut,
    int num_out,
    int kv,
    int nmax,
    int gather_axis,
    int scatter_axis,
    bool skip_center,
    hipStream_t stream);

template <typename IndexT>
void build_indice_conv_lut_epoch(
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    std::int64_t* lut_state,
    int* dup_slots,
    int* dup_inp,
    int* dup_count_dev,
    int dup_capacity,
    int kv,
    int nmax,
    int gather_axis,
    int scatter_axis,
    bool skip_center,
    unsigned int lut_epoch,
    hipStream_t stream);

void launch_indice_conv_forward_fused_f32(
    const float* features,
    const float* filters,
    const int* lut,
    const std::int64_t* lut_state,
    float* out,
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed,
    hipStream_t stream);

void launch_indice_conv_accumulate_duplicates_f32(
    const float* features,
    const float* filters,
    const int* dup_slots,
    const int* dup_inp,
    const int* dup_count_dev,
    int dup_capacity,
    float* out,
    int kv,
    int c_in,
    int c_out,
    bool filters_transposed,
    hipStream_t stream);

#ifdef SPCONV_WITH_HCU_FUSED
void launch_indice_conv_accumulate_duplicates_f16(
    const __half* features,
    const __half* filters,
    const int* dup_slots,
    const int* dup_inp,
    const int* dup_count_dev,
    int dup_capacity,
    __half* out,
    int kv,
    int c_in,
    int c_out,
    bool filters_transposed,
    hipStream_t stream);

void launch_indice_conv_accumulate_duplicates_bf16(
    const __hip_bfloat16* features,
    const __hip_bfloat16* filters,
    const int* dup_slots,
    const int* dup_inp,
    const int* dup_count_dev,
    int dup_capacity,
    __hip_bfloat16* out,
    int kv,
    int c_in,
    int c_out,
    bool filters_transposed,
    hipStream_t stream);


void launch_indice_conv_forward_fused_f16(
    const __half* features,
    const __half* filters,
    const int* lut,
    const std::int64_t* lut_state,
    __half* out,
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed,
    hipStream_t stream);

void launch_indice_conv_forward_fused_bf16(
    const __hip_bfloat16* features,
    const __hip_bfloat16* filters,
    const int* lut,
    const std::int64_t* lut_state,
    __hip_bfloat16* out,
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed,
    hipStream_t stream);

template <typename IndexT>
void launch_indice_conv_dfilters_f16(
    const __half* features,
    const __half* out_bp,
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    __half* dfilters,
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis,
    float* split_partials,
    bool deterministic_split,
    hipStream_t stream);

template <typename IndexT>
void launch_indice_conv_dfilters_bf16(
    const __hip_bfloat16* features,
    const __hip_bfloat16* out_bp,
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    __hip_bfloat16* dfilters,
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis,
    float* split_partials,
    bool deterministic_split,
    hipStream_t stream);

template <typename IndexT>
void launch_indice_conv_dfilters_f32(
    const float* features,
    const float* out_bp,
    const IndexT* indice_pairs,
    const int* indice_num_per_loc,
    float* dfilters,
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis,
    float* split_partials,
    bool deterministic_split,
    hipStream_t stream);

#endif  // SPCONV_WITH_HCU_FUSED

template <typename IndexT, typename FlatIndexT>
void launch_flatten_indice_pairs_rows(
    const IndexT* indice_pairs,
    const int* window_ks,
    const int* seg_offsets,
    int num_segs,
    int nmax,
    int gather_axis,
    int scatter_axis,
    FlatIndexT* flat_inp,
    FlatIndexT* flat_out,
    int total_nhot,
    hipStream_t stream);

template <typename IndexT>
void launch_gather_rows_f32(
    const float* features,
    const IndexT* indices,
    float* gathered,
    int rows,
    int cols,
    hipStream_t stream);


template <typename IndexT>
void launch_indice_conv_scatter_add_f16(
    const __half* src,
    const IndexT* indices,
    __half* dst,
    int rows,
    int cols,
    hipStream_t stream);


}  // namespace spconv_hip

namespace {

struct IndiceConvWorkspace {
    torch::Tensor fused_lut;
    torch::Tensor fused_lut_state;
    torch::Tensor fused_dup_slots;
    torch::Tensor fused_dup_inp;
    torch::Tensor fused_dup_count;

    torch::Tensor grouped_feat_pack;
    torch::Tensor grouped_out_pack;
    torch::Tensor grouped_out_bp_pack;
    torch::Tensor grouped_flat_inp;
    torch::Tensor grouped_flat_out;
    torch::Tensor grouped_ks_dev;
    torch::Tensor grouped_offsets_dev;
    torch::Tensor grouped_ks_host;
    torch::Tensor grouped_offsets_host;
    torch::Tensor host_indice_pair_num;
    torch::Tensor dinput_filters_t;
#ifdef SPCONV_WITH_HCU_FUSED
    torch::Tensor dfilters_split_partials;
#endif
    unsigned int fused_lut_epoch = 1;
};

thread_local std::vector<IndiceConvWorkspace> g_workspaces;

IndiceConvWorkspace& get_workspace(const c10::Device& device) {
    TORCH_CHECK(device.is_cuda(), "workspace device must be CUDA/HIP");
    const int index = device.index();
    TORCH_CHECK(index >= 0, "workspace device index must be non-negative");
    if (static_cast<int>(g_workspaces.size()) <= index) {
        g_workspaces.resize(index + 1);
    }
    return g_workspaces[index];
}

bool tensor_matches(const torch::Tensor& tensor,
                    const c10::Device& device,
                    c10::ScalarType dtype) {
    return tensor.defined() &&
           tensor.device() == device &&
           tensor.scalar_type() == dtype &&
           tensor.is_contiguous();
}

bool host_tensor_matches(const torch::Tensor& tensor,
                         c10::ScalarType dtype) {
    return tensor.defined() &&
           tensor.device().is_cpu() &&
           tensor.scalar_type() == dtype &&
           tensor.is_contiguous() &&
           tensor.is_pinned();
}

torch::Tensor ensure_buffer(torch::Tensor& cache,
                            int64_t numel,
                            const c10::Device& device,
                            c10::ScalarType dtype,
                            const char* name) {
    TORCH_CHECK(numel >= 0, name, " numel must be non-negative");
    if (numel == 0) {
        return torch::empty({0}, torch::TensorOptions().device(device).dtype(dtype));
    }
    if (!tensor_matches(cache, device, dtype) || cache.numel() < numel) {
        cache = torch::empty({numel}, torch::TensorOptions().device(device).dtype(dtype));
    }
    return cache.narrow(0, 0, numel);
}

torch::Tensor ensure_host_buffer(torch::Tensor& cache,
                                 int64_t numel,
                                 c10::ScalarType dtype,
                                 const char* name) {
    TORCH_CHECK(numel >= 0, name, " numel must be non-negative");
    if (numel == 0) {
        return torch::empty({0}, torch::TensorOptions().device(torch::kCPU).dtype(dtype));
    }
    if (!host_tensor_matches(cache, dtype) || cache.numel() < numel) {
        cache = torch::empty(
            {numel},
            torch::TensorOptions().device(torch::kCPU).dtype(dtype).pinned_memory(true));
    }
    return cache.narrow(0, 0, numel);
}

torch::Tensor ensure_matrix(torch::Tensor& cache,
                            int64_t rows,
                            int64_t cols,
                            const c10::Device& device,
                            c10::ScalarType dtype,
                            const char* name) {
    TORCH_CHECK(rows >= 0 && cols >= 0, name, " shape must be non-negative");
    auto flat = ensure_buffer(cache, rows * cols, device, dtype, name);
    return flat.view({rows, cols});
}


}  // namespace

// Check whether the spatial volume fits in int32.
static bool check_use_int32(const std::vector<int>& spatial_shape, int batch_size) {
    int64_t vol = (int64_t)batch_size;
    for (int d : spatial_shape) vol *= d;
    return vol < (int64_t)std::numeric_limits<int32_t>::max();
}

// SubM indice pairs: input == output points.
// Returns: (indice_pairs [kv, 2, N], indice_pair_num [kv])
std::vector<torch::Tensor> get_indice_pairs_subm(
    torch::Tensor indices,       // [N, ndim+1] int32 on GPU
    int batch_size,
    std::vector<int> spatial_shape,
    std::vector<int> ksize,
    std::vector<int> dilation)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be on GPU");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");

    int ndim = (int)spatial_shape.size();
    int N = indices.size(0);
    int kv = 1;
    for (int k : ksize) kv *= k;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(indices.device());
    auto indice_pairs = torch::full({kv, 2, N}, -1, options);
    auto indice_pair_num = torch::zeros({kv}, options);

    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();

    bool use_i32 = check_use_int32(spatial_shape, batch_size);
    if (use_i32) {
        spconv_hip::generate_subm_conv_inds<int32_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), ksize.data(), dilation.data(),
            ndim, stream);
    } else {
        spconv_hip::generate_subm_conv_inds<int64_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), ksize.data(), dilation.data(),
            ndim, stream);
    }

    return {indice_pairs, indice_pair_num};
}

// Regular/transposed conv indice pairs.
// Returns: (out_indices [N_out, ndim+1], indice_pairs [kv, 2, N], indice_pair_num [kv])
std::vector<torch::Tensor> get_indice_pairs_conv(
    torch::Tensor indices,       // [N, ndim+1] int32 on GPU
    int batch_size,
    std::vector<int> spatial_shape,   // input spatial shape
    std::vector<int> output_shape,    // output spatial shape
    std::vector<int> ksize,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    bool transposed)
{
    TORCH_CHECK(indices.is_cuda(), "indices must be on GPU");
    TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");

    int ndim = (int)spatial_shape.size();
    int N = indices.size(0);
    int kv = 1;
    for (int k : ksize) kv *= k;

    // Upper bound on output points: N * kv (before dedup).
    int max_out = N * kv;

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(indices.device());
    auto indice_pairs = torch::full({kv, 2, N}, -1, options);
    auto indice_pair_num = torch::zeros({kv}, options);
    auto out_indices = torch::zeros({max_out, ndim + 1}, options);

    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();

    int num_out_act;
    bool use_i32 = check_use_int32(output_shape, batch_size);
    if (use_i32) {
        num_out_act = spconv_hip::generate_conv_inds<int32_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), output_shape.data(),
            ksize.data(), stride.data(), padding.data(), dilation.data(),
            ndim, transposed, stream);
    } else {
        num_out_act = spconv_hip::generate_conv_inds<int64_t>(
            indices.data_ptr<int>(),
            indice_pairs.data_ptr<int>(),
            indice_pair_num.data_ptr<int>(),
            out_indices.data_ptr<int>(),
            N, batch_size,
            spatial_shape.data(), output_shape.data(),
            ksize.data(), stride.data(), padding.data(), dilation.data(),
            ndim, transposed, stream);
    }

    // Trim output indices to actual count.
    out_indices = out_indices.slice(0, 0, num_out_act).contiguous();

    return {out_indices, indice_pairs, indice_pair_num};
}

// ---------------------------------------------------------------------------
// indice_conv forward: C++ for-loop to eliminate Python→C++ overhead per iter
// ---------------------------------------------------------------------------
torch::Tensor indice_conv_forward(
    torch::Tensor features,         // [N_in, C_in]
    torch::Tensor filters,          // [kv, C_in, C_out]
    torch::Tensor indice_pairs,     // [kv, 2, N_max]
    torch::Tensor indice_pair_num,  // [kv]
    int64_t num_activate_out,
    bool inverse,
    bool subm,
    bool filters_transposed = false)
{
    int kv = filters.size(0);
    int c_in = features.size(1);
    int c_out = filters_transposed ? filters.size(1) : filters.size(2);
    TORCH_CHECK(c_in == (filters_transposed ? filters.size(2) : filters.size(1)),
                "filter channel shape mismatch");
    int gather_axis = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;
    int center = kv / 2;
    bool pair_is_long = indice_pairs.scalar_type() == torch::kLong;
    bool track_grad = features.requires_grad() || filters.requires_grad();

    bool has_center_accum = subm && !inverse;
    auto out_features = has_center_accum
        ? torch::empty({num_activate_out, c_out}, features.options())
        : torch::zeros({num_activate_out, c_out}, features.options());
    if (has_center_accum) {
        auto center_filter = filters.select(0, center);
        if (filters_transposed) {
            center_filter = center_filter.transpose(0, 1);
        }
        if (track_grad) {
            out_features.copy_(at::mm(features, center_filter));
        } else {
            at::mm_out(out_features, features, center_filter);
        }
    }

    // 1 sync instead of kv syncs: copy pair counts to CPU
    auto& workspace = get_workspace(features.device());
    auto pn_host = ensure_host_buffer(
        workspace.host_indice_pair_num, kv, torch::kInt32, "host_indice_pair_num");
    auto* pn = pn_host.data_ptr<int>();
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    hipError_t pn_status = hipMemcpyAsync(
        pn,
        indice_pair_num.data_ptr<int>(),
        kv * sizeof(int),
        hipMemcpyDeviceToHost,
        stream);
    TORCH_CHECK(pn_status == hipSuccess,
                "hipMemcpyAsync indice_pair_num failed: ",
                hipGetErrorString(pn_status));
    hipError_t pn_sync_status = hipStreamSynchronize(stream);
    TORCH_CHECK(pn_sync_status == hipSuccess,
                "hipStreamSynchronize indice_pair_num failed: ",
                hipGetErrorString(pn_sync_status));
    int max_nhot = 0;
    for (int i = 0; i < kv; ++i) {
        if (pn[i] > max_nhot) max_nhot = pn[i];
    }
    auto feat_scratch = (!track_grad && max_nhot > 0)
        ? torch::empty({max_nhot, c_in}, features.options())
        : torch::Tensor();
    auto gemm_scratch = (!track_grad && max_nhot > 0)
        ? torch::empty({max_nhot, c_out}, out_features.options())
        : torch::Tensor();

    for (int i = 0; i < kv; ++i) {
        int nhot = pn[i];
        if (nhot == 0) continue;

        if (has_center_accum && i == center) {
            continue;
        }

        auto pair = indice_pairs.select(0, i);
        auto inp_inds = pair.select(0, gather_axis).slice(0, 0, nhot);
        auto out_inds = pair.select(0, scatter_axis).slice(0, 0, nhot);

        torch::Tensor inp_gathered;
        if (track_grad) {
            inp_gathered = features.index_select(0, inp_inds);  // [nhot, C_in]
        } else {
            inp_gathered = feat_scratch.slice(0, 0, nhot);
            at::index_select_out(inp_gathered, features, 0, inp_inds);  // [nhot, C_in]
        }
        auto w = filters.select(0, i);                           // [C_in, C_out]
        if (filters_transposed) {
            w = w.transpose(0, 1);
        }
        torch::Tensor result;
        if (track_grad) {
            result = at::mm(inp_gathered, w);                    // [nhot, C_out]
        } else {
            result = gemm_scratch.slice(0, 0, nhot);
            at::mm_out(result, inp_gathered, w);                 // [nhot, C_out]
        }

        if (result.dtype() != out_features.dtype()) {
            result = result.to(out_features.dtype());
        }
        out_features.index_add_(0, out_inds, result);
    }

    return out_features;
}

// Dinput for low-channel SubM FP16 is sensitive to the order in which the
// per-offset scatter-adds reach the same input row.  Keep every offset's GEMM
// result separate, then add offsets in their fixed kernel order.  Within one
// SubM offset an input row appears at most once, so index_add_ only materializes
// that offset's partial and does not introduce a reduction-order dependency.
static torch::Tensor indice_conv_dinput_fixed_order_f16(
    torch::Tensor out_bp,
    torch::Tensor filters,
    torch::Tensor indice_pairs,
    torch::Tensor indice_pair_num,
    int64_t num_activate_out,
    bool inverse)
{
    int kv = filters.size(0);
    int c_in = out_bp.size(1);
    int c_out = filters.size(1);
    int nmax = indice_pairs.size(2);
    int gather_axis = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;

    auto& workspace = get_workspace(out_bp.device());
    auto pn_host = ensure_host_buffer(
        workspace.host_indice_pair_num, kv, torch::kInt32, "host_indice_pair_num");
    auto* pn = pn_host.data_ptr<int>();
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    hipError_t pn_status = hipMemcpyAsync(
        pn, indice_pair_num.data_ptr<int>(), kv * sizeof(int),
        hipMemcpyDeviceToHost, stream);
    TORCH_CHECK(pn_status == hipSuccess,
                "hipMemcpyAsync indice_pair_num failed: ",
                hipGetErrorString(pn_status));
    hipError_t pn_sync_status = hipStreamSynchronize(stream);
    TORCH_CHECK(pn_sync_status == hipSuccess,
                "hipStreamSynchronize indice_pair_num failed: ",
                hipGetErrorString(pn_sync_status));

    std::vector<int> active_ks;
    int max_nhot = 0;
    for (int k = 0; k < kv; ++k) {
        max_nhot = std::max(max_nhot, pn[k]);
    }
    // Match the native SubM path: it writes the center GEMM directly, then
    // accumulates the remaining kernel offsets in increasing order.
    int center = kv / 2;
    for (int k = 0; k < kv; ++k) {
        if (k != center && pn[k] > 0) {
            active_ks.push_back(k);
        }
    }

    auto din = torch::empty({num_activate_out, c_out}, out_bp.options());
    at::mm_out(din, out_bp, filters.select(0, center).transpose(0, 1));
    if (active_ks.empty()) return din;

    auto gathered = torch::empty({max_nhot, c_in}, out_bp.options());
    auto gemm = torch::empty({max_nhot, c_out}, out_bp.options());

    for (int k : active_ks) {
        int nhot = pn[k];
        auto pair = indice_pairs.select(0, k);
        auto inp_inds = pair.select(0, gather_axis).slice(0, 0, nhot);
        auto out_inds = pair.select(0, scatter_axis).slice(0, 0, nhot);
        auto gathered_view = gathered.slice(0, 0, nhot);
        auto gemm_view = gemm.slice(0, 0, nhot);

        at::index_select_out(gathered_view, out_bp, 0, inp_inds);
        at::mm_out(gemm_view, gathered_view, filters.select(0, k).transpose(0, 1));
        if (indice_pairs.scalar_type() == torch::kLong) {
            spconv_hip::launch_indice_conv_scatter_add_f16<int64_t>(
                reinterpret_cast<const __half*>(gemm_view.data_ptr<at::Half>()),
                out_inds.data_ptr<int64_t>(), reinterpret_cast<__half*>(din.data_ptr<at::Half>()),
                nhot, c_out, stream);
        } else {
            spconv_hip::launch_indice_conv_scatter_add_f16<int>(
                reinterpret_cast<const __half*>(gemm_view.data_ptr<at::Half>()),
                out_inds.data_ptr<int>(), reinterpret_cast<__half*>(din.data_ptr<at::Half>()),
                nhot, c_out, stream);
        }
    }
    return din;
}

torch::Tensor indice_conv_forward_fused(
    torch::Tensor features,         // [N_in, C_in]
    torch::Tensor filters,          // [kv, C_in, C_out]
    torch::Tensor indice_pairs,     // [kv, 2, N_max]
    torch::Tensor indice_pair_num,  // [kv]
    int64_t num_activate_out,
    bool inverse,
    bool subm,
    bool filters_transposed = false,
    bool assume_unique_indices_arg = false)
{
    TORCH_CHECK(features.is_cuda(), "features must be on GPU");
    TORCH_CHECK(filters.is_cuda(), "filters must be on GPU");
    TORCH_CHECK(indice_pairs.is_cuda(), "indice_pairs must be on GPU");
    TORCH_CHECK(indice_pair_num.is_cuda(), "indice_pair_num must be on GPU");
    TORCH_CHECK(features.scalar_type() == torch::kFloat32,
                "fused indice_conv forward currently supports float32 only");
    TORCH_CHECK(filters.scalar_type() == torch::kFloat32,
                "fused indice_conv forward currently supports float32 filters only");
    TORCH_CHECK(features.is_contiguous(), "features must be contiguous");
    TORCH_CHECK(filters.is_contiguous(), "filters must be contiguous");
    TORCH_CHECK(indice_pairs.is_contiguous(), "indice_pairs must be contiguous");
    TORCH_CHECK(indice_pair_num.scalar_type() == torch::kInt32,
                "indice_pair_num must be int32");

    int kv = filters.size(0);
    int c_in = features.size(1);
    int c_out = filters_transposed ? filters.size(1) : filters.size(2);
    TORCH_CHECK(c_in == (filters_transposed ? filters.size(2) : filters.size(1)),
                "filter channel shape mismatch");
    int nmax = indice_pairs.size(2);
    int gather_axis = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;
    bool skip_center = subm && !inverse;
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    auto& workspace = get_workspace(features.device());

    bool assume_unique_indices = assume_unique_indices_arg;
    if (const char* env = std::getenv("SPCONV_ASSUME_UNIQUE_INDICES")) {
        assume_unique_indices = assume_unique_indices || std::atoi(env) != 0;
    }
    bool use_epoch_lut = false;
    if (!assume_unique_indices) {
        if (const char* env = std::getenv("SPCONV_USE_EPOCH_LUT")) {
            use_epoch_lut = std::atoi(env) != 0;
        }
    }
    torch::Tensor lut;
    torch::Tensor lut_state;
    unsigned int lut_epoch = 0;
    if (use_epoch_lut) {
        lut_state = ensure_matrix(
            workspace.fused_lut_state, num_activate_out, kv,
            features.device(), torch::kInt64, "fused_lut_state");
        lut_epoch = workspace.fused_lut_epoch++;
        if (workspace.fused_lut_epoch == 0) {
            workspace.fused_lut_epoch = 1;
        }
    } else {
        lut = ensure_matrix(
            workspace.fused_lut, num_activate_out, kv,
            features.device(), torch::kInt32, "fused_lut");
    }
    int64_t dup_capacity = static_cast<int64_t>(kv) * nmax;
    auto dup_slots = ensure_buffer(
        workspace.fused_dup_slots, dup_capacity,
        features.device(), torch::kInt32, "fused_dup_slots");
    auto dup_inp = ensure_buffer(
        workspace.fused_dup_inp, dup_capacity,
        features.device(), torch::kInt32, "fused_dup_inp");
    auto dup_count_dev = ensure_buffer(
        workspace.fused_dup_count, 1,
        features.device(), torch::kInt32, "fused_dup_count");

    if (indice_pairs.scalar_type() == torch::kLong) {
        if (use_epoch_lut) {
            spconv_hip::build_indice_conv_lut_epoch<int64_t>(
                indice_pairs.data_ptr<int64_t>(),
                indice_pair_num.data_ptr<int>(),
                lut_state.data_ptr<std::int64_t>(),
                dup_slots.data_ptr<int>(),
                dup_inp.data_ptr<int>(),
                dup_count_dev.data_ptr<int>(),
                static_cast<int>(dup_capacity),
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                lut_epoch,
                stream);
        } else if (assume_unique_indices) {
            spconv_hip::build_indice_conv_lut_unique<int64_t>(
                indice_pairs.data_ptr<int64_t>(),
                indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                num_activate_out,
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                stream);
        } else {
            spconv_hip::build_indice_conv_lut<int64_t>(
                indice_pairs.data_ptr<int64_t>(),
                indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                dup_slots.data_ptr<int>(),
                dup_inp.data_ptr<int>(),
                dup_count_dev.data_ptr<int>(),
                static_cast<int>(dup_capacity),
                num_activate_out,
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                stream);
        }
    } else if (indice_pairs.scalar_type() == torch::kInt32) {
        if (use_epoch_lut) {
            spconv_hip::build_indice_conv_lut_epoch<int>(
                indice_pairs.data_ptr<int>(),
                indice_pair_num.data_ptr<int>(),
                lut_state.data_ptr<std::int64_t>(),
                dup_slots.data_ptr<int>(),
                dup_inp.data_ptr<int>(),
                dup_count_dev.data_ptr<int>(),
                static_cast<int>(dup_capacity),
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                lut_epoch,
                stream);
        } else if (assume_unique_indices) {
            spconv_hip::build_indice_conv_lut_unique<int>(
                indice_pairs.data_ptr<int>(),
                indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                num_activate_out,
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                stream);
        } else {
            spconv_hip::build_indice_conv_lut<int>(
                indice_pairs.data_ptr<int>(),
                indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                dup_slots.data_ptr<int>(),
                dup_inp.data_ptr<int>(),
                dup_count_dev.data_ptr<int>(),
                static_cast<int>(dup_capacity),
                num_activate_out,
                kv,
                nmax,
                gather_axis,
                scatter_axis,
                skip_center,
                stream);
        }
    } else {
        TORCH_CHECK(false, "indice_pairs must be int32 or int64");
    }

    auto out_features = torch::empty({num_activate_out, c_out}, features.options());
    if (skip_center) {
        int center = kv / 2;
        auto center_filter = filters.select(0, center);
        if (filters_transposed) {
            center_filter = center_filter.transpose(0, 1);
        }
        at::mm_out(out_features, features, center_filter);
    }

    spconv_hip::launch_indice_conv_forward_fused_f32(
        features.data_ptr<float>(),
        filters.data_ptr<float>(),
        use_epoch_lut ? nullptr : lut.data_ptr<int>(),
        use_epoch_lut ? lut_state.data_ptr<std::int64_t>() : nullptr,
        out_features.data_ptr<float>(),
        num_activate_out,
        kv,
        c_in,
        c_out,
        lut_epoch,
        skip_center,
        filters_transposed,
        stream);
    if (!assume_unique_indices) {
        spconv_hip::launch_indice_conv_accumulate_duplicates_f32(
            features.data_ptr<float>(),
            filters.data_ptr<float>(),
            dup_slots.data_ptr<int>(),
            dup_inp.data_ptr<int>(),
            dup_count_dev.data_ptr<int>(),
            static_cast<int>(dup_capacity),
            out_features.data_ptr<float>(),
            kv,
            c_in,
            c_out,
            filters_transposed,
            stream);
    }

    return out_features;
}

// ---------------------------------------------------------------------------

// fp16 fused forward — same LUT/workspace infrastructure as f32 path
#ifdef SPCONV_WITH_HCU_FUSED
torch::Tensor indice_conv_forward_fused_f16(
    torch::Tensor features,
    torch::Tensor filters,
    torch::Tensor indice_pairs,
    torch::Tensor indice_pair_num,
    int64_t num_activate_out,
    bool inverse,
    bool subm,
    bool filters_transposed = false,
    bool assume_unique_indices = false)
{
    TORCH_CHECK(features.scalar_type() == torch::kHalf, "fp16 fused requires half features");
    TORCH_CHECK(filters.scalar_type()  == torch::kHalf, "fp16 fused requires half filters");
    TORCH_CHECK(features.is_contiguous(), "features must be contiguous");
    TORCH_CHECK(filters.is_contiguous(),  "filters must be contiguous");
    TORCH_CHECK(indice_pairs.is_contiguous(), "indice_pairs must be contiguous");
    TORCH_CHECK(indice_pair_num.scalar_type() == torch::kInt32, "indice_pair_num must be int32");

    int kv    = filters.size(0);
    int c_in  = features.size(1);
    int c_out = filters_transposed ? filters.size(1) : filters.size(2);
    int nmax  = indice_pairs.size(2);
    int gather_axis  = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;
    bool skip_center = subm && !inverse;
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    auto& workspace = get_workspace(features.device());

    // build LUT (reuse f32 workspace)
    torch::Tensor lut = ensure_matrix(
        workspace.fused_lut, num_activate_out, kv,
        features.device(), torch::kInt32, "fused_lut");
    int64_t dup_capacity = static_cast<int64_t>(kv) * nmax;
    auto dup_slots = ensure_buffer(workspace.fused_dup_slots, dup_capacity,
                                   features.device(), torch::kInt32, "fused_dup_slots");
    auto dup_inp   = ensure_buffer(workspace.fused_dup_inp, dup_capacity,
                                   features.device(), torch::kInt32, "fused_dup_inp");
    auto dup_count = ensure_buffer(workspace.fused_dup_count, 1,
                                   features.device(), torch::kInt32, "fused_dup_count");

    auto build_lut = [&](auto* pairs_ptr) {
        if (assume_unique_indices) {
            spconv_hip::build_indice_conv_lut_unique(
                pairs_ptr, indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                num_activate_out, kv, nmax, gather_axis, scatter_axis, skip_center, stream);
        } else {
            spconv_hip::build_indice_conv_lut(
                pairs_ptr, indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                dup_slots.data_ptr<int>(), dup_inp.data_ptr<int>(), dup_count.data_ptr<int>(),
                static_cast<int>(dup_capacity), num_activate_out,
                kv, nmax, gather_axis, scatter_axis, skip_center, stream);
        }
    };
    if (indice_pairs.scalar_type() == torch::kLong)
        build_lut(indice_pairs.data_ptr<int64_t>());
    else
        build_lut(indice_pairs.data_ptr<int>());

    auto out_features = torch::empty({num_activate_out, c_out}, features.options());
    if (skip_center) {
        int center = kv / 2;
        auto cf = filters.select(0, center);
        if (filters_transposed) cf = cf.transpose(0, 1).contiguous();
        at::mm_out(out_features, features, cf);
    }

    spconv_hip::launch_indice_conv_forward_fused_f16(
        reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(filters.data_ptr<at::Half>()),
        lut.data_ptr<int>(), static_cast<const std::int64_t*>(nullptr),
        reinterpret_cast<__half*>(out_features.data_ptr<at::Half>()),
        static_cast<int>(num_activate_out), kv, c_in, c_out,
        0, skip_center, filters_transposed, stream);

    // accumulate duplicates 在 device 上完成，避免 D2H 同步
    if (!assume_unique_indices) {
        spconv_hip::launch_indice_conv_accumulate_duplicates_f16(
            reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(filters.data_ptr<at::Half>()),
            dup_slots.data_ptr<int>(), dup_inp.data_ptr<int>(),
            dup_count.data_ptr<int>(), static_cast<int>(dup_capacity),
            reinterpret_cast<__half*>(out_features.data_ptr<at::Half>()),
            kv, c_in, c_out, filters_transposed, stream);
    }

    return out_features;
}

// bf16 fused forward — same LUT/workspace infrastructure as fp16 path
torch::Tensor indice_conv_forward_fused_bf16(
    torch::Tensor features,
    torch::Tensor filters,
    torch::Tensor indice_pairs,
    torch::Tensor indice_pair_num,
    int64_t num_activate_out,
    bool inverse,
    bool subm,
    bool filters_transposed = false,
    bool assume_unique_indices = false)
{
    TORCH_CHECK(features.scalar_type() == torch::kBFloat16, "bf16 fused requires bfloat16 features");
    TORCH_CHECK(filters.scalar_type()  == torch::kBFloat16, "bf16 fused requires bfloat16 filters");
    TORCH_CHECK(features.is_contiguous(), "features must be contiguous");
    TORCH_CHECK(filters.is_contiguous(),  "filters must be contiguous");
    TORCH_CHECK(indice_pairs.is_contiguous(), "indice_pairs must be contiguous");
    TORCH_CHECK(indice_pair_num.scalar_type() == torch::kInt32, "indice_pair_num must be int32");

    int kv    = filters.size(0);
    int c_in  = features.size(1);
    int c_out = filters_transposed ? filters.size(1) : filters.size(2);
    int nmax  = indice_pairs.size(2);
    int gather_axis  = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;
    bool skip_center = subm && !inverse;
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    auto& workspace = get_workspace(features.device());

    // build LUT (reuse f32 workspace)
    torch::Tensor lut = ensure_matrix(
        workspace.fused_lut, num_activate_out, kv,
        features.device(), torch::kInt32, "fused_lut");
    int64_t dup_capacity = static_cast<int64_t>(kv) * nmax;
    auto dup_slots = ensure_buffer(workspace.fused_dup_slots, dup_capacity,
                                   features.device(), torch::kInt32, "fused_dup_slots");
    auto dup_inp   = ensure_buffer(workspace.fused_dup_inp, dup_capacity,
                                   features.device(), torch::kInt32, "fused_dup_inp");
    auto dup_count = ensure_buffer(workspace.fused_dup_count, 1,
                                   features.device(), torch::kInt32, "fused_dup_count");

    auto build_lut = [&](auto* pairs_ptr) {
        if (assume_unique_indices) {
            spconv_hip::build_indice_conv_lut_unique(
                pairs_ptr, indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                num_activate_out, kv, nmax, gather_axis, scatter_axis, skip_center, stream);
        } else {
            spconv_hip::build_indice_conv_lut(
                pairs_ptr, indice_pair_num.data_ptr<int>(),
                lut.data_ptr<int>(),
                dup_slots.data_ptr<int>(), dup_inp.data_ptr<int>(), dup_count.data_ptr<int>(),
                static_cast<int>(dup_capacity), num_activate_out,
                kv, nmax, gather_axis, scatter_axis, skip_center, stream);
        }
    };
    if (indice_pairs.scalar_type() == torch::kLong)
        build_lut(indice_pairs.data_ptr<int64_t>());
    else
        build_lut(indice_pairs.data_ptr<int>());

    auto out_features = torch::empty({num_activate_out, c_out}, features.options());
    if (skip_center) {
        int center = kv / 2;
        auto cf = filters.select(0, center);
        if (filters_transposed) cf = cf.transpose(0, 1).contiguous();
        at::mm_out(out_features, features, cf);
    }

    spconv_hip::launch_indice_conv_forward_fused_bf16(
        reinterpret_cast<const __hip_bfloat16*>(features.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __hip_bfloat16*>(filters.data_ptr<at::BFloat16>()),
        lut.data_ptr<int>(), static_cast<const std::int64_t*>(nullptr),
        reinterpret_cast<__hip_bfloat16*>(out_features.data_ptr<at::BFloat16>()),
        static_cast<int>(num_activate_out), kv, c_in, c_out,
        0, skip_center, filters_transposed, stream);

    // accumulate duplicates 在 device 上完成，避免 D2H 同步
    if (!assume_unique_indices) {
        spconv_hip::launch_indice_conv_accumulate_duplicates_bf16(
            reinterpret_cast<const __hip_bfloat16*>(features.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __hip_bfloat16*>(filters.data_ptr<at::BFloat16>()),
            dup_slots.data_ptr<int>(), dup_inp.data_ptr<int>(),
            dup_count.data_ptr<int>(), static_cast<int>(dup_capacity),
            reinterpret_cast<__hip_bfloat16*>(out_features.data_ptr<at::BFloat16>()),
            kv, c_in, c_out, filters_transposed, stream);
    }

    return out_features;
}


#endif  // SPCONV_WITH_HCU_FUSED

// indice_conv backward: C++ for-loop
// ---------------------------------------------------------------------------
std::vector<torch::Tensor> indice_conv_backward(
    torch::Tensor features,         // [N_in, C_in]
    torch::Tensor filters,          // [kv, C_in, C_out]
    torch::Tensor out_bp,           // [N_out, C_out]
    torch::Tensor indice_pairs,     // [kv, 2, N_max]
    torch::Tensor indice_pair_num,  // [kv]
    bool inverse,
    bool subm)
{
    int kv = filters.size(0);
    int c_in = features.size(1);
    int c_out = out_bp.size(1);
    int n_in = features.size(0);
    int nmax = indice_pairs.size(2);
    int gather_axis = inverse ? 1 : 0;
    int scatter_axis = inverse ? 0 : 1;
    int center = kv / 2;
    bool pair_is_long = indice_pairs.scalar_type() == torch::kLong;

    auto out_bp_contig = out_bp.contiguous();
    auto& workspace = get_workspace(features.device());
    const char* force_native_env = std::getenv("SPCONV_FORCE_NATIVE");
    const bool force_native =
        force_native_env != nullptr && std::atoi(force_native_env) != 0;
#ifdef SPCONV_WITH_HCU_FUSED
    bool use_fused_dinput =
        (out_bp_contig.scalar_type() == torch::kFloat32 &&
         filters.scalar_type() == torch::kFloat32) ||
        (out_bp_contig.scalar_type() == torch::kHalf &&
         filters.scalar_type() == torch::kHalf) ||
        (out_bp_contig.scalar_type() == torch::kBFloat16 &&
         filters.scalar_type() == torch::kBFloat16);
    if (const char* env = std::getenv("SPCONV_USE_FUSED_DINPUT")) {
        use_fused_dinput = std::atoi(env) != 0;
    }
    // Prefer the fused FP32 MMAC backward path. Set the environment variable
    // to 0 to request the strict Native GEMM numerical route instead.
    const char* inexact_fused_fp32_env = std::getenv("SPCONV_ALLOW_INEXACT_FUSED_FP32");
    bool allow_inexact_fused_fp32 =
        inexact_fused_fp32_env == nullptr || std::atoi(inexact_fused_fp32_env) != 0;
    if (out_bp_contig.scalar_type() == torch::kFloat32 &&
        filters.scalar_type() == torch::kFloat32 && !allow_inexact_fused_fp32) {
        use_fused_dinput = false;
    }
    if ((out_bp_contig.scalar_type() == torch::kHalf || out_bp_contig.scalar_type() == torch::kBFloat16) &&
        c_in < 8 && c_out <= 16) {
        use_fused_dinput = false;
    }
    bool cache_dinput_filters_t = true;
    if (const char* env = std::getenv("SPCONV_CACHE_DINPUT_FILTERS_T")) {
        cache_dinput_filters_t = std::atoi(env) != 0;
    }
    torch::Tensor din;
    auto filters_t_view = filters.transpose(1, 2);
    torch::Tensor filters_t;
    if (cache_dinput_filters_t) {
        auto filters_t_storage = ensure_buffer(
            workspace.dinput_filters_t,
            static_cast<int64_t>(kv) * c_out * c_in,
            filters.device(),
            filters.scalar_type(),
            "dinput_filters_t");
        filters_t = filters_t_storage.view({kv, c_out, c_in});
        filters_t.copy_(filters_t_view);
    } else {
        filters_t = filters_t_view.contiguous();
    }
    int fixed_order_min_nhot = 4096;
    if (const char* env = std::getenv("SPCONV_FIXED_ORDER_DINPUT_MIN_NHOT")) {
        fixed_order_min_nhot = std::max(1, std::atoi(env));
    }
    // Match the legacy workspace FP16 route for low-channel SubM dinput:
    // FP16 GEMM followed by one ordered half add per offset.
    bool use_workspace_order_dinput =
        out_bp_contig.scalar_type() == torch::kHalf &&
        subm && !inverse && c_in < 8 && c_out <= 16 &&
        nmax >= fixed_order_min_nhot;
    if (const char* env = std::getenv("SPCONV_WORKSPACE_ORDER_DINPUT")) {
        use_workspace_order_dinput = std::atoi(env) != 0;
    }
    if (force_native) {
        use_fused_dinput = false;
        use_workspace_order_dinput = false;
    }
    if (use_workspace_order_dinput) {
        din = indice_conv_dinput_fixed_order_f16(
            out_bp_contig, filters, indice_pairs, indice_pair_num, n_in, !inverse);
    } else if (use_fused_dinput && filters_t.scalar_type() == torch::kFloat32) {
        din = indice_conv_forward_fused(
            out_bp_contig, filters_t, indice_pairs, indice_pair_num,
            n_in, !inverse, subm);
    } else if (use_fused_dinput && filters_t.scalar_type() == torch::kHalf) {
        din = indice_conv_forward_fused_f16(
            out_bp_contig, filters_t, indice_pairs, indice_pair_num,
            n_in, !inverse, subm, false, false);
    } else if (use_fused_dinput && filters_t.scalar_type() == torch::kBFloat16) {
        din = indice_conv_forward_fused_bf16(
            out_bp_contig, filters_t, indice_pairs, indice_pair_num,
            n_in, !inverse, subm);
    } else {
        din = indice_conv_forward(
            out_bp_contig, filters_t, indice_pairs, indice_pair_num,
            n_in, !inverse, subm);
    }
    bool use_fused_dfilters =
        (features.scalar_type() == torch::kFloat32 &&
         out_bp_contig.scalar_type() == torch::kFloat32 &&
         filters.scalar_type() == torch::kFloat32 &&
         (indice_pairs.scalar_type() == torch::kLong ||
          indice_pairs.scalar_type() == torch::kInt32)) ||
        (features.scalar_type() == torch::kHalf &&
         out_bp_contig.scalar_type() == torch::kHalf &&
         filters.scalar_type() == torch::kHalf &&
         (indice_pairs.scalar_type() == torch::kLong ||
          indice_pairs.scalar_type() == torch::kInt32)) ||
        (features.scalar_type() == torch::kBFloat16 &&
         out_bp_contig.scalar_type() == torch::kBFloat16 &&
         filters.scalar_type() == torch::kBFloat16 &&
         (indice_pairs.scalar_type() == torch::kLong ||
          indice_pairs.scalar_type() == torch::kInt32));
    if (const char* env = std::getenv("SPCONV_USE_FUSED_DFILTERS")) {
        use_fused_dfilters = std::atoi(env) != 0;
    }
    if (features.scalar_type() == torch::kFloat32 &&
        out_bp_contig.scalar_type() == torch::kFloat32 &&
        filters.scalar_type() == torch::kFloat32 && !allow_inexact_fused_fp32) {
        use_fused_dfilters = false;
    }

    if ((features.scalar_type() == torch::kHalf || features.scalar_type() == torch::kBFloat16) &&
        c_in < 8) {
        use_fused_dfilters = false;
    }
    if (force_native) {
        use_fused_dfilters = false;
    }
    auto dfilters = use_fused_dfilters ? torch::empty_like(filters)
                                       : torch::zeros_like(filters);
    // Split reduction writes FP32 partials and reduces them in a fixed order,
    // avoiding the inaccurate 16-bit CAS accumulation used by the old path.
    bool deterministic_dfilters_split = true;
    if (const char* env = std::getenv("SPCONV_DFILTERS_DETERMINISTIC_SPLIT")) {
        deterministic_dfilters_split = std::atoi(env) != 0;
    }
    constexpr int64_t kMaxDfiltersSplitPartialBytes = 64LL * 1024 * 1024;
    const bool is_fp32 = features.scalar_type() == torch::kFloat32;
    const int dfilters_split_cap = is_fp32 ? 64 : 16;
    int dfilters_split_count = nmax >= 512 ? dfilters_split_cap : 1;
    if (const char* env = std::getenv("SPCONV_DFILTERS_SPLIT_NHOT")) {
        int parsed = std::atoi(env);
        dfilters_split_count = parsed > 1 ?
            (parsed > dfilters_split_cap ? dfilters_split_cap : parsed) : 1;
    } else if (is_fp32) {
        while (dfilters_split_count > 1 &&
               static_cast<int64_t>(dfilters_split_count) * kv * c_in * c_out *
                   static_cast<int64_t>(sizeof(float)) > kMaxDfiltersSplitPartialBytes) {
            dfilters_split_count >>= 1;
        }
    }
    torch::Tensor dfilters_split_partials;
    if (use_fused_dfilters && deterministic_dfilters_split && dfilters_split_count > 1) {
        dfilters_split_partials = ensure_buffer(
            workspace.dfilters_split_partials,
            static_cast<int64_t>(dfilters_split_count) * kv * c_in * c_out,
            features.device(), torch::kFloat32, "dfilters_split_partials");
    }
    if (use_fused_dfilters) {
        hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
        bool is_fp16 = features.scalar_type() == torch::kHalf;
        bool is_bf16 = features.scalar_type() == torch::kBFloat16;
        if (indice_pairs.scalar_type() == torch::kLong) {
            if (is_fp16) {
                spconv_hip::launch_indice_conv_dfilters_f16<int64_t>(
                    reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(out_bp_contig.data_ptr<at::Half>()),
                    indice_pairs.data_ptr<int64_t>(),
                    indice_pair_num.data_ptr<int>(),
                    reinterpret_cast<__half*>(dfilters.data_ptr<at::Half>()),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            } else if (is_bf16) {
                spconv_hip::launch_indice_conv_dfilters_bf16<int64_t>(
                    reinterpret_cast<const __hip_bfloat16*>(features.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __hip_bfloat16*>(out_bp_contig.data_ptr<at::BFloat16>()),
                    indice_pairs.data_ptr<int64_t>(),
                    indice_pair_num.data_ptr<int>(),
                    reinterpret_cast<__hip_bfloat16*>(dfilters.data_ptr<at::BFloat16>()),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            } else {
                spconv_hip::launch_indice_conv_dfilters_f32<int64_t>(
                    features.data_ptr<float>(),
                    out_bp_contig.data_ptr<float>(),
                    indice_pairs.data_ptr<int64_t>(),
                    indice_pair_num.data_ptr<int>(),
                    dfilters.data_ptr<float>(),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            }
        } else if (indice_pairs.scalar_type() == torch::kInt32) {
            if (is_fp16) {
                spconv_hip::launch_indice_conv_dfilters_f16<int32_t>(
                    reinterpret_cast<const __half*>(features.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(out_bp_contig.data_ptr<at::Half>()),
                    indice_pairs.data_ptr<int>(),
                    indice_pair_num.data_ptr<int>(),
                    reinterpret_cast<__half*>(dfilters.data_ptr<at::Half>()),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            } else if (is_bf16) {
                spconv_hip::launch_indice_conv_dfilters_bf16<int32_t>(
                    reinterpret_cast<const __hip_bfloat16*>(features.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __hip_bfloat16*>(out_bp_contig.data_ptr<at::BFloat16>()),
                    indice_pairs.data_ptr<int>(),
                    indice_pair_num.data_ptr<int>(),
                    reinterpret_cast<__hip_bfloat16*>(dfilters.data_ptr<at::BFloat16>()),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            } else {
                spconv_hip::launch_indice_conv_dfilters_f32<int>(
                    features.data_ptr<float>(),
                    out_bp_contig.data_ptr<float>(),
                    indice_pairs.data_ptr<int>(),
                    indice_pair_num.data_ptr<int>(),
                    dfilters.data_ptr<float>(),
                    kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                    dfilters_split_partials.defined() ? dfilters_split_partials.data_ptr<float>() : nullptr,
                    deterministic_dfilters_split, stream);
            }
        } else {
            use_fused_dfilters = false;
        }
    }
    if (use_fused_dfilters) {
        return {din, dfilters};
    }
#else
    auto filters_t = filters.transpose(1, 2).contiguous();
    auto din = indice_conv_forward(out_bp_contig, filters_t, indice_pairs,
                                   indice_pair_num, n_in, !inverse, subm);
    auto dfilters = torch::zeros_like(filters);
#endif  // SPCONV_WITH_HCU_FUSED

    auto pn_host = ensure_host_buffer(
        workspace.host_indice_pair_num, kv, torch::kInt32, "host_indice_pair_num");
    auto* pn = pn_host.data_ptr<int>();
    hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
    hipError_t pn_status = hipMemcpyAsync(
        pn,
        indice_pair_num.data_ptr<int>(),
        kv * sizeof(int),
        hipMemcpyDeviceToHost,
        stream);
    TORCH_CHECK(pn_status == hipSuccess,
                "hipMemcpyAsync indice_pair_num failed: ",
                hipGetErrorString(pn_status));
    hipError_t pn_sync_status = hipStreamSynchronize(stream);
    TORCH_CHECK(pn_sync_status == hipSuccess,
                "hipStreamSynchronize indice_pair_num failed: ",
                hipGetErrorString(pn_sync_status));
#ifdef SPCONV_WITH_HCU_FUSED
    if (features.scalar_type() == torch::kFloat32 &&
        out_bp_contig.scalar_type() == torch::kFloat32 &&
        filters.scalar_type() == torch::kFloat32) {
        if (subm && !inverse) {
            auto center_grad = dfilters.select(0, center);
            at::mm_out(center_grad, features.t(), out_bp_contig);
        }

        std::vector<int> active_ks;
        active_ks.reserve(kv);
        for (int i = 0; i < kv; ++i) {
            if (pn[i] == 0) {
                continue;
            }
            if (subm && !inverse && i == center) {
                continue;
            }
            active_ks.push_back(i);
        }
        std::sort(active_ks.begin(), active_ks.end(),
                  [&](int a, int b) { return pn[a] < pn[b]; });

        if (!active_ks.empty()) {
            hipblasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
            at::cuda::blas::PointerModeGuard pointer_mode_guard(handle, HIPBLAS_POINTER_MODE_HOST);
            auto& workspace = get_workspace(features.device());
            int64_t kWindowNhotCap = 1500000;
            if (const char* env = std::getenv("SPCONV_GROUPED_DFILTERS_NHOT_CAP")) {
                long long parsed = std::atoll(env);
                if (parsed > 0) {
                    kWindowNhotCap = parsed;
                }
            }
            size_t free_bytes = 0;
            size_t total_bytes = 0;
            if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
                constexpr size_t kReserveBytes = 256ull << 20;
                if (free_bytes > kReserveBytes) {
                    size_t bytes_per_nhot =
                        ((size_t)(c_in + c_out) * sizeof(float) + 2 * sizeof(int64_t)) * 5 / 4;
                    int64_t free_cap = static_cast<int64_t>((free_bytes - kReserveBytes) /
                                                            std::max<size_t>(bytes_per_nhot, 1));
                    if (free_cap > 0) {
                        kWindowNhotCap = std::min(kWindowNhotCap, free_cap);
                    }
                }
            }
            kWindowNhotCap = std::max<int64_t>(kWindowNhotCap, 200000);
            int64_t total_active_nhot = 0;
            for (int i : active_ks) {
                total_active_nhot += pn[i];
            }
            int64_t pack_cap = std::min<int64_t>(kWindowNhotCap, total_active_nhot);
            auto feat_pack_storage = ensure_buffer(
                workspace.grouped_feat_pack, pack_cap * c_in,
                features.device(), features.scalar_type(), "grouped_feat_pack");
            auto out_bp_pack_storage = ensure_buffer(
                workspace.grouped_out_bp_pack, pack_cap * c_out,
                out_bp_contig.device(), out_bp_contig.scalar_type(), "grouped_out_bp_pack");
            auto pair_index_dtype = pair_is_long ? torch::kInt64 : torch::kInt32;
            auto flat_inp_storage = ensure_buffer(
                workspace.grouped_flat_inp, pack_cap,
                features.device(), pair_index_dtype, "grouped_flat_inp");
            auto flat_out_storage = ensure_buffer(
                workspace.grouped_flat_out, pack_cap,
                features.device(), pair_index_dtype, "grouped_flat_out");
            auto ks_dev_storage = ensure_buffer(
                workspace.grouped_ks_dev, active_ks.size(),
                features.device(), torch::kInt32, "grouped_ks_dev");
            auto offsets_dev_storage = ensure_buffer(
                workspace.grouped_offsets_dev, active_ks.size() + 1,
                features.device(), torch::kInt32, "grouped_offsets_dev");
            size_t begin = 0;
            while (begin < active_ks.size()) {
                size_t end = begin;
                int64_t window_nhot = 0;
                while (end < active_ks.size()) {
                    int nhot = pn[active_ks[end]];
                    if (window_nhot > 0 && window_nhot + nhot > kWindowNhotCap) {
                        break;
                    }
                    window_nhot += nhot;
                    ++end;
                }

                auto feat_pack = feat_pack_storage.narrow(0, 0, window_nhot * c_in)
                    .view({window_nhot, c_in});
                auto out_bp_pack = out_bp_pack_storage.narrow(0, 0, window_nhot * c_out)
                    .view({window_nhot, c_out});
                std::vector<hipblasOperation_t> transa_array;
                std::vector<hipblasOperation_t> transb_array;
                std::vector<int> m_array;
                std::vector<int> n_array;
                std::vector<int> k_array;
                std::vector<float> alpha_array;
                std::vector<const float*> a_array;
                std::vector<int> lda_array;
                std::vector<const float*> b_array;
                std::vector<int> ldb_array;
                std::vector<float> beta_array;
                std::vector<float*> c_array;
                std::vector<int> ldc_array;
                std::vector<int> group_size;
                size_t max_groups = end - begin;
                size_t group_count = 0;
                transa_array.reserve(max_groups);
                transb_array.reserve(max_groups);
                m_array.reserve(max_groups);
                n_array.reserve(max_groups);
                k_array.reserve(max_groups);
                alpha_array.reserve(max_groups);
                a_array.reserve(max_groups);
                lda_array.reserve(max_groups);
                b_array.reserve(max_groups);
                ldb_array.reserve(max_groups);
                beta_array.reserve(max_groups);
                c_array.reserve(max_groups);
                ldc_array.reserve(max_groups);
                group_size.reserve(max_groups);
                std::vector<int> window_ks_host;
                std::vector<int> seg_offsets_host;
                window_ks_host.reserve(max_groups);
                seg_offsets_host.reserve(max_groups + 1);
                seg_offsets_host.push_back(0);

                int64_t offset = 0;
                int prev_nhot = -1;
                for (size_t idx = begin; idx < end; ++idx) {
                    int i = active_ks[idx];
                    int nhot = pn[i];
                    window_ks_host.push_back(i);
                    seg_offsets_host.push_back(static_cast<int>(offset + nhot));
                    auto dfilter_i = dfilters.select(0, i);
                    a_array.push_back(out_bp_pack.data_ptr<float>() + offset * c_out);
                    b_array.push_back(feat_pack.data_ptr<float>() + offset * c_in);
                    c_array.push_back(dfilter_i.data_ptr<float>());
                    if (nhot != prev_nhot) {
                        transa_array.push_back(HIPBLAS_OP_N);
                        transb_array.push_back(HIPBLAS_OP_T);
                        m_array.push_back(c_out);
                        n_array.push_back(c_in);
                        k_array.push_back(nhot);
                        alpha_array.push_back(1.0f);
                        lda_array.push_back(c_out);
                        ldb_array.push_back(c_in);
                        beta_array.push_back(0.0f);
                        ldc_array.push_back(c_out);
                        group_size.push_back(1);
                        prev_nhot = nhot;
                        ++group_count;
                    } else {
                        ++group_size.back();
                    }
                    offset += nhot;
                }

                auto ks_dev = ks_dev_storage.narrow(0, 0, window_ks_host.size());
                auto offsets_dev = offsets_dev_storage.narrow(0, 0, seg_offsets_host.size());
                auto ks_host = ensure_host_buffer(
                    workspace.grouped_ks_host, window_ks_host.size(), torch::kInt32,
                    "grouped_ks_host");
                auto offsets_host = ensure_host_buffer(
                    workspace.grouped_offsets_host, seg_offsets_host.size(), torch::kInt32,
                    "grouped_offsets_host");
                std::memcpy(ks_host.data_ptr<int>(), window_ks_host.data(),
                            window_ks_host.size() * sizeof(int));
                std::memcpy(offsets_host.data_ptr<int>(), seg_offsets_host.data(),
                            seg_offsets_host.size() * sizeof(int));
                hipError_t ks_status = hipMemcpyAsync(
                    ks_dev.data_ptr<int>(),
                    ks_host.data_ptr<int>(),
                    window_ks_host.size() * sizeof(int),
                    hipMemcpyHostToDevice,
                    (hipStream_t)at::cuda::getCurrentCUDAStream().stream());
                TORCH_CHECK(ks_status == hipSuccess,
                            "hipMemcpyAsync ks_dev failed: ",
                            hipGetErrorString(ks_status));
                hipError_t offsets_status = hipMemcpyAsync(
                    offsets_dev.data_ptr<int>(),
                    offsets_host.data_ptr<int>(),
                    seg_offsets_host.size() * sizeof(int),
                    hipMemcpyHostToDevice,
                    (hipStream_t)at::cuda::getCurrentCUDAStream().stream());
                TORCH_CHECK(offsets_status == hipSuccess,
                            "hipMemcpyAsync offsets_dev failed: ",
                            hipGetErrorString(offsets_status));
                auto flat_inp = flat_inp_storage.narrow(0, 0, window_nhot);
                auto flat_out = flat_out_storage.narrow(0, 0, window_nhot);
                hipStream_t stream = (hipStream_t)at::cuda::getCurrentCUDAStream().stream();
                if (pair_is_long) {
                    spconv_hip::launch_flatten_indice_pairs_rows<int64_t, int64_t>(
                        indice_pairs.data_ptr<int64_t>(),
                        ks_dev.data_ptr<int>(),
                        offsets_dev.data_ptr<int>(),
                        static_cast<int>(window_ks_host.size()),
                        nmax,
                        gather_axis,
                        scatter_axis,
                        flat_inp.data_ptr<int64_t>(),
                        flat_out.data_ptr<int64_t>(),
                        static_cast<int>(window_nhot),
                        stream);
                } else {
                    spconv_hip::launch_flatten_indice_pairs_rows<int, int>(
                        indice_pairs.data_ptr<int>(),
                        ks_dev.data_ptr<int>(),
                        offsets_dev.data_ptr<int>(),
                        static_cast<int>(window_ks_host.size()),
                        nmax,
                        gather_axis,
                        scatter_axis,
                        flat_inp.data_ptr<int>(),
                        flat_out.data_ptr<int>(),
                        static_cast<int>(window_nhot),
                        stream);
                }
                if (c_in % 4 == 0 && c_out % 4 == 0) {
                    if (pair_is_long) {
                        spconv_hip::launch_gather_rows_f32<int64_t>(
                            features.data_ptr<float>(), flat_inp.data_ptr<int64_t>(),
                            feat_pack.data_ptr<float>(), static_cast<int>(window_nhot), c_in, stream);
                        spconv_hip::launch_gather_rows_f32<int64_t>(
                            out_bp_contig.data_ptr<float>(), flat_out.data_ptr<int64_t>(),
                            out_bp_pack.data_ptr<float>(), static_cast<int>(window_nhot), c_out, stream);
                    } else {
                        spconv_hip::launch_gather_rows_f32<int>(
                            features.data_ptr<float>(), flat_inp.data_ptr<int>(),
                            feat_pack.data_ptr<float>(), static_cast<int>(window_nhot), c_in, stream);
                        spconv_hip::launch_gather_rows_f32<int>(
                            out_bp_contig.data_ptr<float>(), flat_out.data_ptr<int>(),
                            out_bp_pack.data_ptr<float>(), static_cast<int>(window_nhot), c_out, stream);
                    }
                } else {
                    at::index_select_out(feat_pack, features, 0, flat_inp);
                    at::index_select_out(out_bp_pack, out_bp_contig, 0, flat_out);
                }

                hipblasStatus_t status = hipblasSgemmGroupedBatched(
                    handle,
                    transa_array.data(),
                    transb_array.data(),
                    m_array.data(),
                    n_array.data(),
                    k_array.data(),
                    alpha_array.data(),
                    a_array.data(),
                    lda_array.data(),
                    b_array.data(),
                    ldb_array.data(),
                    beta_array.data(),
                    c_array.data(),
                    ldc_array.data(),
                    static_cast<int>(group_count),
                    group_size.data());
                TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS,
                            "hipblasSgemmGroupedBatched failed with status ", static_cast<int>(status));
                begin = end;
            }

            return {din, dfilters};
        }
        return {din, dfilters};
    }
#endif

    int max_nhot = 0;
    for (int i = 0; i < kv; ++i) {
        if (pn[i] > max_nhot) max_nhot = pn[i];
    }
    auto feat_scratch = max_nhot > 0
        ? torch::empty({max_nhot, c_in}, features.options())
        : torch::Tensor();
    auto out_bp_scratch = max_nhot > 0
        ? torch::empty({max_nhot, c_out}, out_bp_contig.options())
        : torch::Tensor();

    for (int i = 0; i < kv; ++i) {
        int nhot = pn[i];
        if (nhot == 0) continue;

        if (subm && !inverse && i == center) {
            auto center_grad = dfilters.select(0, i);
            at::mm_out(center_grad, features.t(), out_bp_contig);
            continue;
        }

        auto pair = indice_pairs.select(0, i);
        auto inp_inds = pair.select(0, gather_axis).slice(0, 0, nhot);
        auto out_inds = pair.select(0, scatter_axis).slice(0, 0, nhot);

        auto inp_gathered = feat_scratch.slice(0, 0, nhot);
        at::index_select_out(inp_gathered, features, 0, inp_inds);    // [nhot, C_in]
        auto out_bp_gathered = out_bp_scratch.slice(0, 0, nhot);
        at::index_select_out(out_bp_gathered, out_bp_contig, 0, out_inds);    // [nhot, C_out]

        // dfilters[i] = inp_gathered^T @ out_bp_gathered
        auto dfilter_i = dfilters.select(0, i);
        at::mm_out(dfilter_i, inp_gathered.t(), out_bp_gathered);
    }

    return {din, dfilters};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_indice_pairs_subm", &get_indice_pairs_subm,
          "SubM conv indice pairs (HIP kernel)");
    m.def("get_indice_pairs_conv", &get_indice_pairs_conv,
          "Regular/transposed conv indice pairs (HIP kernel)");
    m.def("indice_conv_forward",
          [](torch::Tensor features,
             torch::Tensor filters,
             torch::Tensor indice_pairs,
             torch::Tensor indice_pair_num,
             int64_t num_activate_out,
             bool inverse,
             bool subm) {
              return indice_conv_forward(
                  features, filters, indice_pairs, indice_pair_num,
                  num_activate_out, inverse, subm, false);
          },
          "Sparse conv forward with C++ for-loop (eliminates Python overhead)");
#ifdef SPCONV_WITH_HCU_FUSED
    m.def("indice_conv_forward_fused_f16",
              [](torch::Tensor features, torch::Tensor filters,
                 torch::Tensor indice_pairs, torch::Tensor indice_pair_num,
                 int64_t num_activate_out, bool inverse, bool subm) {
                  return indice_conv_forward_fused_f16(
                      features, filters, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm, false, false);
              },
              "Fused fp16 sparse conv forward (gfx936/gfx938)");
    m.def("indice_conv_forward_fused_f16_unique",
              [](torch::Tensor features, torch::Tensor filters,
                 torch::Tensor indice_pairs, torch::Tensor indice_pair_num,
                 int64_t num_activate_out, bool inverse, bool subm) {
                  return indice_conv_forward_fused_f16(
                      features, filters, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm, false, true);
              },
              "Fused fp16 sparse conv forward with unique indices (gfx936/gfx938)");
#endif  // SPCONV_WITH_HCU_FUSED
    m.def("indice_conv_forward_fused",
          [](torch::Tensor features,
             torch::Tensor filters,
             torch::Tensor indice_pairs,
             torch::Tensor indice_pair_num,
             int64_t num_activate_out,
             bool inverse,
             bool subm) {
              return indice_conv_forward_fused(
                  features, filters, indice_pairs, indice_pair_num,
                  num_activate_out, inverse, subm, false);
          },
          "Sparse conv forward with fused HIP kernel");
    m.def("indice_conv_forward_fused_unique",
          [](torch::Tensor features,
             torch::Tensor filters,
             torch::Tensor indice_pairs,
             torch::Tensor indice_pair_num,
             int64_t num_activate_out,
             bool inverse,
             bool subm) {
              return indice_conv_forward_fused(
                  features, filters, indice_pairs, indice_pair_num,
                  num_activate_out, inverse, subm, false, true);
          },
          "Sparse conv forward with fused HIP kernel and unique input pairs");
    m.def("indice_conv_backward", &indice_conv_backward,
          "Sparse conv backward with C++ for-loop");
#ifdef SPCONV_WITH_HCU_FUSED
    m.def("indice_conv_forward_fused_bf16",
              [](torch::Tensor features, torch::Tensor filters,
                 torch::Tensor indice_pairs, torch::Tensor indice_pair_num,
                 int64_t num_activate_out, bool inverse, bool subm) {
                  return indice_conv_forward_fused_bf16(
                      features, filters, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm, false, false);
              },
              "Fused bf16 sparse conv forward (gfx936/gfx938)");
    m.def("indice_conv_forward_fused_bf16_unique",
              [](torch::Tensor features, torch::Tensor filters,
                 torch::Tensor indice_pairs, torch::Tensor indice_pair_num,
                 int64_t num_activate_out, bool inverse, bool subm) {
                  return indice_conv_forward_fused_bf16(
                      features, filters, indice_pairs, indice_pair_num,
                      num_activate_out, inverse, subm, false, true);
              },
              "Fused bf16 sparse conv forward with unique indices (gfx936/gfx938)");
#endif  // SPCONV_WITH_HCU_FUSED
}
