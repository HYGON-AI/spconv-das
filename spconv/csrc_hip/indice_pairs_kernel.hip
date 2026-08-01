#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "hash_table.h"

namespace spconv_hip {

namespace {

__device__ __forceinline__ unsigned long long pack_lut_epoch_value(
    unsigned int epoch,
    int inp)
{
    return (static_cast<unsigned long long>(epoch) << 32) |
           static_cast<unsigned int>(inp + 1);
}

__device__ __forceinline__ int unpack_lut_epoch_value(
    unsigned long long value,
    unsigned int epoch)
{
    if (static_cast<unsigned int>(value >> 32) != epoch) {
        return -1;
    }
    unsigned int encoded_inp = static_cast<unsigned int>(value & 0xffffffffu);
    return encoded_inp == 0 ? -1 : static_cast<int>(encoded_inp) - 1;
}

struct RawWorkspaceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
    int device = -1;
};

struct RawHostBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;
};

struct IndicePairsRawWorkspace {
    RawWorkspaceBuffer subm_hash_keys;
    RawWorkspaceBuffer subm_hash_vals;
    RawWorkspaceBuffer conv_hash_keys;
    RawWorkspaceBuffer conv_hash_vals;
    RawWorkspaceBuffer indice_pairs_uniq;
    RawWorkspaceBuffer indice_pairs_uniq_copy;
    RawWorkspaceBuffer sort_temp;
    RawWorkspaceBuffer sorted_keys;
    RawWorkspaceBuffer d_num_selected;
    RawWorkspaceBuffer unique_out;
    RawWorkspaceBuffer unique_temp;
    RawHostBuffer h_num_selected;
};

thread_local std::vector<IndicePairsRawWorkspace> g_raw_workspaces;

IndicePairsRawWorkspace& get_raw_workspace() {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        device = 0;
    }
    if (static_cast<int>(g_raw_workspaces.size()) <= device) {
        g_raw_workspaces.resize(device + 1);
    }
    return g_raw_workspaces[device];
}

void* ensure_raw_bytes(RawWorkspaceBuffer& buffer, size_t bytes) {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess) {
        device = 0;
    }
    if (bytes == 0) {
        return nullptr;
    }
    if (buffer.ptr == nullptr || buffer.bytes < bytes || buffer.device != device) {
        if (buffer.ptr != nullptr) {
            hipFree(buffer.ptr);
        }
        hipMalloc(&buffer.ptr, bytes);
        buffer.bytes = bytes;
        buffer.device = device;
    }
    return buffer.ptr;
}

template <typename T>
T* ensure_raw(RawWorkspaceBuffer& buffer, size_t count) {
    return static_cast<T*>(ensure_raw_bytes(buffer, count * sizeof(T)));
}

void* ensure_host_bytes(RawHostBuffer& buffer, size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    if (buffer.ptr == nullptr || buffer.bytes < bytes) {
        if (buffer.ptr != nullptr) {
            hipHostFree(buffer.ptr);
        }
        hipHostMalloc(&buffer.ptr, bytes, hipHostMallocDefault);
        buffer.bytes = bytes;
    }
    return buffer.ptr;
}

template <typename T>
T* ensure_host(RawHostBuffer& buffer, size_t count) {
    return static_cast<T*>(ensure_host_bytes(buffer, count * sizeof(T)));
}

}  // namespace

typedef float v2f __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void wave_barrier_64() {
#if defined(__gfx936__) || defined(__gfx938__)
    __builtin_amdgcn_wave_barrier();
#else
    __syncthreads();
#endif
}

// Max supported spatial dimensions (1D, 2D, 3D, 4D).
constexpr int kMaxNDim = 4;

// Convolution problem descriptor — replaces tensorview's ConvProblem.
struct ConvProblem {
    int batch_size;
    int ndim;
    int input_dims[kMaxNDim];
    int output_dims[kMaxNDim];
    int ksize[kMaxNDim];
    int padding[kMaxNDim];
    int stride[kMaxNDim];
    int dilation[kMaxNDim];
};

// Compute linear layout offset from (batch, spatial...) coordinates.
// Equivalent to TensorGeneric layout_npq: key = batch * vol + z*Y*X + y*X + x
template <typename T>
__host__ __device__ __forceinline__
T coord_to_linear(const int* coord, const int* dims, int ndim) {
    T key = (T)coord[0];  // batch
    for (int d = 0; d < ndim; d++) {
        key = key * (T)dims[d] + (T)coord[d + 1];
    }
    return key;
}

// Inverse: linear offset → (batch, spatial...) coordinates.
template <typename T>
__host__ __device__ __forceinline__
void linear_to_coord(T key, int* coord, const int* dims, int ndim) {
    for (int d = ndim - 1; d >= 0; d--) {
        coord[d + 1] = (int)(key % (T)dims[d]);
        key /= (T)dims[d];
    }
    coord[0] = (int)key;  // batch
}

// Decompose linear filter offset into per-dimension filter indices.
__host__ __device__ __forceinline__
void filter_offset_to_ndim(int filter_offset, int* r, const int* ksize, int ndim) {
    for (int d = ndim - 1; d >= 0; d--) {
        r[d] = filter_offset % ksize[d];
        filter_offset /= ksize[d];
    }
}

// ============================================================================
// SubM convolution kernels
// ============================================================================

// Build hash table: insert all input coordinates.
template <typename K>
__global__ void build_subm_hash_table_kernel(
    LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,  // [N, ndim+1]
    ConvProblem problem,
    int num_indices)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices;
         i += gridDim.x * blockDim.x) {
        K key = coord_to_linear<K>(
            indices_in + i * (problem.ndim + 1),
            problem.input_dims,
            problem.ndim);
        table.insert(key, i);
    }
}

// Compute SubM indice pairs. Uses 2D grid: blockIdx.y = filter_offset.
// Only processes half the kernel positions (+ center); symmetry fills the other half.
template <typename K>
__global__ void calc_subm_conv_indices_kernel(
    LinearHashTable<K, int> table,
    const int* __restrict__ indices_in,  // [N, ndim+1]
    int* __restrict__ indice_pairs,      // [2, kv, N]
    int* __restrict__ indice_num_per_loc, // [kv]
    ConvProblem problem,
    int num_indices,
    int indices_pair_size,
    int kv)
{
    int filter_offset = blockIdx.y;
    int filter_offset_mul_ips = filter_offset * indices_pair_size;
    int mirror_offset = kv - 1 - filter_offset;
    int mirror_offset_mul_ips = mirror_offset * indices_pair_size;
    int ips_mul_kv = indices_pair_size * kv;

    if (filter_offset == kv / 2) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            indice_num_per_loc[filter_offset] = num_indices;
        }
        // Center kernel position: identity mapping.
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices;
             i += gridDim.x * blockDim.x) {
            indice_pairs[filter_offset_mul_ips + i] = i;
            indice_pairs[ips_mul_kv + filter_offset_mul_ips + i] = i;
        }
    } else {
        // Compute filter indices for this offset.
        int r[kMaxNDim];
        filter_offset_to_ndim(filter_offset, r, problem.ksize, problem.ndim);

        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < num_indices;
             i += gridDim.x * blockDim.x) {
            const int* cur_idx = indices_in + i * (problem.ndim + 1);

            // Compute neighbor coordinate: input_coord + (r - ksize/2) * dilation
            int neighbor[kMaxNDim + 1];
            neighbor[0] = cur_idx[0];  // batch
            bool valid = true;
            for (int d = 0; d < problem.ndim; d++) {
                int offset_d = r[d] - problem.ksize[d] / 2;
                neighbor[d + 1] = cur_idx[d + 1] + offset_d * problem.dilation[d];
                if (neighbor[d + 1] < 0 || neighbor[d + 1] >= problem.input_dims[d]) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                K key = coord_to_linear<K>(neighbor, problem.input_dims, problem.ndim);
                int slot = table.lookup_offset(key);
                if (slot != -1) {
                    int v = table.values[slot];
                    int old_num = atomicAdd(indice_num_per_loc + filter_offset, 1);
                    // indice_pairs[0] = input (gather source), [1] = output (scatter dest)
                    // For filter_offset: i queries, v is the neighbor found
                    //   → gather from v (input), scatter to i (output)
                    indice_pairs[filter_offset_mul_ips + old_num] = v;
                    indice_pairs[ips_mul_kv + filter_offset_mul_ips + old_num] = i;
                    // Mirror: reversed — gather from i, scatter to v
                    indice_pairs[mirror_offset_mul_ips + old_num] = i;
                    indice_pairs[ips_mul_kv + mirror_offset_mul_ips + old_num] = v;
                }
            }
        }
    }
}

// ============================================================================
// Regular (strided / transposed) convolution kernels
// ============================================================================

// Stage 1: For each input point × filter offset, compute output coord offset.
// Store input index in indice_pairs[0], output coord offset in indice_pairs_uniq.
template <typename K>
__global__ void calc_conv_indices_stage1_kernel(
    const int* __restrict__ indices_in,  // [N, ndim+1]
    int* __restrict__ indice_pairs_in,   // [kv, N] — stores input indices
    K* __restrict__ indice_pairs_uniq,   // [kv * N + 1] — stores output coord offsets
    int* __restrict__ indice_num_per_loc, // [kv]
    ConvProblem problem,
    int num_indices_in,
    int indices_pair_size,
    int kv,
    bool transposed)
{
    int filter_offset = blockIdx.y;
    int fo_mul_ips = filter_offset * indices_pair_size;

    int r[kMaxNDim];
    filter_offset_to_ndim(filter_offset, r, problem.ksize, problem.ndim);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices_in;
         i += gridDim.x * blockDim.x) {
        const int* cur = indices_in + i * (problem.ndim + 1);

        int out_coord[kMaxNDim + 1];
        out_coord[0] = cur[0];  // batch
        bool valid = (out_coord[0] >= 0);

        if (transposed) {
            for (int d = 0; d < problem.ndim; d++) {
                out_coord[d + 1] = cur[d + 1] * problem.stride[d] - problem.padding[d]
                                   + r[d] * problem.dilation[d];
                if (out_coord[d + 1] < 0 || out_coord[d + 1] >= problem.output_dims[d]) {
                    valid = false;
                    break;
                }
            }
        } else {
            for (int d = 0; d < problem.ndim; d++) {
                int tmp = cur[d + 1] + problem.padding[d] - r[d] * problem.dilation[d];
                if (tmp % problem.stride[d] != 0) { valid = false; break; }
                out_coord[d + 1] = tmp / problem.stride[d];
                if (out_coord[d + 1] < 0 || out_coord[d + 1] >= problem.output_dims[d]) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid) {
            int old_num = atomicAdd(indice_num_per_loc + filter_offset, 1);
            K offset = coord_to_linear<K>(out_coord, problem.output_dims, problem.ndim);
            if (old_num < indices_pair_size) {
                indice_pairs_in[fo_mul_ips + old_num] = i;
                indice_pairs_uniq[fo_mul_ips + old_num] = offset;
            }
        }
    }
}

// Initialize uniq buffer with sentinel values.
template <typename K>
__global__ void clean_indices_uniq_kernel(K* data, int64_t size) {
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < size;
         i += (int64_t)gridDim.x * blockDim.x) {
        data[i] = std::numeric_limits<K>::max();
    }
}

// Build hash table from unique output coord offsets, assign sequential indices.
// Also writes output coordinates by inverting the linear offset.
template <typename K>
__global__ void build_conv_hash_and_assign_kernel(
    LinearHashTable<K, int> table,
    int* __restrict__ out_indices,  // [num_out, ndim+1]
    const K* __restrict__ unique_offsets,
    ConvProblem problem,
    int num_out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_out;
         i += gridDim.x * blockDim.x) {
        K offset = unique_offsets[i];
        table.insert(offset, i);
        linear_to_coord<K>(
            offset,
            out_indices + i * (problem.ndim + 1),
            problem.output_dims,
            problem.ndim);
    }
}

// Stage 2: Use hash table to map output coord offsets → sequential output indices.
template <typename K>
__global__ void calc_conv_indices_stage2_kernel(
    LinearHashTable<K, int> table,
    const K* __restrict__ indice_pairs_uniq_before_sort,  // [kv, N]
    int* __restrict__ indice_pairs_out,  // [kv, N]
    int num_indices_in,
    int indices_pair_size)
{
    int filter_offset = blockIdx.y;
    int fo_mul_ips = filter_offset * indices_pair_size;
    const K* uniq_filter = indice_pairs_uniq_before_sort + fo_mul_ips;
    int* out_filter = indice_pairs_out + fo_mul_ips;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices_in;
         i += gridDim.x * blockDim.x) {
        K coord_offset = uniq_filter[i];
        if (coord_offset != std::numeric_limits<K>::max()) {
            int slot = table.lookup_offset(coord_offset);
            if (slot != -1) {
                out_filter[i] = table.values[slot];
            }
        }
    }
}

// ============================================================================
// Fused indice_conv forward helpers
// ============================================================================

template <typename IndexT>
__global__ void build_indice_conv_lut_kernel(
    const IndexT* __restrict__ indice_pairs,   // [kv, 2, N]
    const int* __restrict__ indice_num_per_loc, // [kv]
    int* __restrict__ lut,                     // [num_out, kv]
    int* __restrict__ dup_slots,
    int* __restrict__ dup_inp,
    int* __restrict__ dup_meta,
    int dup_capacity,
    int kv,
    int nmax,
    int gather_axis,
    int scatter_axis,
    int center,
    bool skip_center)
{
    int filter_offset = blockIdx.y;
    if (skip_center && filter_offset == center) {
        return;
    }

    int nhot = indice_num_per_loc[filter_offset];
    int pair_base = filter_offset * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nhot;
         i += gridDim.x * blockDim.x) {
        int inp = static_cast<int>(inp_inds[i]);
        int out = static_cast<int>(out_inds[i]);
        if (inp >= 0 && out >= 0) {
            int slot = out * kv + filter_offset;
            int prev = atomicCAS(lut + slot, -1, inp);
            if (prev != -1 && prev != inp) {
                int dup_idx = atomicAdd(dup_meta, 1);
                if (dup_idx < dup_capacity) {
                    dup_slots[dup_idx] = slot;
                    dup_inp[dup_idx] = inp;
                }
            }
        }
    }
}

template <typename IndexT>
__global__ void build_indice_conv_lut_epoch_kernel(
    const IndexT* __restrict__ indice_pairs,   // [kv, 2, N]
    const int* __restrict__ indice_num_per_loc, // [kv]
    unsigned long long* __restrict__ lut_state, // [num_out, kv], high32=epoch, low32=inp+1
    int* __restrict__ dup_slots,
    int* __restrict__ dup_inp,
    int* __restrict__ dup_meta,
    int dup_capacity,
    int kv,
    int nmax,
    int gather_axis,
    int scatter_axis,
    int center,
    bool skip_center,
    unsigned int lut_epoch)
{
    int filter_offset = blockIdx.y;
    if (skip_center && filter_offset == center) {
        return;
    }

    int nhot = indice_num_per_loc[filter_offset];
    int pair_base = filter_offset * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nhot;
         i += gridDim.x * blockDim.x) {
        int inp = static_cast<int>(inp_inds[i]);
        int out = static_cast<int>(out_inds[i]);
        if (inp < 0 || out < 0) {
            continue;
        }

        int slot = out * kv + filter_offset;
        unsigned long long desired = pack_lut_epoch_value(lut_epoch, inp);
        unsigned long long old = lut_state[slot];
        while (true) {
            int old_inp = unpack_lut_epoch_value(old, lut_epoch);
            if (old_inp >= 0) {
                if (old_inp != inp) {
                    int dup_idx = atomicAdd(dup_meta, 1);
                    if (dup_idx < dup_capacity) {
                        dup_slots[dup_idx] = slot;
                        dup_inp[dup_idx] = inp;
                    }
                }
                break;
            }

            unsigned long long observed = atomicCAS(lut_state + slot, old, desired);
            if (observed == old) {
                break;
            }
            old = observed;
        }
    }
}

__global__ void mirror_indice_num_per_loc_kernel(
    int* __restrict__ indice_num_per_loc,
    int kv)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int half = kv / 2;
    if (k >= half) {
        return;
    }
    indice_num_per_loc[kv - 1 - k] = indice_num_per_loc[k];
}

__global__ void indice_conv_accumulate_duplicates_f32_kernel(
    const float* __restrict__ features,
    const float* __restrict__ filters,
    const int* __restrict__ dup_slots,
    const int* __restrict__ dup_inp,
    const int* __restrict__ dup_count_ptr,
    float* __restrict__ out,
    int kv,
    int c_in,
    int c_out,
    bool filters_transposed)
{
    int total = dup_count_ptr[0] * c_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        int dup_idx = idx / c_out;
        int co = idx % c_out;
        int slot = dup_slots[dup_idx];
        int inp = dup_inp[dup_idx];
        int out_row = slot / kv;
        int k = slot % kv;

        const float* feat = features + inp * c_in;
        float acc = 0.0f;
        for (int ci = 0; ci < c_in; ++ci) {
            float weight = filters_transposed
                ? filters[(k * c_out + co) * c_in + ci]
                : filters[(k * c_in + ci) * c_out + co];
            acc += feat[ci] * weight;
        }
        atomicAdd(out + out_row * c_out + co, acc);
    }
}

template <typename IndexT, typename FlatIndexT>
__global__ void flatten_indice_pairs_rows_kernel(
    const IndexT* __restrict__ indice_pairs,   // [kv, 2, N]
    const int* __restrict__ window_ks,         // [num_segs]
    const int* __restrict__ seg_offsets,       // [num_segs + 1]
    int num_segs,
    int nmax,
    int gather_axis,
    int scatter_axis,
    FlatIndexT* __restrict__ flat_inp,
    FlatIndexT* __restrict__ flat_out,
    int total_nhot)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_nhot) {
        return;
    }

    int seg = 0;
    while (seg + 1 < num_segs && idx >= seg_offsets[seg + 1]) {
        ++seg;
    }
    int local = idx - seg_offsets[seg];
    int k = window_ks[seg];
    int pair_base = k * 2 * nmax;
    flat_inp[idx] = static_cast<FlatIndexT>(
        indice_pairs[pair_base + gather_axis * nmax + local]);
    flat_out[idx] = static_cast<FlatIndexT>(
        indice_pairs[pair_base + scatter_axis * nmax + local]);
}

__device__ __forceinline__ float load_filter_value_f32(
    const float* __restrict__ filters,
    int k,
    int ci,
    int co,
    int c_in,
    int c_out,
    bool filters_transposed)
{
    return filters_transposed
        ? filters[(k * c_out + co) * c_in + ci]
        : filters[(k * c_in + ci) * c_out + co];
}

__global__ void indice_conv_fused_forward_f32_mmac_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ filters,    // [kv, C_in, C_out]
    const int* __restrict__ lut,          // [N_out, kv]
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,              // [N_out, C_out]
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];

    int lane = threadIdx.x & 63;
    int row = lane % 16;
    int col_group = lane / 16;
    int out_row = blockIdx.x * 16 + row;
    int col0 = blockIdx.y * 16 + col_group + 0;
    int col1 = blockIdx.y * 16 + col_group + 4;
    int col2 = blockIdx.y * 16 + col_group + 8;
    int col3 = blockIdx.y * 16 + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        if (col0 < c_out) acc[0] = out[out_row * c_out + col0];
        if (col1 < c_out) acc[1] = out[out_row * c_out + col1];
        if (col2 < c_out) acc[2] = out[out_row * c_out + col2];
        if (col3 < c_out) acc[3] = out[out_row * c_out + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (lane < 16) {
            int load_row = blockIdx.x * 16 + lane;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[lane] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[lane] = -1;
            }
        }
        wave_barrier_64();

        int inp = sh_inp[row];
        for (int ck = 0; ck < c_in; ck += 8) {
            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};

            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid && inp >= 0) {
                if (a_ci0 < c_in) va[0] = features[inp * c_in + a_ci0];
                if (a_ci1 < c_in) va[1] = features[inp * c_in + a_ci1];
            }

            int b_col = blockIdx.y * 16 + row;
            if (b_col < c_out) {
                if (a_ci0 < c_in) {
                    vb[0] = load_filter_value_f32(filters, k, a_ci0, b_col, c_in, c_out,
                                                  filters_transposed);
                }
                if (a_ci1 < c_in) {
                    vb[1] = load_filter_value_f32(filters, k, a_ci1, b_col, c_in, c_out,
                                                  filters_transposed);
                }
            }

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
        }
    }

    if (row_valid) {
        if (col0 < c_out) out[out_row * c_out + col0] = acc[0];
        if (col1 < c_out) out[out_row * c_out + col1] = acc[1];
        if (col2 < c_out) out[out_row * c_out + col2] = acc[2];
        if (col3 < c_out) out[out_row * c_out + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)c_in;
    (void)c_out;
    (void)lut_epoch;
    (void)accumulate_existing;
    (void)filters_transposed;
#endif
}

template <int CIn, int COut>
__global__ void indice_conv_fused_forward_f32_mmac_static_single_cc_kernel(
    const float* __restrict__ features,
    const float* __restrict__ filters,
    const int* __restrict__ lut,
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,
    int num_out,
    int kv,
    unsigned int lut_epoch,
    bool accumulate_existing)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];

    int lane = threadIdx.x & 63;
    int row = lane % 16;
    int col_group = lane / 16;
    int out_row = blockIdx.x * 16 + row;
    int col0 = blockIdx.y * 16 + col_group + 0;
    int col1 = blockIdx.y * 16 + col_group + 4;
    int col2 = blockIdx.y * 16 + col_group + 8;
    int col3 = blockIdx.y * 16 + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        acc[0] = out[out_row * COut + col0];
        acc[1] = out[out_row * COut + col1];
        acc[2] = out[out_row * COut + col2];
        acc[3] = out[out_row * COut + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (lane < 16) {
            int load_row = blockIdx.x * 16 + lane;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[lane] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[lane] = -1;
            }
        }
        wave_barrier_64();

        int inp = sh_inp[row];
        #pragma unroll
        for (int ck = 0; ck < CIn; ck += 8) {
            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};

            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid && inp >= 0) {
                va[0] = features[inp * CIn + a_ci0];
                va[1] = features[inp * CIn + a_ci1];
            }

            int b_col = blockIdx.y * 16 + row;
            vb[0] = filters[(k * CIn + a_ci0) * COut + b_col];
            vb[1] = filters[(k * CIn + a_ci1) * COut + b_col];

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
        }
    }

    if (row_valid) {
        out[out_row * COut + col0] = acc[0];
        out[out_row * COut + col1] = acc[1];
        out[out_row * COut + col2] = acc[2];
        out[out_row * COut + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)lut_epoch;
    (void)accumulate_existing;
#endif
}

__global__ void indice_conv_fused_forward_f32_mmac_multiwave_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ filters,    // [kv, C_in, C_out]
    const int* __restrict__ lut,          // [N_out, kv]
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,              // [N_out, C_out]
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];
    __shared__ float sh_feat[16][8];

    int wave_count = blockDim.x >> 6;
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = lane & 15;
    int col_group = lane >> 4;
    int out_row = blockIdx.x * 16 + row;
    int tile_col_base = blockIdx.y * (wave_count * 16) + wave * 16;
    int col0 = tile_col_base + col_group + 0;
    int col1 = tile_col_base + col_group + 4;
    int col2 = tile_col_base + col_group + 8;
    int col3 = tile_col_base + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        if (col0 < c_out) acc[0] = out[out_row * c_out + col0];
        if (col1 < c_out) acc[1] = out[out_row * c_out + col1];
        if (col2 < c_out) acc[2] = out[out_row * c_out + col2];
        if (col3 < c_out) acc[3] = out[out_row * c_out + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (threadIdx.x < 16) {
            int load_row = blockIdx.x * 16 + threadIdx.x;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[threadIdx.x] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[threadIdx.x] = -1;
            }
        }
        __syncthreads();

        for (int ck = 0; ck < c_in; ck += 8) {
            int load_idx = threadIdx.x;
            if (load_idx < 16 * 8) {
                int load_row = load_idx >> 3;
                int load_k = load_idx & 7;
                int ci = ck + load_k;
                int inp = sh_inp[load_row];
                sh_feat[load_row][load_k] =
                    (inp >= 0 && ci < c_in) ? features[inp * c_in + ci] : 0.0f;
            }
            __syncthreads();

            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};
            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid) {
                if (a_ci0 < c_in) va[0] = sh_feat[row][a_ci0 - ck];
                if (a_ci1 < c_in) va[1] = sh_feat[row][a_ci1 - ck];
            }

            int b_col = tile_col_base + row;
            if (b_col < c_out) {
                if (a_ci0 < c_in) {
                    vb[0] = load_filter_value_f32(filters, k, a_ci0, b_col, c_in, c_out,
                                                  filters_transposed);
                }
                if (a_ci1 < c_in) {
                    vb[1] = load_filter_value_f32(filters, k, a_ci1, b_col, c_in, c_out,
                                                  filters_transposed);
                }
            }

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            __syncthreads();
        }
    }

    if (row_valid) {
        if (col0 < c_out) out[out_row * c_out + col0] = acc[0];
        if (col1 < c_out) out[out_row * c_out + col1] = acc[1];
        if (col2 < c_out) out[out_row * c_out + col2] = acc[2];
        if (col3 < c_out) out[out_row * c_out + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)c_in;
    (void)c_out;
    (void)lut_epoch;
    (void)accumulate_existing;
    (void)filters_transposed;
#endif
}

__global__ void indice_conv_fused_forward_f32_mmac_multiwave_c64_kernel(
    const float* __restrict__ features,   // [N_in, 64]
    const float* __restrict__ filters,    // [kv, 64, C_out]
    const int* __restrict__ lut,          // [N_out, kv]
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,              // [N_out, C_out]
    int num_out,
    int kv,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];
    __shared__ float sh_feat[16][8];

    int wave_count = blockDim.x >> 6;
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = lane & 15;
    int col_group = lane >> 4;
    int out_row = blockIdx.x * 16 + row;
    int tile_col_base = blockIdx.y * (wave_count * 16) + wave * 16;
    int col0 = tile_col_base + col_group + 0;
    int col1 = tile_col_base + col_group + 4;
    int col2 = tile_col_base + col_group + 8;
    int col3 = tile_col_base + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        if (col0 < c_out) acc[0] = out[out_row * c_out + col0];
        if (col1 < c_out) acc[1] = out[out_row * c_out + col1];
        if (col2 < c_out) acc[2] = out[out_row * c_out + col2];
        if (col3 < c_out) acc[3] = out[out_row * c_out + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (threadIdx.x < 16) {
            int load_row = blockIdx.x * 16 + threadIdx.x;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[threadIdx.x] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[threadIdx.x] = -1;
            }
        }
        __syncthreads();

        for (int ck = 0; ck < 64; ck += 8) {
            int load_idx = threadIdx.x;
            if (load_idx < 16 * 8) {
                int load_row = load_idx >> 3;
                int load_k = load_idx & 7;
                int inp = sh_inp[load_row];
                sh_feat[load_row][load_k] =
                    inp >= 0 ? features[inp * 64 + ck + load_k] : 0.0f;
            }
            __syncthreads();

            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};
            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid) {
                va[0] = sh_feat[row][a_ci0 - ck];
                va[1] = sh_feat[row][a_ci1 - ck];
            }

            int b_col = tile_col_base + row;
            if (b_col < c_out) {
                vb[0] = filters[(k * 64 + a_ci0) * c_out + b_col];
                vb[1] = filters[(k * 64 + a_ci1) * c_out + b_col];
            }

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            __syncthreads();
        }
    }

    if (row_valid) {
        if (col0 < c_out) out[out_row * c_out + col0] = acc[0];
        if (col1 < c_out) out[out_row * c_out + col1] = acc[1];
        if (col2 < c_out) out[out_row * c_out + col2] = acc[2];
        if (col3 < c_out) out[out_row * c_out + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)c_out;
    (void)lut_epoch;
    (void)accumulate_existing;
#endif
}

__global__ void indice_conv_fused_forward_f32_mmac_multiwave_c64x64_kernel(
    const float* __restrict__ features,   // [N_in, 64]
    const float* __restrict__ filters,    // [kv, 64, 64]
    const int* __restrict__ lut,          // [N_out, kv]
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,              // [N_out, 64]
    int num_out,
    int kv,
    unsigned int lut_epoch,
    bool accumulate_existing)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];
    __shared__ float sh_feat[16][8];

    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = lane & 15;
    int col_group = lane >> 4;
    int out_row = blockIdx.x * 16 + row;
    int tile_col_base = wave * 16;
    int col0 = tile_col_base + col_group + 0;
    int col1 = tile_col_base + col_group + 4;
    int col2 = tile_col_base + col_group + 8;
    int col3 = tile_col_base + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        acc[0] = out[out_row * 64 + col0];
        acc[1] = out[out_row * 64 + col1];
        acc[2] = out[out_row * 64 + col2];
        acc[3] = out[out_row * 64 + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (threadIdx.x < 16) {
            int load_row = blockIdx.x * 16 + threadIdx.x;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[threadIdx.x] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[threadIdx.x] = -1;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int ck = 0; ck < 64; ck += 8) {
            int load_idx = threadIdx.x;
            if (load_idx < 16 * 8) {
                int load_row = load_idx >> 3;
                int load_k = load_idx & 7;
                int inp = sh_inp[load_row];
                sh_feat[load_row][load_k] =
                    inp >= 0 ? features[inp * 64 + ck + load_k] : 0.0f;
            }
            __syncthreads();

            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};
            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid) {
                va[0] = sh_feat[row][a_ci0 - ck];
                va[1] = sh_feat[row][a_ci1 - ck];
            }

            int b_col = tile_col_base + row;
            vb[0] = filters[(k * 64 + a_ci0) * 64 + b_col];
            vb[1] = filters[(k * 64 + a_ci1) * 64 + b_col];

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            __syncthreads();
        }
    }

    if (row_valid) {
        out[out_row * 64 + col0] = acc[0];
        out[out_row * 64 + col1] = acc[1];
        out[out_row * 64 + col2] = acc[2];
        out[out_row * 64 + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)lut_epoch;
    (void)accumulate_existing;
#endif
}

template <int CIn>
__global__ void indice_conv_fused_forward_f32_mmac_multiwave_static_c_kernel(
    const float* __restrict__ features,
    const float* __restrict__ filters,
    const int* __restrict__ lut,
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,
    int num_out,
    int kv,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[16];
    __shared__ float sh_feat[16][8];

    int wave_count = blockDim.x >> 6;
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = lane & 15;
    int col_group = lane >> 4;
    int out_row = blockIdx.x * 16 + row;
    int tile_col_base = blockIdx.y * (wave_count * 16) + wave * 16;
    int col0 = tile_col_base + col_group + 0;
    int col1 = tile_col_base + col_group + 4;
    int col2 = tile_col_base + col_group + 8;
    int col3 = tile_col_base + col_group + 12;
    bool row_valid = out_row < num_out;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};
    if (accumulate_existing && row_valid) {
        if (col0 < c_out) acc[0] = out[out_row * c_out + col0];
        if (col1 < c_out) acc[1] = out[out_row * c_out + col1];
        if (col2 < c_out) acc[2] = out[out_row * c_out + col2];
        if (col3 < c_out) acc[3] = out[out_row * c_out + col3];
    }

    for (int k = 0; k < kv; ++k) {
        if (threadIdx.x < 16) {
            int load_row = blockIdx.x * 16 + threadIdx.x;
            if (load_row < num_out) {
                int slot = load_row * kv + k;
                sh_inp[threadIdx.x] = lut_state
                    ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                    : lut[slot];
            } else {
                sh_inp[threadIdx.x] = -1;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int ck = 0; ck < CIn; ck += 8) {
            int load_idx = threadIdx.x;
            if (load_idx < 16 * 8) {
                int load_row = load_idx >> 3;
                int load_k = load_idx & 7;
                int inp = sh_inp[load_row];
                sh_feat[load_row][load_k] =
                    inp >= 0 ? features[inp * CIn + ck + load_k] : 0.0f;
            }
            __syncthreads();

            v2f va = {0.0f, 0.0f};
            v2f vb = {0.0f, 0.0f};
            int a_ci0 = ck + col_group * 2 + 0;
            int a_ci1 = ck + col_group * 2 + 1;
            if (row_valid) {
                va[0] = sh_feat[row][a_ci0 - ck];
                va[1] = sh_feat[row][a_ci1 - ck];
            }

            int b_col = tile_col_base + row;
            if (b_col < c_out) {
                vb[0] = filters[(k * CIn + a_ci0) * c_out + b_col];
                vb[1] = filters[(k * CIn + a_ci1) * c_out + b_col];
            }

            acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            __syncthreads();
        }
    }

    if (row_valid) {
        if (col0 < c_out) out[out_row * c_out + col0] = acc[0];
        if (col1 < c_out) out[out_row * c_out + col1] = acc[1];
        if (col2 < c_out) out[out_row * c_out + col2] = acc[2];
        if (col3 < c_out) out[out_row * c_out + col3] = acc[3];
    }
#else
    (void)features;
    (void)filters;
    (void)lut;
    (void)lut_state;
    (void)out;
    (void)num_out;
    (void)kv;
    (void)c_out;
    (void)lut_epoch;
    (void)accumulate_existing;
#endif
}

__global__ void indice_conv_fused_forward_f32_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ filters,    // [kv, C_in, C_out]
    const int* __restrict__ lut,          // [N_out, kv]
    const unsigned long long* __restrict__ lut_state,
    float* __restrict__ out,              // [N_out, C_out]
    int num_out,
    int kv,
    int c_in,
    int c_out,
    unsigned int lut_epoch,
    bool accumulate_existing,
    bool filters_transposed)
{
    constexpr int kRowTile = 8;
    constexpr int kCoutChunk = 32;
    constexpr int kKTile = 8;
    __shared__ float sh_feat[kRowTile][kKTile];
    __shared__ float sh_weight[kKTile][kCoutChunk];

    int tid = threadIdx.x;
    int local_row = tid / kCoutChunk;
    int local_col = tid % kCoutChunk;

    int out_row = blockIdx.x * kRowTile + local_row;
    int out_col = blockIdx.y * kCoutChunk + local_col;
    bool row_valid = out_row < num_out;
    bool col_valid = out_col < c_out;
    bool out_valid = row_valid && col_valid;

    float acc = (accumulate_existing && out_valid) ? out[out_row * c_out + out_col] : 0.0f;
    int safe_out_row = row_valid ? out_row : 0;
    const int* row_lut = lut + safe_out_row * kv;

    for (int k = 0; k < kv; ++k) {
        for (int ck = 0; ck < c_in; ck += kKTile) {
            int feat_load_idx = tid;
            if (feat_load_idx < kRowTile * kKTile) {
                int fr = feat_load_idx / kKTile;
                int fk = feat_load_idx % kKTile;
                int load_row = blockIdx.x * kRowTile + fr;
                int load_inp = -1;
                if (load_row < num_out) {
                    int slot = load_row * kv + k;
                    load_inp = lut_state
                        ? unpack_lut_epoch_value(lut_state[slot], lut_epoch)
                        : lut[slot];
                }
                int ci = ck + fk;
                sh_feat[fr][fk] = (load_inp >= 0 && ci < c_in)
                    ? features[load_inp * c_in + ci]
                    : 0.0f;
            }

            int weight_load_idx = tid;
            int wk = weight_load_idx / kCoutChunk;
            int wc = weight_load_idx % kCoutChunk;
            if (wk < kKTile) {
                int ci = ck + wk;
                int co = blockIdx.y * kCoutChunk + wc;
                sh_weight[wk][wc] = (ci < c_in && co < c_out)
                    ? load_filter_value_f32(filters, k, ci, co, c_in, c_out,
                                            filters_transposed)
                    : 0.0f;
            }
            __syncthreads();

            if (out_valid) {
                #pragma unroll
                for (int kk = 0; kk < kKTile; ++kk) {
                    acc += sh_feat[local_row][kk] * sh_weight[kk][local_col];
                }
            }
            __syncthreads();
        }
    }

    if (out_valid) {
        out[out_row * c_out + out_col] = acc;
    }
}

template <typename IndexT>
__global__ void indice_conv_dfilters_f32_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ out_bp,     // [N_out, C_out]
    const IndexT* __restrict__ indice_pairs,  // [kv, 2, N_max]
    const int* __restrict__ indice_num_per_loc, // [kv]
    float* __restrict__ dfilters,         // [kv, C_in, C_out]
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis)
{
    constexpr int kTile = 16;
    constexpr int kChunk = 8;
    __shared__ int sh_inp[kChunk];
    __shared__ int sh_out[kChunk];
    __shared__ float sh_feat[kChunk][kTile];
    __shared__ float sh_bp[kChunk][kTile];

    int k = blockIdx.z;
    int ci = blockIdx.y * kTile + threadIdx.y;
    int co = blockIdx.x * kTile + threadIdx.x;
    int nhot = indice_num_per_loc[k];
    int linear_tid = threadIdx.y * kTile + threadIdx.x;

    int pair_base = k * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    float acc = 0.0f;

    for (int start = 0; start < nhot; start += kChunk) {
        if (threadIdx.y == 0 && threadIdx.x < kChunk) {
            int idx = start + threadIdx.x;
            sh_inp[threadIdx.x] = idx < nhot ? static_cast<int>(inp_inds[idx]) : -1;
            sh_out[threadIdx.x] = idx < nhot ? static_cast<int>(out_inds[idx]) : -1;
        }
        __syncthreads();

        if (linear_tid < kChunk * kTile) {
            int kk = linear_tid / kTile;
            int ci_off = linear_tid % kTile;
            int inp = sh_inp[kk];
            int feat_ci = blockIdx.y * kTile + ci_off;
            sh_feat[kk][ci_off] = (inp >= 0 && feat_ci < c_in)
                ? features[inp * c_in + feat_ci]
                : 0.0f;
        } else {
            int idx2 = linear_tid - kChunk * kTile;
            if (idx2 < kChunk * kTile) {
                int kk = idx2 / kTile;
                int co_off = idx2 % kTile;
                int out = sh_out[kk];
                int grad_co = blockIdx.x * kTile + co_off;
                sh_bp[kk][co_off] = (out >= 0 && grad_co < c_out)
                    ? out_bp[out * c_out + grad_co]
                    : 0.0f;
            }
        }
        __syncthreads();

        if (ci < c_in && co < c_out) {
            #pragma unroll
            for (int kk = 0; kk < kChunk; ++kk) {
                acc += sh_feat[kk][threadIdx.y] * sh_bp[kk][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (ci < c_in && co < c_out) {
        dfilters[(k * c_in + ci) * c_out + co] = acc;
    }
}

template <typename IndexT>
__global__ void indice_conv_dfilters_f32_mmac_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ out_bp,     // [N_out, C_out]
    const IndexT* __restrict__ indice_pairs,  // [kv, 2, N_max]
    const int* __restrict__ indice_num_per_loc, // [kv]
    float* __restrict__ dfilters,         // [kv, C_in, C_out]
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis)
{
#if defined(__gfx936__) || defined(__gfx938__)
    constexpr int kPairChunk = 64;
    __shared__ int sh_inp[kPairChunk];
    __shared__ int sh_out[kPairChunk];

    int lane = threadIdx.x & 63;
    int row = lane % 16;
    int col_group = lane / 16;
    int k = blockIdx.z;
    int ci = blockIdx.y * 16 + row;
    int co_vec = blockIdx.x * 16 + row;
    int co0 = blockIdx.x * 16 + col_group + 0;
    int co1 = blockIdx.x * 16 + col_group + 4;
    int co2 = blockIdx.x * 16 + col_group + 8;
    int co3 = blockIdx.x * 16 + col_group + 12;
    int nhot = indice_num_per_loc[k];

    int pair_base = k * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int start = 0; start < nhot; start += kPairChunk) {
        int valid_pairs = nhot - start;
        if (valid_pairs > kPairChunk) {
            valid_pairs = kPairChunk;
        }
        int valid_steps = (valid_pairs + 7) >> 3;
        if (lane < kPairChunk) {
            int idx = start + lane;
            sh_inp[lane] = idx < nhot ? static_cast<int>(inp_inds[idx]) : -1;
            sh_out[lane] = idx < nhot ? static_cast<int>(out_inds[idx]) : -1;
        }
        wave_barrier_64();

        if (valid_steps == 8) {
            #pragma unroll
            for (int base = 0; base < kPairChunk; base += 8) {
                v2f va = {0.0f, 0.0f};
                v2f vb = {0.0f, 0.0f};
                int kk0 = col_group * 2 + base;
                int kk1 = col_group * 2 + base + 1;
                if (ci < c_in) {
                    int inp0 = sh_inp[kk0];
                    int inp1 = sh_inp[kk1];
                    if (inp0 >= 0) va[0] = features[inp0 * c_in + ci];
                    if (inp1 >= 0) va[1] = features[inp1 * c_in + ci];
                }
                if (co_vec < c_out) {
                    int out0 = sh_out[kk0];
                    int out1 = sh_out[kk1];
                    if (out0 >= 0) vb[0] = out_bp[out0 * c_out + co_vec];
                    if (out1 >= 0) vb[1] = out_bp[out1 * c_out + co_vec];
                }
                acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            }
        } else {
            for (int step = 0; step < valid_steps; ++step) {
                int base = step * 8;
                v2f va = {0.0f, 0.0f};
                v2f vb = {0.0f, 0.0f};
                int kk0 = col_group * 2 + base;
                int kk1 = col_group * 2 + base + 1;
                if (ci < c_in) {
                    int inp0 = sh_inp[kk0];
                    int inp1 = sh_inp[kk1];
                    if (inp0 >= 0) va[0] = features[inp0 * c_in + ci];
                    if (inp1 >= 0) va[1] = features[inp1 * c_in + ci];
                }
                if (co_vec < c_out) {
                    int out0 = sh_out[kk0];
                    int out1 = sh_out[kk1];
                    if (out0 >= 0) vb[0] = out_bp[out0 * c_out + co_vec];
                    if (out1 >= 0) vb[1] = out_bp[out1 * c_out + co_vec];
                }
                acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            }
        }
    }

    if (ci < c_in) {
        if (co0 < c_out) dfilters[(k * c_in + ci) * c_out + co0] = acc[0];
        if (co1 < c_out) dfilters[(k * c_in + ci) * c_out + co1] = acc[1];
        if (co2 < c_out) dfilters[(k * c_in + ci) * c_out + co2] = acc[2];
        if (co3 < c_out) dfilters[(k * c_in + ci) * c_out + co3] = acc[3];
    }
#else
    (void)features;
    (void)out_bp;
    (void)indice_pairs;
    (void)indice_num_per_loc;
    (void)dfilters;
    (void)kv;
    (void)c_in;
    (void)c_out;
    (void)nmax;
    (void)gather_axis;
    (void)scatter_axis;
#endif
}

template <typename IndexT>
__global__ void indice_conv_dfilters_f32_mmac_multiwave_kernel(
    const float* __restrict__ features,   // [N_in, C_in]
    const float* __restrict__ out_bp,     // [N_out, C_out]
    const IndexT* __restrict__ indice_pairs,  // [kv, 2, N_max]
    const int* __restrict__ indice_num_per_loc, // [kv]
    float* __restrict__ dfilters,         // [kv, C_in, C_out]
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis)
{
#if defined(__gfx936__) || defined(__gfx938__)
    __shared__ int sh_inp[8];
    __shared__ int sh_out[8];
    __shared__ float sh_feat[16][8];

    int wave_count = blockDim.x >> 6;
    int wave = threadIdx.x >> 6;
    int lane = threadIdx.x & 63;
    int row = lane & 15;
    int col_group = lane >> 4;
    int k = blockIdx.z;
    int ci = blockIdx.y * 16 + row;
    int tile_col_base = blockIdx.x * (wave_count * 16) + wave * 16;
    int co0 = tile_col_base + col_group + 0;
    int co1 = tile_col_base + col_group + 4;
    int co2 = tile_col_base + col_group + 8;
    int co3 = tile_col_base + col_group + 12;
    int nhot = indice_num_per_loc[k];

    int pair_base = k * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int start = 0; start < nhot; start += 8) {
        if (threadIdx.x < 8) {
            int idx = start + threadIdx.x;
            sh_inp[threadIdx.x] = idx < nhot ? static_cast<int>(inp_inds[idx]) : -1;
            sh_out[threadIdx.x] = idx < nhot ? static_cast<int>(out_inds[idx]) : -1;
        }
        __syncthreads();

        int load_idx = threadIdx.x;
        if (load_idx < 16 * 8) {
            int load_row = load_idx >> 3;
            int load_k = load_idx & 7;
            int inp = sh_inp[load_k];
            int feat_ci = blockIdx.y * 16 + load_row;
            sh_feat[load_row][load_k] =
                (inp >= 0 && feat_ci < c_in) ? features[inp * c_in + feat_ci] : 0.0f;
        }
        __syncthreads();

        int kk0 = col_group * 2 + 0;
        int kk1 = col_group * 2 + 1;
        int out0 = sh_out[kk0];
        int out1 = sh_out[kk1];

        v2f va = {0.0f, 0.0f};
        v2f vb = {0.0f, 0.0f};
        if (ci < c_in) {
            va[0] = sh_feat[row][kk0];
            va[1] = sh_feat[row][kk1];
        }
        int b_col = tile_col_base + row;
        if (b_col < c_out) {
            if (out0 >= 0) vb[0] = out_bp[out0 * c_out + b_col];
            if (out1 >= 0) vb[1] = out_bp[out1 * c_out + b_col];
        }

        acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
        __syncthreads();
    }

    if (ci < c_in) {
        if (co0 < c_out) dfilters[(k * c_in + ci) * c_out + co0] = acc[0];
        if (co1 < c_out) dfilters[(k * c_in + ci) * c_out + co1] = acc[1];
        if (co2 < c_out) dfilters[(k * c_in + ci) * c_out + co2] = acc[2];
        if (co3 < c_out) dfilters[(k * c_in + ci) * c_out + co3] = acc[3];
    }
#else
    (void)features;
    (void)out_bp;
    (void)indice_pairs;
    (void)indice_num_per_loc;
    (void)dfilters;
    (void)kv;
    (void)c_in;
    (void)c_out;
    (void)nmax;
    (void)gather_axis;
    (void)scatter_axis;
#endif
}

template <typename IndexT>
__global__ void indice_conv_dfilters_f32_mmac_split_kernel(
    const float* __restrict__ features,
    const float* __restrict__ out_bp,
    const IndexT* __restrict__ indice_pairs,
    const int* __restrict__ indice_num_per_loc,
    float* __restrict__ dfilters,
    int kv,
    int c_in,
    int c_out,
    int nmax,
    int gather_axis,
    int scatter_axis,
    bool skip_center,
    int center,
    int split_count)
{
#if defined(__gfx936__) || defined(__gfx938__)
    constexpr int kPairChunk = 64;
    __shared__ int sh_inp[kPairChunk];
    __shared__ int sh_out[kPairChunk];

    int lane = threadIdx.x & 63;
    int row = lane % 16;
    int col_group = lane / 16;
    int split_id = blockIdx.z / kv;
    int k = blockIdx.z - split_id * kv;
    int ci = blockIdx.y * 16 + row;
    int co_vec = blockIdx.x * 16 + row;
    int co0 = blockIdx.x * 16 + col_group + 0;
    int co1 = blockIdx.x * 16 + col_group + 4;
    int co2 = blockIdx.x * 16 + col_group + 8;
    int co3 = blockIdx.x * 16 + col_group + 12;
    int nhot = indice_num_per_loc[k];

    if (skip_center && k == center) {
        return;
    }

    int pairs_per_split = (nhot + split_count - 1) / split_count;
    int split_begin = split_id * pairs_per_split;
    int split_end = split_begin + pairs_per_split;
    if (split_end > nhot) {
        split_end = nhot;
    }
    if (split_begin >= split_end) {
        return;
    }

    int pair_base = k * 2 * nmax;
    const IndexT* inp_inds = indice_pairs + pair_base + gather_axis * nmax;
    const IndexT* out_inds = indice_pairs + pair_base + scatter_axis * nmax;

    v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int start = split_begin; start < split_end; start += kPairChunk) {
        int valid_pairs = split_end - start;
        if (valid_pairs > kPairChunk) {
            valid_pairs = kPairChunk;
        }
        int valid_steps = (valid_pairs + 7) >> 3;
        if (lane < kPairChunk) {
            int idx = start + lane;
            sh_inp[lane] = idx < split_end ? static_cast<int>(inp_inds[idx]) : -1;
            sh_out[lane] = idx < split_end ? static_cast<int>(out_inds[idx]) : -1;
        }
        wave_barrier_64();

        if (valid_steps == 8) {
            #pragma unroll
            for (int base = 0; base < kPairChunk; base += 8) {
                v2f va = {0.0f, 0.0f};
                v2f vb = {0.0f, 0.0f};
                int kk0 = col_group * 2 + base;
                int kk1 = col_group * 2 + base + 1;
                if (ci < c_in) {
                    int inp0 = sh_inp[kk0];
                    int inp1 = sh_inp[kk1];
                    if (inp0 >= 0) va[0] = features[inp0 * c_in + ci];
                    if (inp1 >= 0) va[1] = features[inp1 * c_in + ci];
                }
                if (co_vec < c_out) {
                    int out0 = sh_out[kk0];
                    int out1 = sh_out[kk1];
                    if (out0 >= 0) vb[0] = out_bp[out0 * c_out + co_vec];
                    if (out1 >= 0) vb[1] = out_bp[out1 * c_out + co_vec];
                }
                acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            }
        } else {
            for (int step = 0; step < valid_steps; ++step) {
                int base = step * 8;
                v2f va = {0.0f, 0.0f};
                v2f vb = {0.0f, 0.0f};
                int kk0 = col_group * 2 + base;
                int kk1 = col_group * 2 + base + 1;
                if (ci < c_in) {
                    int inp0 = sh_inp[kk0];
                    int inp1 = sh_inp[kk1];
                    if (inp0 >= 0) va[0] = features[inp0 * c_in + ci];
                    if (inp1 >= 0) va[1] = features[inp1 * c_in + ci];
                }
                if (co_vec < c_out) {
                    int out0 = sh_out[kk0];
                    int out1 = sh_out[kk1];
                    if (out0 >= 0) vb[0] = out_bp[out0 * c_out + co_vec];
                    if (out1 >= 0) vb[1] = out_bp[out1 * c_out + co_vec];
                }
                acc = __builtin_amdgcn_mmac_f32_16x16x8f32(va, vb, acc);
            }
        }
    }

    if (ci < c_in) {
        if (co0 < c_out) atomicAdd(dfilters + (k * c_in + ci) * c_out + co0, acc[0]);
        if (co1 < c_out) atomicAdd(dfilters + (k * c_in + ci) * c_out + co1, acc[1]);
        if (co2 < c_out) atomicAdd(dfilters + (k * c_in + ci) * c_out + co2, acc[2]);
        if (co3 < c_out) atomicAdd(dfilters + (k * c_in + ci) * c_out + co3, acc[3]);
    }
#else
    (void)features;
    (void)out_bp;
    (void)indice_pairs;
    (void)indice_num_per_loc;
    (void)dfilters;
    (void)kv;
    (void)c_in;
    (void)c_out;
    (void)nmax;
    (void)gather_axis;
    (void)scatter_axis;
    (void)skip_center;
    (void)center;
    (void)split_count;
#endif
}

// ============================================================================
// Host-side orchestration (called from PyTorch C++ extension)
// ============================================================================

// SubM convolution: build hash table + compute indice pairs in 2 kernel launches.
template <typename K>
void generate_subm_conv_inds(
    const int* indices_in,   // [N, ndim+1] on device
    int* indice_pairs,       // [2, kv, N] on device
    int* indice_num_per_loc, // [kv] on device
    int num_act_in,
    int batch_size,
    const int* input_dims_h,  // host
    const int* ksize_h,       // host
    const int* dilation_h,    // host
    int ndim,
    hipStream_t stream)
{
    int kv = 1;
    for (int d = 0; d < ndim; d++) kv *= ksize_h[d];

    ConvProblem problem{};
    problem.batch_size = batch_size;
    problem.ndim = ndim;
    for (int d = 0; d < ndim; d++) {
        problem.input_dims[d] = input_dims_h[d];
        problem.output_dims[d] = input_dims_h[d];
        problem.ksize[d] = ksize_h[d];
        problem.stride[d] = 1;
        problem.padding[d] = 0;
        problem.dilation[d] = dilation_h[d];
    }

    int hash_size = num_act_in * 2;
    if (hash_size < 128) hash_size = 128;
    auto& workspace = get_raw_workspace();

    K* hash_keys = ensure_raw<K>(workspace.subm_hash_keys, hash_size);
    int* hash_vals = ensure_raw<int>(workspace.subm_hash_vals, hash_size);

    LinearHashTable<K, int> table(hash_keys, hash_vals, hash_size);

    // Clear hash table.
    int threads = 256;
    int blocks_clear = (hash_size + threads - 1) / threads;
    hipLaunchKernelGGL(clear_hash_table_kernel<K>, dim3(blocks_clear), dim3(threads), 0, stream,
                       hash_keys, hash_size);

    // Build hash table from input coordinates.
    int blocks_build = (num_act_in + threads - 1) / threads;
    hipLaunchKernelGGL(build_subm_hash_table_kernel<K>, dim3(blocks_build), dim3(threads), 0, stream,
                       table, indices_in, problem, num_act_in);

    // Compute indice pairs: 2D grid with blockIdx.y = filter_offset.
    // Only process half the kernel + center (kv/2 + 1 slices), mirror fills the rest.
    int blocks_x = (num_act_in + threads - 1) / threads;
    dim3 grid(blocks_x, kv / 2 + 1);
    hipLaunchKernelGGL(calc_subm_conv_indices_kernel<K>, grid, dim3(threads), 0, stream,
                       table, indices_in, indice_pairs, indice_num_per_loc,
                       problem, num_act_in, num_act_in, kv);

    int mirror_blocks = (kv / 2 + threads - 1) / threads;
    hipLaunchKernelGGL(mirror_indice_num_per_loc_kernel,
                       dim3(mirror_blocks), dim3(threads), 0, stream,
                       indice_num_per_loc, kv);

}

// Regular convolution: 3-stage pipeline.
// Stage 1: compute output coords per input × filter_offset.
// Stage 1.5: sort + unique to get unique output points.
// Stage 2: hash table lookup to assign sequential output indices.
// Returns: number of unique output points.
template <typename K>
int generate_conv_inds(
    const int* indices_in,   // [N, ndim+1] on device
    int* indice_pairs,       // [2, kv, N] on device — first half is input indices
    int* indice_num_per_loc, // [kv] on device
    int* out_indices,        // [max_out, ndim+1] on device
    int num_act_in,
    int batch_size,
    const int* input_dims_h,
    const int* output_dims_h,
    const int* ksize_h,
    const int* stride_h,
    const int* padding_h,
    const int* dilation_h,
    int ndim,
    bool transposed,
    hipStream_t stream)
{
    int kv = 1;
    for (int d = 0; d < ndim; d++) kv *= ksize_h[d];

    ConvProblem problem{};
    problem.batch_size = batch_size;
    problem.ndim = ndim;
    for (int d = 0; d < ndim; d++) {
        problem.input_dims[d] = input_dims_h[d];
        problem.output_dims[d] = output_dims_h[d];
        problem.ksize[d] = ksize_h[d];
        problem.stride[d] = stride_h[d];
        problem.padding[d] = padding_h[d];
        problem.dilation[d] = dilation_h[d];
    }

    int64_t uniq_size = (int64_t)num_act_in * kv + 1;
    auto& workspace = get_raw_workspace();

    K* indice_pairs_uniq = ensure_raw<K>(workspace.indice_pairs_uniq, uniq_size);
    K* indice_pairs_uniq_copy = ensure_raw<K>(workspace.indice_pairs_uniq_copy, uniq_size);

    int threads = 256;

    // Clean uniq buffer.
    int blocks_clean = ((int)uniq_size + threads - 1) / threads;
    hipLaunchKernelGGL(clean_indices_uniq_kernel<K>, dim3(blocks_clean), dim3(threads), 0, stream,
                       indice_pairs_uniq, uniq_size);

    // Stage 1: compute output coord offsets.
    int blocks_x = (num_act_in + threads - 1) / threads;
    dim3 grid_s1(blocks_x, kv);
    hipLaunchKernelGGL(calc_conv_indices_stage1_kernel<K>, grid_s1, dim3(threads), 0, stream,
                       indices_in, indice_pairs,  // indice_pairs[0] part
                       indice_pairs_uniq, indice_num_per_loc,
                       problem, num_act_in, num_act_in, kv, transposed);

    // Copy uniq before sort (stage 2 needs original order for each filter position).
    hipMemcpyAsync(indice_pairs_uniq_copy, indice_pairs_uniq,
                   uniq_size * sizeof(K), hipMemcpyDeviceToDevice, stream);

    // Stage 1.5: sort + unique via hipCUB.
    size_t sort_temp_bytes = 0;
    hipcub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes,
                                       indice_pairs_uniq, indice_pairs_uniq,
                                       (int)uniq_size, 0, sizeof(K) * 8, stream);
    void* sort_temp = ensure_raw_bytes(workspace.sort_temp, sort_temp_bytes);

    // Sort in-place (double-buffer: input=output uses temp internally).
    K* sorted_keys = ensure_raw<K>(workspace.sorted_keys, uniq_size);
    hipcub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_bytes,
                                       indice_pairs_uniq, sorted_keys,
                                       (int)uniq_size, 0, sizeof(K) * 8, stream);

    // Unique.
    int* d_num_selected = ensure_raw<int>(workspace.d_num_selected, 1);
    size_t unique_temp_bytes = 0;
    K* unique_out = ensure_raw<K>(workspace.unique_out, uniq_size);
    hipcub::DeviceSelect::Unique(nullptr, unique_temp_bytes,
                                  sorted_keys, unique_out,
                                  d_num_selected, (int)uniq_size, stream);
    void* unique_temp = ensure_raw_bytes(workspace.unique_temp, unique_temp_bytes);
    hipcub::DeviceSelect::Unique(unique_temp, unique_temp_bytes,
                                  sorted_keys, unique_out,
                                  d_num_selected, (int)uniq_size, stream);

    int* h_num_selected = ensure_host<int>(workspace.h_num_selected, 1);
    hipMemcpyAsync(h_num_selected, d_num_selected, sizeof(int), hipMemcpyDeviceToHost, stream);
    hipStreamSynchronize(stream);
    int num_out_act = h_num_selected[0];
    // The last unique element is the sentinel (max value), subtract 1.
    num_out_act -= 1;

    // Stage 2: build hash table from unique output offsets, then lookup.
    int hash_size = num_out_act * 2;
    if (hash_size < 128) hash_size = 128;
    K* hash_keys = ensure_raw<K>(workspace.conv_hash_keys, hash_size);
    int* hash_vals = ensure_raw<int>(workspace.conv_hash_vals, hash_size);

    LinearHashTable<K, int> table(hash_keys, hash_vals, hash_size);

    int blocks_ht = (hash_size + threads - 1) / threads;
    hipLaunchKernelGGL(clear_hash_table_kernel<K>, dim3(blocks_ht), dim3(threads), 0, stream,
                       hash_keys, hash_size);

    int blocks_out = (num_out_act + threads - 1) / threads;
    hipLaunchKernelGGL(build_conv_hash_and_assign_kernel<K>, dim3(blocks_out), dim3(threads), 0, stream,
                       table, out_indices, unique_out, problem, num_out_act);

    // Stage 2 lookup: map each (filter, input) pair's output coord to sequential index.
    dim3 grid_s2(blocks_x, kv);
    hipLaunchKernelGGL(calc_conv_indices_stage2_kernel<K>, grid_s2, dim3(threads), 0, stream,
                       table, indice_pairs_uniq_copy,
                       indice_pairs + kv * num_act_in,  // indice_pairs[1] part
                       num_act_in, num_act_in);

    return num_out_act;
}

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
    hipStream_t stream)
{
    int threads = 256;
    hipMemsetAsync(lut, 0xFF, (size_t)num_out * kv * sizeof(int), stream);
    hipMemsetAsync(dup_count_dev, 0, sizeof(int), stream);
    int blocks_x = (nmax + threads - 1) / threads;
    int center = kv / 2;
    hipLaunchKernelGGL(build_indice_conv_lut_kernel<IndexT>,
                       dim3(blocks_x, kv), dim3(threads), 0, stream,
                       indice_pairs, indice_num_per_loc, lut,
                       dup_slots, dup_inp, dup_count_dev, dup_capacity,
                       kv, nmax, gather_axis, scatter_axis, center, skip_center);
}

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
    hipStream_t stream)
{
    int threads = 256;
    hipMemsetAsync(dup_count_dev, 0, sizeof(int), stream);
    int blocks_x = (nmax + threads - 1) / threads;
    int center = kv / 2;
    hipLaunchKernelGGL(build_indice_conv_lut_epoch_kernel<IndexT>,
                       dim3(blocks_x, kv), dim3(threads), 0, stream,
                       indice_pairs, indice_num_per_loc,
                       reinterpret_cast<unsigned long long*>(lut_state),
                       dup_slots, dup_inp, dup_count_dev, dup_capacity,
                       kv, nmax, gather_axis, scatter_axis, center, skip_center,
                       lut_epoch);
}

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
    hipStream_t stream)
{
    int threads = 256;
    int max_blocks = (dup_capacity * c_out + threads - 1) / threads;
    int blocks = std::max(1, std::min(max_blocks, 4096));
    hipLaunchKernelGGL(indice_conv_accumulate_duplicates_f32_kernel,
                       dim3(blocks), dim3(threads), 0, stream,
                       features, filters, dup_slots, dup_inp,
                       dup_count_dev, out, kv, c_in, c_out, filters_transposed);
}

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
    hipStream_t stream)
{
    int threads = 256;
    int blocks = (total_nhot + threads - 1) / threads;
    hipLaunchKernelGGL((flatten_indice_pairs_rows_kernel<IndexT, FlatIndexT>),
                       dim3(blocks), dim3(threads), 0, stream,
                       indice_pairs, window_ks, seg_offsets, num_segs, nmax,
                       gather_axis, scatter_axis, flat_inp, flat_out, total_nhot);
}

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
    hipStream_t stream)
{
    bool use_mmac = false;
    int mmac_waves = c_out >= 64 ? 4 : (c_out >= 32 ? 2 : 1);
    if (c_in % 8 == 0) {
        thread_local int cached_device = -2;
        thread_local bool cached_use_mmac = false;
        int device = -1;
        if (hipGetDevice(&device) == hipSuccess) {
            if (device != cached_device) {
                hipDeviceProp_t props{};
                cached_use_mmac = hipGetDeviceProperties(&props, device) == hipSuccess &&
                                  (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 ||
                                   std::strncmp(props.gcnArchName, "gfx938", 6) == 0);
                cached_device = device;
            }
            use_mmac = cached_use_mmac;
        }
    }
    if (use_mmac) {
        if (const char* env = std::getenv("SPCONV_FUSED_FORWARD_MMAC_WAVES")) {
            int parsed = std::atoi(env);
            if (parsed > 0) {
                mmac_waves = parsed;
            }
        }
        if (mmac_waves >= 2) {
            mmac_waves = mmac_waves >= 4 ? 4 : 2;
            int tile_cout = mmac_waves * 16;
            dim3 grid((num_out + 15) / 16, (c_out + tile_cout - 1) / tile_cout);
            bool use_c64 = c_in == 64 && !filters_transposed;
            if (const char* env = std::getenv("SPCONV_FUSED_FORWARD_C64")) {
                use_c64 = use_c64 && std::atoi(env) != 0;
            }
            bool use_static_c = !filters_transposed && (c_in == 16 || c_in == 32);
            if (const char* env = std::getenv("SPCONV_FUSED_FORWARD_STATIC_C")) {
                use_static_c = use_static_c && std::atoi(env) != 0;
            }
            if (use_c64) {
                bool use_c64x64 = c_out == 64 && mmac_waves == 4;
                if (const char* env = std::getenv("SPCONV_FUSED_FORWARD_C64X64")) {
                    use_c64x64 = use_c64x64 && std::atoi(env) != 0;
                }
                if (use_c64x64) {
                    hipLaunchKernelGGL(indice_conv_fused_forward_f32_mmac_multiwave_c64x64_kernel,
                                       grid, dim3(mmac_waves * 64), 0, stream,
                                       features, filters, lut,
                                       reinterpret_cast<const unsigned long long*>(lut_state),
                                       out, num_out, kv, lut_epoch,
                                       accumulate_existing);
                } else {
                    hipLaunchKernelGGL(indice_conv_fused_forward_f32_mmac_multiwave_c64_kernel,
                                       grid, dim3(mmac_waves * 64), 0, stream,
                                       features, filters, lut,
                                       reinterpret_cast<const unsigned long long*>(lut_state),
                                       out, num_out, kv, c_out, lut_epoch,
                                       accumulate_existing);
                }
            } else if (use_static_c && c_in == 32) {
                hipLaunchKernelGGL((indice_conv_fused_forward_f32_mmac_multiwave_static_c_kernel<32>),
                                   grid, dim3(mmac_waves * 64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, c_out, lut_epoch,
                                   accumulate_existing);
            } else if (use_static_c && c_in == 16) {
                hipLaunchKernelGGL((indice_conv_fused_forward_f32_mmac_multiwave_static_c_kernel<16>),
                                   grid, dim3(mmac_waves * 64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, c_out, lut_epoch,
                                   accumulate_existing);
            } else {
                hipLaunchKernelGGL(indice_conv_fused_forward_f32_mmac_multiwave_kernel,
                                   grid, dim3(mmac_waves * 64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, c_in, c_out, lut_epoch,
                                   accumulate_existing,
                                   filters_transposed);
            }
        } else {
            dim3 grid((num_out + 15) / 16, (c_out + 15) / 16);
            bool use_single_cc = !filters_transposed &&
                                 ((c_in == 64 && c_out == 16) ||
                                  (c_in == 16 && c_out == 16));
            if (const char* env = std::getenv("SPCONV_FUSED_FORWARD_SINGLE_CC")) {
                use_single_cc = use_single_cc && std::atoi(env) != 0;
            } else {
                use_single_cc = false;
            }
            if (use_single_cc && c_in == 64 && c_out == 16) {
                hipLaunchKernelGGL((indice_conv_fused_forward_f32_mmac_static_single_cc_kernel<64, 16>),
                                   grid, dim3(64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, lut_epoch,
                                   accumulate_existing);
            } else if (use_single_cc && c_in == 16 && c_out == 16) {
                hipLaunchKernelGGL((indice_conv_fused_forward_f32_mmac_static_single_cc_kernel<16, 16>),
                                   grid, dim3(64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, lut_epoch,
                                   accumulate_existing);
            } else {
                hipLaunchKernelGGL(indice_conv_fused_forward_f32_mmac_kernel,
                                   grid, dim3(64), 0, stream,
                                   features, filters, lut,
                                   reinterpret_cast<const unsigned long long*>(lut_state),
                                   out, num_out, kv, c_in, c_out, lut_epoch,
                                   accumulate_existing,
                                   filters_transposed);
            }
        }
    } else {
        constexpr int kRowTile = 8;
        constexpr int kCoutChunk = 32;
        constexpr int kThreads = kRowTile * kCoutChunk;
        dim3 grid((num_out + kRowTile - 1) / kRowTile,
                  (c_out + kCoutChunk - 1) / kCoutChunk);
        hipLaunchKernelGGL(indice_conv_fused_forward_f32_kernel,
                           grid, dim3(kThreads), 0, stream,
                           features, filters, lut,
                           reinterpret_cast<const unsigned long long*>(lut_state),
                           out, num_out, kv, c_in, c_out, lut_epoch,
                           accumulate_existing,
                           filters_transposed);
    }
}

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
    hipStream_t stream)
{
    bool use_mmac = false;
    int mmac_waves = 1;
    int device = -1;
    if (hipGetDevice(&device) == hipSuccess) {
        thread_local int cached_device = -2;
        thread_local bool cached_use_mmac = false;
        if (device != cached_device) {
            hipDeviceProp_t props{};
            cached_use_mmac = hipGetDeviceProperties(&props, device) == hipSuccess &&
                              (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 ||
                               std::strncmp(props.gcnArchName, "gfx938", 6) == 0);
            cached_device = device;
        }
        use_mmac = cached_use_mmac;
    }
    if (const char* env = std::getenv("SPCONV_DFILTERS_USE_MMAC")) {
        use_mmac = std::atoi(env) != 0;
    }
    if (use_mmac) {
        if (const char* env = std::getenv("SPCONV_DFILTERS_MMAC_WAVES")) {
            int parsed = std::atoi(env);
            if (parsed > 0) {
                mmac_waves = parsed;
            }
        }
        int split_count = nmax >= 512 ? 12 : 1;
        if (const char* env = std::getenv("SPCONV_DFILTERS_SPLIT_NHOT")) {
            int parsed = std::atoi(env);
            split_count = parsed > 1 ? (parsed > 16 ? 16 : parsed) : 1;
        }
        if (split_count > 1) {
            hipMemsetAsync(dfilters, 0, (size_t)kv * c_in * c_out * sizeof(float), stream);
            dim3 grid((c_out + 15) / 16, (c_in + 15) / 16, kv * split_count);
            hipLaunchKernelGGL((indice_conv_dfilters_f32_mmac_split_kernel<IndexT>),
                               grid, dim3(64), 0, stream,
                               features, out_bp, indice_pairs, indice_num_per_loc,
                               dfilters, kv, c_in, c_out, nmax, gather_axis, scatter_axis,
                               false, 0, split_count);
        } else if (mmac_waves >= 2) {
            mmac_waves = mmac_waves >= 4 ? 4 : 2;
            int tile_cout = mmac_waves * 16;
            dim3 grid((c_out + tile_cout - 1) / tile_cout, (c_in + 15) / 16, kv);
            hipLaunchKernelGGL((indice_conv_dfilters_f32_mmac_multiwave_kernel<IndexT>),
                               grid, dim3(mmac_waves * 64), 0, stream,
                               features, out_bp, indice_pairs, indice_num_per_loc,
                               dfilters, kv, c_in, c_out, nmax, gather_axis, scatter_axis);
        } else {
            dim3 grid((c_out + 15) / 16, (c_in + 15) / 16, kv);
            hipLaunchKernelGGL((indice_conv_dfilters_f32_mmac_kernel<IndexT>),
                               grid, dim3(64), 0, stream,
                               features, out_bp, indice_pairs, indice_num_per_loc,
                               dfilters, kv, c_in, c_out, nmax, gather_axis, scatter_axis);
        }
    } else {
        dim3 grid((c_out + 15) / 16, (c_in + 15) / 16, kv);
        dim3 block(16, 16);
        hipLaunchKernelGGL((indice_conv_dfilters_f32_kernel<IndexT>),
                           grid, block, 0, stream,
                           features, out_bp, indice_pairs, indice_num_per_loc,
                           dfilters, kv, c_in, c_out, nmax, gather_axis, scatter_axis);
    }
}

// Explicit instantiations for int32 and int64 key types.
template void generate_subm_conv_inds<int32_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);
template void generate_subm_conv_inds<int64_t>(
    const int*, int*, int*, int, int, const int*, const int*, const int*, int, hipStream_t);

template int generate_conv_inds<int32_t>(
    const int*, int*, int*, int*, int, int,
    const int*, const int*, const int*, const int*, const int*, const int*,
    int, bool, hipStream_t);
template int generate_conv_inds<int64_t>(
    const int*, int*, int*, int*, int, int,
    const int*, const int*, const int*, const int*, const int*, const int*,
    int, bool, hipStream_t);

template void build_indice_conv_lut<int32_t>(
    const int*, const int*, int*, int*, int*, int*, int, int, int, int, int, int, bool, hipStream_t);
template void build_indice_conv_lut<int64_t>(
    const int64_t*, const int*, int*, int*, int*, int*, int, int, int, int, int, int, bool, hipStream_t);

template void build_indice_conv_lut_epoch<int32_t>(
    const int*, const int*, int64_t*, int*, int*, int*,
    int, int, int, int, int, bool, unsigned int, hipStream_t);
template void build_indice_conv_lut_epoch<int64_t>(
    const int64_t*, const int*, int64_t*, int*, int*, int*,
    int, int, int, int, int, bool, unsigned int, hipStream_t);

template void launch_indice_conv_dfilters_f32<int32_t>(
    const float*, const float*, const int*, const int*, float*,
    int, int, int, int, int, int, hipStream_t);
template void launch_indice_conv_dfilters_f32<int64_t>(
    const float*, const float*, const int64_t*, const int*, float*,
    int, int, int, int, int, int, hipStream_t);
template void launch_flatten_indice_pairs_rows<int32_t, int32_t>(
    const int*, const int*, const int*, int, int, int, int,
    int*, int*, int, hipStream_t);
template void launch_flatten_indice_pairs_rows<int64_t, int64_t>(
    const int64_t*, const int*, const int*, int, int, int, int,
    int64_t*, int64_t*, int, hipStream_t);

}  // namespace spconv_hip
