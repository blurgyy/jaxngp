#include <cstdint>

#include <cuda_device_runtime_api.h>
#include <serde-helper/serde.h>
#include <stdexcept>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/gpu_memory.h>

#include "fmt/core.h"
#include "tcnnutils.h"


namespace jaxtcnn {

namespace {

void hashgrid_forward_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    std::uint32_t static constexpr DIM = 3u;

    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    HashGridDescriptor const &desc =
        *deserialize<HashGridDescriptor>(opaque, opaque_len);
    std::uint32_t const n_coords = desc.n_coords;
    std::uint32_t const L = desc.L;
    std::uint32_t const F = desc.F;
    std::uint32_t const N_min = desc.N_min;
    float const per_level_scale = desc.per_level_scale;

    // inputs
    std::uint32_t * const offset_table_data = static_cast<std::uint32_t *>(next_buffer());  // [L+1]
    float const * const __restrict__ coords_rm = static_cast<float *>(next_buffer());  // [dim, n_coords]
    float const * const __restrict__ params = static_cast<float *>(next_buffer());  // [n_params, F]

    // outputs
    float * const __restrict__ encoded_positions_rm = static_cast<float *>(next_buffer());  // [L*F, n_coords]
    float * const __restrict__ dy_dcoords_rm = static_cast<float *>(next_buffer());  // [dim*L*F, n_coords]

    // prepare input data for tcnn:kernel_grid
    tcnn::GridOffsetTable offset_table{{}, L + 1};
    CUDA_CHECK_THROW(cudaMemcpyAsync(offset_table.data, offset_table_data, (L + 1) * sizeof(std::uint32_t),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dy_dcoords_rm, 0x00, DIM * L * F * n_coords * sizeof(float), stream));
    // tcnn::GPUMatrixDynamic<float const> positions_in(coords, n_points, DIM);
    tcnn::MatrixView<float const> positions_in(coords_rm, n_coords, 1);  // row major

    // kernel launch
    std::uint32_t static constexpr n_threads = 512;
    dim3 const blocks = { tcnn::div_round_up(n_coords, n_threads), L, 1 };
#define PARAMS \
    n_coords \
    , L * F \
    , offset_table \
    , N_min \
    , log2f(per_level_scale) \
    , 0.f \
    , 1e3f \
    , nullptr \
    , tcnn::InterpolationType::Linear \
    , tcnn::GridType::Hash \
    , params \
    , positions_in \
    , encoded_positions_rm \
    , dy_dcoords_rm

    if (F == 2) {
        std::uint32_t static constexpr N_FEATURES_PER_LEVEL = 2;
        tcnn::kernel_grid<float, 3, N_FEATURES_PER_LEVEL, tcnn::HashType::CoherentPrime><<<blocks, n_threads, 0, stream>>>(
            PARAMS
        );
    } else if (F == 4) {
        std::uint32_t static constexpr N_FEATURES_PER_LEVEL = 4;
        tcnn::kernel_grid<float, 3, N_FEATURES_PER_LEVEL, tcnn::HashType::CoherentPrime><<<blocks, n_threads, 0, stream>>>(
            PARAMS
        );
    } else {
        throw std::runtime_error{
            fmt::format("supported values of F (n_features_per_level) are [2, 4], got {}", F)
        };
    }

    CUDA_CHECK_THROW(cudaGetLastError());
}

void hashgrid_backward_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    HashGridDescriptor const &desc =
        *deserialize<HashGridDescriptor>(opaque, opaque_len);
    std::uint32_t const n_coords = desc.n_coords;
    std::uint32_t const L = desc.L;
    std::uint32_t const F = desc.F;
    std::uint32_t const N_min = desc.N_min;
    float const per_level_scale = desc.per_level_scale;

    // input
    std::uint32_t * const offset_table_data = static_cast<std::uint32_t *>(next_buffer());  // [L+1]
    float const * const __restrict__ coords_rm = static_cast<float *>(next_buffer());  // [dim, n_coords]
    float const * const __restrict__ dL_dy_rm = static_cast<float *>(next_buffer());  // [L*F, n_coords]
    float const * const __restrict__ dy_dcoords_rm = static_cast<float *>(next_buffer());  // [dim*L*F, n_coords]

    // output
    float * const __restrict__ dL_dparams = static_cast<float *>(next_buffer());  // [n_params, F]
    float * const __restrict__ dL_dcoords_rm = static_cast<float *>(next_buffer());  // [dim, n_coords]

    // prepare input data for tcnn::kernel_grid_backward
    tcnn::GridOffsetTable offset_table{{}, L + 1};
    CUDA_CHECK_THROW(cudaMemcpyAsync(offset_table.data, offset_table_data, (L + 1) * sizeof(std::uint32_t),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dparams, 0x00, offset_table.data[L] * F * sizeof(float), stream));
    tcnn::MatrixView<float const> positions_in(coords_rm, n_coords, 1);  // row major

    // kernel launch
    std::uint32_t static constexpr n_threads = 256;

#define PARAMS \
    n_coords \
    , L * F \
    , offset_table \
    , N_min \
    , log2f(per_level_scale) \
    , 1e3f \
    , nullptr \
    , false \
    , tcnn::InterpolationType::Linear \
    , tcnn::GridType::Hash \
    , dL_dparams \
    , positions_in \
    , dL_dy_rm

    std::uint32_t static constexpr DIM = 3u;
    if (F == 2u) {
        std::uint32_t static constexpr N_FEATURES_PER_LEVEL = 2u;
        std::uint32_t static constexpr n_features_per_thread = std::min(2u, N_FEATURES_PER_LEVEL);
        const dim3 blocks = { tcnn::div_round_up(n_coords * F / n_features_per_thread, n_threads), L, 1 };
        tcnn::kernel_grid_backward<float, float, DIM, N_FEATURES_PER_LEVEL, n_features_per_thread, tcnn::HashType::CoherentPrime><<<blocks, n_threads, 0, stream>>>(
            PARAMS
        );
    } else if (F == 4) {
        std::uint32_t static constexpr N_FEATURES_PER_LEVEL = 4u;
        std::uint32_t static constexpr n_features_per_thread = std::min(2u, N_FEATURES_PER_LEVEL);
        const dim3 blocks = { tcnn::div_round_up(n_coords * F / n_features_per_thread, n_threads), L, 1 };
        tcnn::kernel_grid_backward<float, float, DIM, N_FEATURES_PER_LEVEL, n_features_per_thread, tcnn::HashType::CoherentPrime><<<blocks, n_threads, 0, stream>>>(
            PARAMS
        );
    } else {
        throw std::runtime_error{
            fmt::format("supported values of F (n_features_per_level) are 2, 4, got {}", F)
        };
    }

    CUDA_CHECK_THROW(cudaGetLastError());

    // gradients w.r.t. input coordinates
    // prepare input data for tcnn::kernel_grid_backward_input
    tcnn::MatrixView<float> dL_dinput_view(dL_dcoords_rm, n_coords, 1);

    tcnn::linear_kernel(tcnn::kernel_grid_backward_input<float, DIM>, 0, stream,
        n_coords,
        L * F,
        dL_dy_rm,
        dy_dcoords_rm,
        dL_dinput_view
    );

    CUDA_CHECK_THROW(cudaGetLastError());
}

}

void hashgrid_encode(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    hashgrid_forward_launcher(stream, buffers, opaque, opaque_len);
}

void hashgrid_encode_backward(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    hashgrid_backward_launcher(stream, buffers, opaque, opaque_len);
}

}  // namespace jaxtcnn
