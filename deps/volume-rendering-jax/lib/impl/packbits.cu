#include <serde-helper/serde.h>

#include "volrend.h"

namespace volrendjax {

namespace {

__global__ void pack_bits_kernel(
    // inputs
    /// static
    std::uint32_t const n_bytes

    /// array
    , float const * const __restrict__ density_threshold
    , float const * const __restrict__ density_grid

    // output
    , bool * const __restrict__ occupied_mask
    , std::uint8_t * const __restrict__ occupancy_bitfield
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bytes) { return; }

    std::uint8_t byte = (std::uint8_t)0x00;

    #pragma unroll
    for (std::uint8_t idx = 0; idx < 8; ++idx) {
        bool const predicate = (density_grid[i*8+idx] > density_threshold[i*8+idx]);
        occupied_mask[i*8+idx] = predicate;
        byte |= predicate ? ((std::uint8_t)0x01 << idx) : (std::uint8_t)0x00;
    }
    occupancy_bitfield[i] = byte;
}

void pack_bits_launcher(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    PackbitsDescriptor const &desc = *deserialize<PackbitsDescriptor>(opaque, opaque_len);

    /// array
    float const * const __restrict__ density_threshold = static_cast<float *>(next_buffer());
    float const * const __restrict__ density_grid = static_cast<float *>(next_buffer());

    // output
    bool * const __restrict__ occupied_mask = static_cast<bool *>(next_buffer());
    std::uint8_t * const __restrict__ occupancy_bitfield = static_cast<std::uint8_t *>(next_buffer());

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (desc.n_bytes + blockSize - 1) / blockSize;
    pack_bits_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // inputs
        /// static
        desc.n_bytes

        /// array
        , density_threshold
        , density_grid

        /// output
        , occupied_mask
        , occupancy_bitfield
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

}

void pack_density_into_bits(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    pack_bits_launcher(stream, buffers, opaque, opaque_len);
}

}
