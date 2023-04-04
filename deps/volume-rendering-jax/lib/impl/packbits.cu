#include "volrend.h"
#include "../serde.h"

namespace volrendjax {

namespace {

__global__ void pack_bits_kernel(
    // inputs
    /// static
    std::uint32_t n_bytes
    , float const density_threshold

    /// array
    , float const * __restrict__ density_grid

    // output
    , std::uint8_t * const __restrict__ occupancy_bitfield
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bytes) { return; }

    density_grid += i * 8;

    std::uint8_t byte = (std::uint8_t)0x00;

    #pragma unroll
    for (std::uint8_t idx = 0; idx < 8; ++idx) {
        byte |= (density_grid[idx] > density_threshold) ? ((std::uint8_t)0x01 << idx) : (std::uint8_t)0x00;
    }

    occupancy_bitfield[i] = byte;
}

void pack_bits_launcher(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    PackbitsDescriptor const &desc = *deserialize<PackbitsDescriptor>(opaque, opaque_len);
    std::uint32_t const n_bytes = desc.n_bytes;
    float const density_threshold = desc.density_threshold;

    /// array
    float const * __restrict__ density_grid = static_cast<float *>(next_buffer());

    // output
    std::uint8_t * const __restrict__ occupancy_bitfield = static_cast<std::uint8_t *>(next_buffer());

    // kernel launch
    int blockSize = 256;
    int numBlocks = (n_bytes + blockSize - 1) / blockSize;
    pack_bits_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // inputs
        /// static
        n_bytes
        , density_threshold

        /// array
        , density_grid

        /// output
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
