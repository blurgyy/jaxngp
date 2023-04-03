#include "volrend.h"
#include "../serde.h"

namespace volrendjax {

namespace {

// kernel
__global__ void march_rays_kernel() {
}

void march_rays_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    MarchingDescriptor const &desc =
        *deserialize<MarchingDescriptor>(opaque, opaque_len);
    std::uint32_t n_rays = desc.n_rays;
    std::uint32_t K = desc.K;
    std::uint32_t G = desc.G;
    float bound_max = desc.bound_max;
    float stepsize_portion = desc.stepsize_portion;

    /// arrays
    /* ... */

    // outputs
    float * const __restrict__ z_vals = static_cast<float *>(next_buffer());

    // kernel launch
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    march_rays_kernel<<<numBlocks, blockSize, 0, stream>>>();
}

}

void march_rays(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    march_rays_launcher(stream, buffers, opaque, opaque_len);
}

}
