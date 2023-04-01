#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace volrendjax {

struct RayIntegratingDescriptor {
    // number of input rays
    std::uint32_t n_rays;

    // sum of number of samples of each ray
    std::uint32_t total_samples;
};

// functions to register
void integrate_rays(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

void integrate_rays_backward(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

}  // namespace volrendjax
