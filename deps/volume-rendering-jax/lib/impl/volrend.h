#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CUDA_CHECK_THROW(x)                                                                        \
    do {                                                                                           \
        cudaError_t result = x;                                                                    \
        if (result != cudaSuccess)                                                                 \
            throw std::runtime_error(                                                              \
                std::string(FILE_LINE " " #x " failed with error ")                                \
                + cudaGetErrorString(result));                                                     \
    } while(0)

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace volrendjax {

// Static parameters passed to `integrate_rays` and `integrate_rays_backward`
struct IntegratingDescriptor {
    // number of input rays
    std::uint32_t n_rays;

    // sum of number of samples of each ray
    std::uint32_t total_samples;
};

// Static parameters passed to `march_rays`
struct MarchingDescriptor {
    // number of input rays
    std::uint32_t n_rays;

    // paper: we maintain a cascade of 𝐾 multiscale occupancy grids, where 𝐾 = 1 for all synthetic
    // NeRF scenes (single grid) and 𝐾 ∈ [1, 5] for larger real-world scenes (up to 5 grids,
    // depending on scene size)
    std::uint32_t K;

    // density grid resolution, the paper uses 128 for every cascade
    std::uint32_t G;

    // the size of the largest axis of the scene’s bounding box
    float bound_max;

    // next step size is calculated as:
    //      clamp(z_val[i] * stepsize_portion, sqrt3/1024.f, bound_max * sqrt3/1024.f)
    // where bound_max is the size of the largest axis of the scene’s bounding box, as mentioned
    // in Appendix E.1 of the NGP paper (the intercept theorem)
    float stepsize_portion;
};

// functions to register
void march_rays(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

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
