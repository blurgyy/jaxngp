#include <cstdint>

#include "volrend.h"
#include "../serde.h"


namespace volrendjax {

namespace {

// debugging kernel for inspecting data passed to custom op
__global__ void copy_left_to_right(std::uint32_t length, float const *lhs, float * const rhs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < length; i += stride) {
        rhs[i] = lhs[i];
    }
}

// kernel
__global__ void integrate_rays_kernel(
    // static arguments
    std::uint32_t n_rays

    // input arrays (7)
    , float const * const __restrict__ transmittance_threshold  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    , float const * const __restrict__ dss  // [total_samples]
    , float const * const __restrict__ z_vals  // [\sum rays_n_samples] = [total_samples]
    , float const * const __restrict__ densities  // [\sum rays_n_samples, 1] = [total_samples, 1]
    , float const * const __restrict__ rgbs  // [\sum rays_n_samples, 3] = [total_samples, 3]

    // output arrays (4)
    , std::uint32_t * const __restrict__ effective_samples  // [n_rays]
    , float * const __restrict__ opacities  // [n_rays]
    , float * const __restrict__ final_rgbs  // [n_rays]
    , float * const __restrict__ depths  // [n_rays]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input
    std::uint32_t start_idx = rays_sample_startidx[i];
    std::uint32_t n_samples = rays_n_samples[i];

    if (n_samples == 0) { return; }

    float const * const __restrict__ ray_dss = dss + start_idx;  // [n_samples]
    float const * const __restrict__ ray_z_vals = z_vals + start_idx;  // [n_samples]
    float const * const __restrict__ ray_densities = densities + start_idx;  // [n_samples]
    float const * const __restrict__ ray_rgbs = rgbs + start_idx * 3;  // [n_samples, 3]

    // output
    std::uint32_t * const __restrict__ ray_effective_samples = effective_samples + i;  // [1]
    float * const __restrict__ ray_opacity = opacities + i;  // [1]
    float * const __restrict__ ray_final_rgb = final_rgbs + i * 3;  // [3]
    float * const __restrict__ ray_depth = depths + i;  // [1]

    // front-to-back composition, with early stop
    std::uint32_t sample_idx = 0;
    float transmittance = 1.f;
    float r = 0.f, g = 0.f, b = 0.f;
    for (; transmittance > transmittance_threshold[i] && sample_idx < n_samples; ++sample_idx) {
        float z_val = ray_z_vals[sample_idx];
        float delta_t = ray_dss[sample_idx];
        float alpha = 1.f - __expf(-ray_densities[sample_idx] * delta_t);

        float weight = transmittance * alpha;

        // set outputs
        /// accumulate opacity
        *ray_opacity += weight;

        /// composite colors
        r += weight * ray_rgbs[sample_idx * 3 + 0];
        g += weight * ray_rgbs[sample_idx * 3 + 1];
        b += weight * ray_rgbs[sample_idx * 3 + 2];

        /// composite depth
        *ray_depth += weight * z_val;

        // decay transmittance at last, reflects the probability of the ray not hitting this sample
        transmittance *= 1.f - alpha;
    }

    // write to global memory at last
    ray_final_rgb[0] = r;
    ray_final_rgb[1] = g;
    ray_final_rgb[2] = b;
    *ray_effective_samples = sample_idx;
}

__global__ void integrate_rays_backward_kernel(
    // static arguments
    std::uint32_t n_rays

    // input arrays
    , float const * const __restrict__ transmittance_threshold
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    /// original inputs
    , float const * const __restrict__ dss  // [n_rays, ray's n_samples]
    , float const * const __restrict__ z_vals  // [n_rays, ray's n_samples]
    , float const * const __restrict__ densities  // [n_rays, ray's n_samples]
    , float const * const __restrict__ rgbs  // [n_rays, ray's n_samples, 3]

    // original outputs
    , float const * const __restrict__ opacities  // [n_rays]
    , float const * const __restrict__ final_rgbs  // [n_rays, 3]
    , float const * const __restrict__ depths  // [n_rays]

    /// gradient inputs
    , float const * const __restrict__ dL_dopacities  // [n_rays]
    , float const * const __restrict__ dL_dfinal_rgbs  // [n_rays, 3]
    , float const * const __restrict__ dL_ddepths  // [n_rays]

    // output arrays
    , float * const __restrict__ dL_dz_vals  // [n_rays, ray's n_samples]
    , float * const __restrict__ dL_ddensities  // [n_rays, ray's n_samples]
    , float * const __restrict__ dL_drgbs  // [n_rays, ray's n_samples, 3]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input
    std::uint32_t start_idx = rays_sample_startidx[i];
    std::uint32_t n_samples = rays_n_samples[i];

    if (n_samples == 0) { return; }

    /// original inputs
    float const * const __restrict__ ray_dss = dss + start_idx;  // [n_samples]
    float const * const __restrict__ ray_z_vals = z_vals + start_idx;  // [n_samples]
    float const * const __restrict__ ray_densities = densities + start_idx;  // [n_samples]
    float const * const __restrict__ ray_rgbs = rgbs + start_idx * 3;  // [n_samples, 3]

    /// original outputs
    float const ray_opacity = opacities[i];  // [] (a scalar has no shape)
    float const * const __restrict__ ray_final_rgb = final_rgbs + i * 3;  // [3]
    float const ray_depth = depths[i];  // [] (a scalar has no shape)

    /// gradient inputs
    float const ray_dL_dopacity = dL_dopacities[i];  // [] (a scalar has no shape)
    float const * const __restrict__ ray_dL_dfinal_rgb = dL_dfinal_rgbs + i * 3;  // [3]
    float const ray_dL_ddepth = dL_ddepths[i];  // [] (a scalar has no shape)

    // outputs
    float * const __restrict__ ray_dL_dz_vals = dL_dz_vals + start_idx;  // [n_samples]
    float * const __restrict__ ray_dL_ddensities = dL_ddensities + start_idx;  // [n_samples]
    float * const __restrict__ ray_dL_drgbs = dL_drgbs + start_idx * 3;  // [n_samples, 3]

    // front-to-back composition, with early stop
    float transmittance = 1.f;
    float cur_rgb[3] = {0.f, 0.f, 0.f};
    float cur_depth = 0.f;
    for (std::uint32_t sample_idx = 0; transmittance > transmittance_threshold[i] && sample_idx < n_samples; ++sample_idx) {
        float z_val = ray_z_vals[sample_idx];
        float delta_t = ray_dss[sample_idx];
        float alpha = 1.f - __expf(-ray_densities[sample_idx] * delta_t);

        float weight = transmittance * alpha;

        cur_rgb[0] += weight * ray_rgbs[sample_idx * 3 + 0];
        cur_rgb[1] += weight * ray_rgbs[sample_idx * 3 + 1];
        cur_rgb[2] += weight * ray_rgbs[sample_idx * 3 + 2];
        cur_depth += weight * ray_z_vals[sample_idx];

        // decay transmittance before gradient calculation, as transmittance used in gradient
        // calculation is T_{i+1}.  REF: <https://note.kiui.moe/others/nerf_gradient/>
        transmittance *= 1.f - alpha;

        // set outputs
        /// z_val gradients
        ray_dL_dz_vals[sample_idx] = weight * ray_dL_ddepth;

        /// density gradients
        ray_dL_ddensities[sample_idx] = delta_t * (
            0.f
            //// gradients from final_rgbs
            + ray_dL_dfinal_rgb[0] * (transmittance * ray_rgbs[sample_idx * 3 + 0] - (ray_final_rgb[0] - cur_rgb[0]))
            + ray_dL_dfinal_rgb[1] * (transmittance * ray_rgbs[sample_idx * 3 + 1] - (ray_final_rgb[1] - cur_rgb[1]))
            + ray_dL_dfinal_rgb[2] * (transmittance * ray_rgbs[sample_idx * 3 + 2] - (ray_final_rgb[2] - cur_rgb[2]))
            //// gradients from depth
            + ray_dL_ddepth * (transmittance * ray_z_vals[sample_idx] - (ray_depth - cur_depth))
            //// gradients from opacity
            + ray_dL_dopacity * (1 - ray_opacity)
        );

        /// color gradients
        ray_dL_drgbs[sample_idx * 3 + 0] = weight * ray_dL_dfinal_rgb[0];
        ray_dL_drgbs[sample_idx * 3 + 1] = weight * ray_dL_dfinal_rgb[1];
        ray_dL_drgbs[sample_idx * 3 + 2] = weight * ray_dL_dfinal_rgb[2];
    }
}

// kernel launchers
void integrate_rays_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    IntegratingDescriptor const &desc = *deserialize<IntegratingDescriptor>(opaque, opaque_len);
    std::uint32_t n_rays = desc.n_rays;
    std::uint32_t total_samples = desc.total_samples;
    /// arrays
    float const * const __restrict__ transmittance_threshold = static_cast<float *>(next_buffer());
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ densities = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ rgbs = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples, 3] = [total_samples, 3]

    // outputs
    std::uint32_t * const __restrict__ effective_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float * const __restrict__ opacities = static_cast<float *>(next_buffer());  // [n_rays]
    float * const __restrict__ final_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float * const __restrict__ depths = static_cast<float *>(next_buffer());  // [n_rays]

    // reset all outputs to zero
    cudaMemset(effective_samples, 0x00, n_rays * sizeof(std::uint32_t));
    cudaMemset(opacities, 0x00, n_rays * sizeof(float));
    cudaMemset(final_rgbs, 0x00, n_rays * 3 * sizeof(float));
    cudaMemset(depths, 0x00, n_rays * sizeof(float));

    // kernel launch
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays
        // input arrays (7)
        , transmittance_threshold
        , rays_sample_startidx
        , rays_n_samples
        , dss
        , z_vals
        , densities
        , rgbs
        // output arrays (4)
        , effective_samples
        , opacities
        , final_rgbs
        , depths
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void integrate_rays_backward_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    IntegratingDescriptor const &desc =
        *deserialize<IntegratingDescriptor>(opaque, opaque_len);
    std::uint32_t n_rays = desc.n_rays;
    std::uint32_t total_samples = desc.total_samples;

    /// arrays
    float const * const __restrict__ transmittance_threshold = static_cast<float *>(next_buffer());
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    //// original inputs
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ densities = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float const * const __restrict__ rgbs = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples, 3] = [total_samples, 3]
    //// original outputs
    float const * const __restrict__ opacities = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ final_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ depths = static_cast<float *>(next_buffer());  // [n_rays]
    //// gradient inputs
    float const * const __restrict__ dL_dopacities = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ dL_dfinal_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dL_ddepths = static_cast<float *>(next_buffer());  // [n_rays]

    // outputs
    float * const __restrict__ dL_dz_vals = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float * const __restrict__ dL_ddensities = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples] = [total_samples]
    float * const __restrict__ dL_drgbs = static_cast<float *>(next_buffer());  // [n_rays, ray's n_samples, 3] = [total_samples, 3]

    // reset all outputs to zero
    cudaMemset(dL_dz_vals, 0x00, total_samples * sizeof(float));
    cudaMemset(dL_ddensities, 0x00, total_samples * sizeof(float));
    cudaMemset(dL_drgbs, 0x00, total_samples * 3 * sizeof(float));

    // kernel launch
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays

        // input arrays
        , transmittance_threshold
        , rays_sample_startidx
        , rays_n_samples
        /// original inputs
        , dss
        , z_vals
        , densities
        , rgbs
        /// original outputs
        , opacities
        , final_rgbs
        , depths
        /// gradient inputs
        , dL_dopacities
        , dL_dfinal_rgbs
        , dL_ddepths

        // output arrays
        , dL_dz_vals
        , dL_ddensities
        , dL_drgbs
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

}  // namespace

// functions to register
void integrate_rays(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    integrate_rays_launcher(stream, buffers, opaque, opaque_len);
}

void integrate_rays_backward(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    integrate_rays_backward_launcher(stream, buffers, opaque, opaque_len);
}

}  // namespace volrendjax
