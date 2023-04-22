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
    std::uint32_t const n_rays

    // input arrays (7)
    , float const * const __restrict__ transmittance_threshold  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    , float const * const __restrict__ bgs  // [n_rays, 3]
    , float const * const __restrict__ dss  // [total_samples]
    , float const * const __restrict__ z_vals  // [total_samples]
    , float const * const __restrict__ densities  // [total_samples, 1]
    , float const * const __restrict__ rgbs  // [total_samples, 3]

    // helper
    , std::uint32_t * const __restrict__ counter  // [1]

    // output arrays (4)
    , float * const __restrict__ opacities  // [n_rays]
    , float * const __restrict__ final_rgbs  // [n_rays]
    , float * const __restrict__ depths  // [n_rays]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input
    std::uint32_t const start_idx = rays_sample_startidx[i];
    std::uint32_t const n_samples = rays_n_samples[i];

    float const * const __restrict__ ray_bgs = bgs + i * 3;  // [3]
    float const * const __restrict__ ray_dss = dss + start_idx;  // [n_samples]
    float const * const __restrict__ ray_z_vals = z_vals + start_idx;  // [n_samples]
    float const * const __restrict__ ray_densities = densities + start_idx;  // [n_samples]
    float const * const __restrict__ ray_rgbs = rgbs + start_idx * 3;  // [n_samples, 3]

    // front-to-back composition, with early stop
    std::uint32_t sample_idx = 0;
    float ray_opacity = 0.f;
    float ray_depth = 0.f;
    float ray_transmittance = 1.f;
    float r = 0.f, g = 0.f, b = 0.f;
    for (; ray_transmittance > transmittance_threshold[i] && sample_idx < n_samples; ++sample_idx) {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const alpha = 1.f - __expf(-ray_densities[sample_idx] * delta_t);

        float const weight = ray_transmittance * alpha;

        // set outputs
        /// accumulate opacity
        ray_opacity += weight;

        /// composite colors
        r += weight * ray_rgbs[sample_idx * 3 + 0];
        g += weight * ray_rgbs[sample_idx * 3 + 1];
        b += weight * ray_rgbs[sample_idx * 3 + 2];

        /// composite depth
        ray_depth += weight * z_val;

        // decay transmittance at last, reflects the probability of the ray not hitting this sample
        ray_transmittance *= 1.f - alpha;
    }

    // stop ray marching and **set the remaining contribution to zero** as soon as the transmittance
    // of the ray drops below a threshold
    bool const ray_reached_bg = ray_transmittance > transmittance_threshold[i];
    ray_transmittance = ray_reached_bg ? ray_transmittance : 0.f;

    // write to global memory at last
    opacities[i] = ray_reached_bg ? 1.f : ray_opacity;
    depths[i] = ray_depth;
    // NOTE: `ray_transmittance` equals to `1 - ray_opacity`
    final_rgbs[i*3+0] = r + ray_transmittance * ray_bgs[0];
    final_rgbs[i*3+1] = g + ray_transmittance * ray_bgs[1];
    final_rgbs[i*3+2] = b + ray_transmittance * ray_bgs[2];

    // `counter` stores effective batch size (`measured_batch_size` in NGP)
    __shared__ std::uint32_t kernel_counter;
    if (threadIdx.x == 0) { kernel_counter = 0; }
    __syncthreads();
    atomicAdd(&kernel_counter, sample_idx);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(counter, kernel_counter); }
}

__global__ void integrate_rays_backward_kernel(
    // static arguments
    std::uint32_t const n_rays

    // input arrays
    , float const * const __restrict__ transmittance_threshold
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    /// original inputs
    , float const * const __restrict__ bgs  // [n_rays, 3]
    , float const * const __restrict__ dss  // [total_samples]
    , float const * const __restrict__ z_vals  // [total_samples]
    , float const * const __restrict__ densities  // [total_samples]
    , float const * const __restrict__ rgbs  // [total_samples, 3]

    /// original outputs
    , float const * const __restrict__ opacities  // [n_rays]
    , float const * const __restrict__ final_rgbs  // [n_rays, 3]
    , float const * const __restrict__ depths  // [n_rays]

    /// gradient inputs
    , float const * const __restrict__ dL_dopacities  // [n_rays]
    , float const * const __restrict__ dL_dfinal_rgbs  // [n_rays, 3]
    , float const * const __restrict__ dL_ddepths  // [n_rays]

    // output arrays
    , float * const __restrict__ dL_dbgs  // [n_rays]
    , float * const __restrict__ dL_dz_vals  // [total_samples]
    , float * const __restrict__ dL_ddensities  // [total_samples]
    , float * const __restrict__ dL_drgbs  // [total_samples, 3]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input
    std::uint32_t const start_idx = rays_sample_startidx[i];
    std::uint32_t const n_samples = rays_n_samples[i];

    /// original inputs
    float const * const __restrict__ ray_bgs = bgs + i * 3;
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
    float * const __restrict__ ray_dL_dbgs = dL_dbgs + i;  // [3]
    float * const __restrict__ ray_dL_dz_vals = dL_dz_vals + start_idx;  // [n_samples]
    float * const __restrict__ ray_dL_ddensities = dL_ddensities + start_idx;  // [n_samples]
    float * const __restrict__ ray_dL_drgbs = dL_drgbs + start_idx * 3;  // [n_samples, 3]

    // gradients for background colors
    // NOTE: `ray_transmittance` equals to `1 - ray_opacity`
    ray_dL_dbgs[0] = (1.f - ray_opacity) * ray_dL_dfinal_rgb[0];
    ray_dL_dbgs[1] = (1.f - ray_opacity) * ray_dL_dfinal_rgb[1];
    ray_dL_dbgs[2] = (1.f - ray_opacity) * ray_dL_dfinal_rgb[2];

    // front-to-back composition, with early stop
    float transmittance = 1.f;
    float cur_rgb[3] = {0.f, 0.f, 0.f};
    float cur_depth = 0.f;
    for (std::uint32_t sample_idx = 0; transmittance > transmittance_threshold[i] && sample_idx < n_samples; ++sample_idx) {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const alpha = 1.f - __expf(-ray_densities[sample_idx] * delta_t);

        float const weight = transmittance * alpha;

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
            ///// NOTE:
            /////   although `ray_final_rgb` now includes both the rays composed color and
            /////   optionally a background color, gradients from final_rgbs to densities does not
            /////   change.
            ///// TODO:
            /////   write this up in a public post
            + ray_dL_dfinal_rgb[0] * (transmittance * ray_rgbs[sample_idx * 3 + 0] - (ray_final_rgb[0] - cur_rgb[0]))
            + ray_dL_dfinal_rgb[1] * (transmittance * ray_rgbs[sample_idx * 3 + 1] - (ray_final_rgb[1] - cur_rgb[1]))
            + ray_dL_dfinal_rgb[2] * (transmittance * ray_rgbs[sample_idx * 3 + 2] - (ray_final_rgb[2] - cur_rgb[2]))
            //// gradients from depth
            + ray_dL_ddepth * (transmittance * ray_z_vals[sample_idx] - (ray_depth - cur_depth))
            //// gradients from opacity
            + ray_dL_dopacity * (1.f - ray_opacity)
        );

        /// color gradients
        ray_dL_drgbs[sample_idx * 3 + 0] = weight * ray_dL_dfinal_rgb[0];
        ray_dL_drgbs[sample_idx * 3 + 1] = weight * ray_dL_dfinal_rgb[1];
        ray_dL_drgbs[sample_idx * 3 + 2] = weight * ray_dL_dfinal_rgb[2];
    }
}

__global__ void integrate_rays_inference_kernel(
    std::uint32_t const n_total_rays
    , std::uint32_t const n_rays
    , std::uint32_t const march_steps_cap

    , float const * const __restrict__ rays_bg  // [n_total_rays, 3]
    , float const * const __restrict__ rays_rgb  // [n_total_rays, 3]
    , float const * const __restrict__ rays_T  // [n_total_rays]
    , float const * const __restrict__ rays_depth  // [n_total_rays]

    , std::uint32_t const * const __restrict__ n_samples  // [n_rays]
    , std::uint32_t const * const __restrict__ indices  // [n_rays]
    , float const * const __restrict__ dss  // [n_rays, march_steps_cap]
    , float const * const __restrict__ z_vals  // [n_rays, march_steps_cap]
    , float const * const __restrict__ densities  // [n_rays, march_steps_cap]
    , float const * const __restrict__ rgbs  // [n_rays, march_steps_cap, 3]

    , std::uint32_t * const __restrict__ terminate_cnt
    , bool * const __restrict__ terminated
    , float * const __restrict__ rays_rgb_out
    , float * const __restrict__ rays_T_out
    , float * const __restrict__ rays_depth_out
) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    std::uint32_t const ray_n_samples = n_samples[i];
    std::uint32_t const ray_idx = indices[i];
    bool ray_terminated = false;

    if (ray_idx < n_total_rays) {
        float const * const __restrict__ ray_dss = dss + i * march_steps_cap;
        float const * const __restrict__ ray_z_vals = z_vals + i * march_steps_cap;
        float const * const __restrict__ ray_densities = densities + i * march_steps_cap;
        float const * const __restrict__ ray_rgbs = rgbs + i * march_steps_cap * 3;

        float ray_T = rays_T[ray_idx];
        float r = rays_rgb[ray_idx * 3 + 0];
        float g = rays_rgb[ray_idx * 3 + 1];
        float b = rays_rgb[ray_idx * 3 + 2];
        float ray_depth = rays_depth[ray_idx];
        for (std::uint32_t sample_idx = 0; ray_T > 1e-4 && sample_idx < ray_n_samples; ++sample_idx) {
            float const ds = ray_dss[sample_idx];
            float const density = ray_densities[sample_idx];
            float const alpha = 1.f - __expf(-density * ds);
            float const weight = ray_T * alpha;
            r += weight * ray_rgbs[sample_idx * 3 + 0];
            g += weight * ray_rgbs[sample_idx * 3 + 1];
            b += weight * ray_rgbs[sample_idx * 3 + 2];
            ray_depth += weight * ray_z_vals[sample_idx];;
            ray_T *= (1.f - alpha);
        }

        ray_terminated = (ray_n_samples < march_steps_cap || ray_T <= 1e-4);
        terminated[i] = ray_terminated;

        bool const ray_reached_bg = (ray_terminated && ray_T > 1e-4);
        rays_rgb_out[i*3+0] = r + (ray_reached_bg ? ray_T * rays_bg[ray_idx*3+0] : 0.f);
        rays_rgb_out[i*3+1] = g + (ray_reached_bg ? ray_T * rays_bg[ray_idx*3+1] : 0.f);
        rays_rgb_out[i*3+2] = b + (ray_reached_bg ? ray_T * rays_bg[ray_idx*3+2] : 0.f);
        rays_T_out[i] = ray_T;
        rays_depth_out[i] = ray_depth;
    }

    __shared__ std::uint32_t kernel_terminate_cnt;
    if (threadIdx.x == 0) { kernel_terminate_cnt = 0; }
    __syncthreads();
    if (ray_terminated) { atomicAdd(&kernel_terminate_cnt, 1u); }
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(terminate_cnt, kernel_terminate_cnt); }
    __syncthreads();
}

// kernel launchers
void integrate_rays_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    IntegratingDescriptor const &desc = *deserialize<IntegratingDescriptor>(opaque, opaque_len);
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const total_samples = desc.total_samples;
    /// arrays
    float const * const __restrict__ transmittance_threshold = static_cast<float *>(next_buffer());
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float const * const __restrict__ bgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ densities = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ rgbs = static_cast<float *>(next_buffer());  // [total_samples, 3]

    // helper counter for measured_batch_size
    std::uint32_t * const __restrict__ counter = static_cast<std::uint32_t *>(next_buffer());  // [1]

    // outputs
    float * const __restrict__ opacities = static_cast<float *>(next_buffer());  // [n_rays]
    float * const __restrict__ final_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float * const __restrict__ depths = static_cast<float *>(next_buffer());  // [n_rays]

    // reset all outputs to zero
    CUDA_CHECK_THROW(cudaMemsetAsync(counter, 0x00, sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(opacities, 0x00, n_rays * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(final_rgbs, 0x00, n_rays * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(depths, 0x00, n_rays * sizeof(float), stream));

    // kernel launch
    std::uint32_t const blockSize = 256;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays
        // input arrays (7)
        , transmittance_threshold
        , rays_sample_startidx
        , rays_n_samples
        , bgs
        , dss
        , z_vals
        , densities
        , rgbs
        // helper counter
        , counter
        // output arrays (3)
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
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    IntegratingDescriptor const &desc =
        *deserialize<IntegratingDescriptor>(opaque, opaque_len);
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const total_samples = desc.total_samples;

    /// arrays
    float const * const __restrict__ transmittance_threshold = static_cast<float *>(next_buffer());
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    //// original inputs
    float const * const __restrict__ bgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ densities = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ rgbs = static_cast<float *>(next_buffer());  // [total_samples, 3]
    //// original outputs
    float const * const __restrict__ opacities = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ final_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ depths = static_cast<float *>(next_buffer());  // [n_rays]
    //// gradient inputs
    float const * const __restrict__ dL_dopacities = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ dL_dfinal_rgbs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dL_ddepths = static_cast<float *>(next_buffer());  // [n_rays]

    // outputs
    float * const __restrict__ dL_dbgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float * const __restrict__ dL_dz_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float * const __restrict__ dL_ddensities = static_cast<float *>(next_buffer());  // [total_samples]
    float * const __restrict__ dL_drgbs = static_cast<float *>(next_buffer());  // [total_samples, 3]

    // reset all outputs to zeros
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dbgs, 0x00, n_rays * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dz_vals, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddensities, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_drgbs, 0x00, total_samples * 3 * sizeof(float), stream));

    // kernel launch
    std::uint32_t const blockSize = 256;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays

        // input arrays
        , transmittance_threshold
        , rays_sample_startidx
        , rays_n_samples
        /// original inputs
        , bgs
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
        , dL_dbgs
        , dL_dz_vals
        , dL_ddensities
        , dL_drgbs
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void integrate_rays_inference_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    IntegratingInferenceDescriptor const &desc =
        *deserialize<IntegratingInferenceDescriptor>(opaque, opaque_len);
    std::uint32_t const n_total_rays = desc.n_total_rays;
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const march_steps_cap = desc.march_steps_cap;

    /// arrays
    float const * const __restrict__ rays_bg = static_cast<float *>(next_buffer());  // [n_total_rays, 3]
    float const * const __restrict__ rays_rgb = static_cast<float *>(next_buffer());  // [n_total_rays, 3]
    float const * const __restrict__ rays_T = static_cast<float *>(next_buffer());  // [n_total_rays]
    float const * const __restrict__ rays_depth = static_cast<float *>(next_buffer());  // [n_total_rays]

    std::uint32_t const * const __restrict__ n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ indices = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float const * const __restrict__ densities = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float const * const __restrict__ rgbs = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap, 3]

    // outputs
    std::uint32_t * const __restrict__ terminate_cnt = static_cast<std::uint32_t *>(next_buffer());  // [1]
    bool * const __restrict__ terminated = static_cast<bool *>(next_buffer());  // [n_rays]
    float * const __restrict__ rays_rgb_out = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float * const __restrict__ rays_T_out = static_cast<float *>(next_buffer());  // [n_rays]
    float * const __restrict__ rays_depth_out = static_cast<float *>(next_buffer());  // [n_rays]

    CUDA_CHECK_THROW(cudaMemsetAsync(terminate_cnt, 0x00, sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(terminated, false, n_rays * sizeof(bool), stream));

    CUDA_CHECK_THROW(cudaMemsetAsync(rays_rgb_out, 0x00, n_rays * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_T_out, 0x00, n_rays * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_depth_out, 0x00, n_rays * sizeof(float), stream));

    // kernel launch
    std::uint32_t const blockSize = 256;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_inference_kernel<<<numBlocks, blockSize, 1 * sizeof(std::uint32_t), stream>>>(
        n_total_rays
        , n_rays
        , march_steps_cap

        , rays_bg
        , rays_rgb
        , rays_T
        , rays_depth

        , n_samples
        , indices
        , dss
        , z_vals
        , densities
        , rgbs

        , terminate_cnt
        , terminated
        , rays_rgb_out
        , rays_T_out
        , rays_depth_out
    );
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

void integrate_rays_inference(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    integrate_rays_inference_launcher(stream, buffers, opaque, opaque_len);
}

}  // namespace volrendjax
