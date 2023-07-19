#include <cstdint>

#include <serde-helper/serde.h>

#include "volrend.h"


namespace volrendjax {

namespace {

static constexpr float T_THRESHOLD = 1e-4f;

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
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    , float const * const __restrict__ bgs  // [n_rays, 3]
    , float const * const __restrict__ dss  // [total_samples]
    , float const * const __restrict__ z_vals  // [total_samples]
    , float const * const __restrict__ drgbs  // [total_samples, 4]

    // helper
    , std::uint32_t * const __restrict__ measured_batch_size  // [1]

    // output arrays (2)
    , float * const __restrict__ final_rgbds  // [n_rays, 4]
    , float * const __restrict__ final_opacities  // [n_rays]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input
    std::uint32_t const start_idx = rays_sample_startidx[i];
    std::uint32_t const n_samples = rays_n_samples[i];

    float const * const __restrict__ ray_bgs = bgs + i * 3;  // [3]
    float const * const __restrict__ ray_dss = dss + start_idx;  // [n_samples]
    float const * const __restrict__ ray_z_vals = z_vals + start_idx;  // [n_samples]
    float const * const __restrict__ ray_drgbs = drgbs + start_idx * 4;  // [n_samples, 4]

    // front-to-back composition, with early stop
    std::uint32_t sample_idx = 0;
    float ray_depth = 0.f;
    float ray_transmittance = 1.f;
    float r = 0.f, g = 0.f, b = 0.f;
    for (; ray_transmittance > T_THRESHOLD && sample_idx < n_samples; ++sample_idx) {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const alpha = 1.f - __expf(-ray_drgbs[sample_idx * 4] * delta_t);

        float const weight = ray_transmittance * alpha;

        /// composite colors
        r += weight * ray_drgbs[sample_idx * 4 + 1];
        g += weight * ray_drgbs[sample_idx * 4 + 2];
        b += weight * ray_drgbs[sample_idx * 4 + 3];

        /// composite depth
        ray_depth += weight * z_val;

        // decay transmittance at last, reflects the probability of the ray not hitting this sample
        ray_transmittance *= 1.f - alpha;
    }

    // write to global memory at last
    // stop ray marching and **set the remaining contribution to zero** as soon as the transmittance
    // of the ray drops below a threshold
    float const opacity = 1.f - ray_transmittance;
    final_opacities[i] = opacity;
    if (ray_transmittance <= T_THRESHOLD){ 
        float idenom = 1.f / opacity;
        final_rgbds[i*4+0] = r * idenom;
        final_rgbds[i*4+1] = g * idenom;
        final_rgbds[i*4+2] = b * idenom;
        final_rgbds[i*4+3] = ray_depth * idenom;
    } else {
        final_rgbds[i*4+0] = r + ray_transmittance * ray_bgs[0];
        final_rgbds[i*4+1] = g + ray_transmittance * ray_bgs[1];
        final_rgbds[i*4+2] = b + ray_transmittance * ray_bgs[2];
        final_rgbds[i*4+3] = ray_depth;
    }

    __shared__ std::uint32_t kernel_measured_batch_size;
    if (threadIdx.x == 0) { kernel_measured_batch_size = 0; }
    __syncthreads();
    atomicAdd(&kernel_measured_batch_size, sample_idx);
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(measured_batch_size, kernel_measured_batch_size); }
}

__global__ void integrate_rays_backward_kernel(
    // static arguments
    std::uint32_t const n_rays
    , float const near_distance

    // input arrays
    , std::uint32_t const * const __restrict__ rays_sample_startidx  // [n_rays]
    , std::uint32_t const * const __restrict__ rays_n_samples  // [n_rays]

    /// original inputs
    , float const * const __restrict__ bgs  // [n_rays, 3]
    , float const * const __restrict__ dss  // [total_samples]
    , float const * const __restrict__ z_vals  // [total_samples]
    , float const * const __restrict__ drgbs  // [total_samples, 4]

    /// original outputs
    , float const * const __restrict__ final_rgbds  // [n_rays, 4]
    , float const * const __restrict__ final_opacities  // [n_rays]

    /// gradient inputs
    , float const * const __restrict__ dL_dfinal_rgbds  // [n_rays, 4]
    /* background color blending is done inside the integrate_rays kernel, so no need to accept a
     * `dL_dfinal_opacities` parameter, it would be all zeros anyway
     **/

    // output arrays
    , float * const __restrict__ dL_dbgs  // [n_rays, 3]
    , float * const __restrict__ dL_dz_vals  // [total_samples]
    , float * const __restrict__ dL_ddrgbs  // [total_samples, 4]
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
    float const * const __restrict__ ray_drgbs = drgbs + start_idx * 4;  // [n_samples, 4]

    /// original outputs
    float const ray_final_rgbd[4] = {
        final_rgbds[i * 4 + 0],
        final_rgbds[i * 4 + 1],
        final_rgbds[i * 4 + 2],
        final_rgbds[i * 4 + 3],
    };
    float const ray_final_opacity = final_opacities[i];

    /// gradient inputs
    float const ray_dL_dfinal_rgbd[4] = {
        dL_dfinal_rgbds[i * 4 + 0],
        dL_dfinal_rgbds[i * 4 + 1],
        dL_dfinal_rgbds[i * 4 + 2],
        dL_dfinal_rgbds[i * 4 + 3],
    };

    // outputs
    float * const __restrict__ ray_dL_dbgs = dL_dbgs + i * 3;  // [3]
    float * const __restrict__ ray_dL_dz_vals = dL_dz_vals + start_idx;  // [n_samples]
    float * const __restrict__ ray_dL_ddrgbs = dL_ddrgbs + start_idx * 4;  // [n_samples, 4]

    // front-to-back composition, with early stop
    float transmittance = 1.f;
    float cur_rgb[3] = {0.f, 0.f, 0.f};
    float cur_depth = 0.f;
    for (std::uint32_t sample_idx = 0; transmittance > T_THRESHOLD && sample_idx < n_samples; ++sample_idx) {
        float const z_val = ray_z_vals[sample_idx];
        float const delta_t = ray_dss[sample_idx];
        float const density = ray_drgbs[sample_idx * 4];
        float const alpha = 1.f - __expf(-density * delta_t);

        float const weight = transmittance * alpha;

        cur_rgb[0] += weight * ray_drgbs[sample_idx * 4 + 1];
        cur_rgb[1] += weight * ray_drgbs[sample_idx * 4 + 2];
        cur_rgb[2] += weight * ray_drgbs[sample_idx * 4 + 3];
        cur_depth += weight * z_val;

        // decay transmittance before gradient calculation, as transmittance used in gradient
        // calculation is T_{i+1}.  REF: <https://note.kiui.moe/others/nerf_gradient/>
        transmittance *= 1.f - alpha;

        // set outputs
        /// z_val gradients
        ray_dL_dz_vals[sample_idx] = weight * ray_dL_dfinal_rgbd[3];

        /// density gradients
        float sample_dL_ddensity = delta_t * (
            //// gradients from final_rgbs
            + ray_dL_dfinal_rgbd[0] * (
                transmittance * ray_drgbs[sample_idx * 4 + 1] - (ray_final_rgbd[0] - cur_rgb[0])
                - ray_bgs[0] * (1.f - ray_final_opacity)
            )
            + ray_dL_dfinal_rgbd[1] * (
                transmittance * ray_drgbs[sample_idx * 4 + 2] - (ray_final_rgbd[1] - cur_rgb[1])
                - ray_bgs[1] * (1.f - ray_final_opacity)
            )
            + ray_dL_dfinal_rgbd[2] * (
                transmittance * ray_drgbs[sample_idx * 4 + 3] - (ray_final_rgbd[2] - cur_rgb[2])
                - ray_bgs[2] * (1.f - ray_final_opacity)
            )
            //// gradients from depth
            + ray_dL_dfinal_rgbd[3] * (transmittance * z_val - (ray_final_rgbd[3] - cur_depth))
        );
        //// gradients from regularizations
        ///// Penalize samples for being behind the camera's near plane.  This loss requires there
        ///// to be samples behind the camera's near plane, so the ray's starting point should only
        ///// be clipped above zero, instead of being clipped above the near distance.
        ///// REF: <https://github.com/NVlabs/instant-ngp/commit/2b825d383e11655f46786bc0a67fd0681bfceb60>
        float sample_dReg_ddensity = (density > 4e-5 && z_val < near_distance ? 1e-4f : 0.0f);

        // gradient scaling, as proposed in _Floaters No More: Radiance Field Gradient Scaling for
        // Improved Near-Camera Training_, EGSR23
        float grad_scalar = fminf(z_val * z_val, 1.f);
        // assign density gradients to output
        ray_dL_ddrgbs[sample_idx * 4 + 0] = grad_scalar * sample_dL_ddensity + sample_dReg_ddensity;

        /// color gradients
        ray_dL_ddrgbs[sample_idx * 4 + 1] = weight * ray_dL_dfinal_rgbd[0];
        ray_dL_ddrgbs[sample_idx * 4 + 2] = weight * ray_dL_dfinal_rgbd[1];
        ray_dL_ddrgbs[sample_idx * 4 + 3] = weight * ray_dL_dfinal_rgbd[2];
    }

    if (transmittance > T_THRESHOLD) {  // gradients for background colors
        ray_dL_dbgs[0] = transmittance * ray_dL_dfinal_rgbd[0];
        ray_dL_dbgs[1] = transmittance * ray_dL_dfinal_rgbd[1];
        ray_dL_dbgs[2] = transmittance * ray_dL_dfinal_rgbd[2];
    }
}

__global__ void integrate_rays_inference_kernel(
    std::uint32_t const n_total_rays
    , std::uint32_t const n_rays
    , std::uint32_t const march_steps_cap

    , float const * const __restrict__ rays_bg  // [n_total_rays, 3]
    , float const * const __restrict__ rays_rgbd  // [n_total_rays, 4]
    , float const * const __restrict__ rays_T  // [n_total_rays]

    , std::uint32_t const * const __restrict__ n_samples  // [n_rays]
    , std::uint32_t const * const __restrict__ indices  // [n_rays]
    , float const * const __restrict__ dss  // [n_rays, march_steps_cap]
    , float const * const __restrict__ z_vals  // [n_rays, march_steps_cap]
    , float const * const __restrict__ drgbs  // [n_rays, march_steps_cap, 4]

    , std::uint32_t * const __restrict__ terminate_cnt  // []
    , bool * const __restrict__ terminated  // [n_rays]
    , float * const __restrict__ rays_rgbd_out  // [n_rays, 4]
    , float * const __restrict__ rays_T_out  // [n_rays]
) {
    std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    std::uint32_t const ray_n_samples = n_samples[i];
    std::uint32_t const ray_idx = indices[i];

    if (ray_idx < n_total_rays) {
        float const * const __restrict__ ray_dss = dss + i * march_steps_cap;
        float const * const __restrict__ ray_z_vals = z_vals + i * march_steps_cap;
        float const * const __restrict__ ray_drgbs = drgbs + i * march_steps_cap * 4;

        float ray_T = rays_T[ray_idx];
        float r = rays_rgbd[ray_idx * 4 + 0];
        float g = rays_rgbd[ray_idx * 4 + 1];
        float b = rays_rgbd[ray_idx * 4 + 2];
        float ray_depth = rays_rgbd[ray_idx * 4 + 3];
        for (std::uint32_t sample_idx = 0; ray_T > T_THRESHOLD && sample_idx < ray_n_samples; ++sample_idx) {
            float const ds = ray_dss[sample_idx];
            float const z_val = ray_z_vals[sample_idx];
            float const density = ray_drgbs[sample_idx * 4];
            float const alpha = 1.f - __expf(-density * ds);
            float const weight = ray_T * alpha;
            r += weight * ray_drgbs[sample_idx * 4 + 1];
            g += weight * ray_drgbs[sample_idx * 4 + 2];
            b += weight * ray_drgbs[sample_idx * 4 + 3];
            ray_depth += weight * z_val;
            ray_T *= (1.f - alpha);
        }

        if (ray_T <= T_THRESHOLD) {
            float const denom = 1.f - ray_T;
            float const idenom = 1.f / denom;
            terminated[i] = true;
            rays_T_out[i] = 0.f;
            rays_rgbd_out[i*4+0] = r * idenom;
            rays_rgbd_out[i*4+1] = g * idenom;
            rays_rgbd_out[i*4+2] = b * idenom;
            rays_rgbd_out[i*4+3] = ray_depth * idenom;
        } else {
            terminated[i] = ray_n_samples < march_steps_cap;
            rays_rgbd_out[i*4+3] = ray_depth;
            rays_T_out[i] = ray_T;
            if (terminated[i]) {
                rays_rgbd_out[i*4+0] = r + ray_T * rays_bg[ray_idx*3+0];
                rays_rgbd_out[i*4+1] = g + ray_T * rays_bg[ray_idx*3+1];
                rays_rgbd_out[i*4+2] = b + ray_T * rays_bg[ray_idx*3+2];
            } else {
                rays_rgbd_out[i*4+0] = r;
                rays_rgbd_out[i*4+1] = g;
                rays_rgbd_out[i*4+2] = b;
            }
        }
    }

    __shared__ std::uint32_t kernel_terminate_cnt;
    if (threadIdx.x == 0) { kernel_terminate_cnt = 0; }
    __syncthreads();
    if (terminated[i]) { atomicAdd(&kernel_terminate_cnt, 1u); }
    __syncthreads();
    if (threadIdx.x == 0) { atomicAdd(terminate_cnt, kernel_terminate_cnt); }
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
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float const * const __restrict__ bgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ drgbs = static_cast<float *>(next_buffer());  // [total_samples, 4]

    // helper counter for measured_batch_size
    std::uint32_t * const __restrict__ measured_batch_size = static_cast<std::uint32_t *>(next_buffer());  // [1]

    // outputs
    float * const __restrict__ final_rgbds = static_cast<float *>(next_buffer());  // [n_rays, 4]
    float * const __restrict__ final_opacities = static_cast<float *>(next_buffer());  // [n_rays]

    // reset all outputs to zero
    CUDA_CHECK_THROW(cudaMemsetAsync(measured_batch_size, 0x00, sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(final_rgbds, 0x00, n_rays * 4 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(final_opacities, 0x00, n_rays * sizeof(float), stream));

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays
        // input arrays (7)
        , rays_sample_startidx
        , rays_n_samples
        , bgs
        , dss
        , z_vals
        , drgbs
        // helper counter
        , measured_batch_size
        // output arrays (2)
        , final_rgbds
        , final_opacities
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
    IntegratingBackwardDescriptor const &desc =
        *deserialize<IntegratingBackwardDescriptor>(opaque, opaque_len);
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const total_samples = desc.total_samples;
    float near_distance = desc.near_distance;

    /// arrays
    std::uint32_t const * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    //// original inputs
    float const * const __restrict__ bgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float const * const __restrict__ drgbs = static_cast<float *>(next_buffer());  // [total_samples, 4]
    //// original outputs
    float const * const __restrict__ final_rgbds = static_cast<float *>(next_buffer());  // [n_rays, 4]
    float const * const __restrict__ final_opacities = static_cast<float *>(next_buffer());  // [n_rays]
    //// gradient inputs
    float const * const __restrict__ dL_dfinal_rgbds = static_cast<float *>(next_buffer());  // [n_rays, 4]

    // outputs
    float * const __restrict__ dL_dbgs = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float * const __restrict__ dL_dz_vals = static_cast<float *>(next_buffer());  // [total_samples]
    float * const __restrict__ dL_ddrgbs = static_cast<float *>(next_buffer());  // [total_samples, 4]

    // reset all outputs to zeros
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dbgs, 0x00, n_rays * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_dz_vals, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dL_ddrgbs, 0x00, total_samples * 4 * sizeof(float), stream));

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static arguments
        n_rays
        , near_distance

        // input arrays
        , rays_sample_startidx
        , rays_n_samples
        /// original inputs
        , bgs
        , dss
        , z_vals
        , drgbs
        /// original outputs
        , final_rgbds
        , final_opacities
        /// gradient inputs
        , dL_dfinal_rgbds

        // output arrays
        , dL_dbgs
        , dL_dz_vals
        , dL_ddrgbs
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
    float const * const __restrict__ rays_rgbd = static_cast<float *>(next_buffer());  // [n_total_rays, 4]
    float const * const __restrict__ rays_T = static_cast<float *>(next_buffer());  // [n_total_rays]

    std::uint32_t const * const __restrict__ n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ indices = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float const * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float const * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float const * const __restrict__ drgbs = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap, 4]

    // outputs
    std::uint32_t * const __restrict__ terminate_cnt = static_cast<std::uint32_t *>(next_buffer());  // [1]
    bool * const __restrict__ terminated = static_cast<bool *>(next_buffer());  // [n_rays]
    float * const __restrict__ rays_rgbd_out = static_cast<float *>(next_buffer());  // [n_rays, 4]
    float * const __restrict__ rays_T_out = static_cast<float *>(next_buffer());  // [n_rays]

    CUDA_CHECK_THROW(cudaMemsetAsync(terminate_cnt, 0x00, sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(terminated, false, n_rays * sizeof(bool), stream));

    CUDA_CHECK_THROW(cudaMemsetAsync(rays_rgbd_out, 0x00, n_rays * 4 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_T_out, 0x00, n_rays * sizeof(float), stream));

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    integrate_rays_inference_kernel<<<numBlocks, blockSize, 1 * sizeof(std::uint32_t), stream>>>(
        n_total_rays
        , n_rays
        , march_steps_cap

        , rays_bg
        , rays_rgbd
        , rays_T

        , n_samples
        , indices
        , dss
        , z_vals
        , drgbs

        , terminate_cnt
        , terminated
        , rays_rgbd_out
        , rays_T_out
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
