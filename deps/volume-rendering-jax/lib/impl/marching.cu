#include <stdexcept>

#include <driver_types.h>
#include <serde-helper/serde.h>

#include "volrend.h"


namespace volrendjax {

namespace {

// this is the same function as `calc_dt` in ngp_pl's implementation
inline __device__ float calc_ds(float ray_t, float stepsize_portion, float bound, float inv_grid_res, std::uint32_t diagonal_n_steps) {
    // from appendix E.1 of the NGP paper (the paper sets stepsize_portion=0 for synthetic scenes
    // and 1/256 for others)
    return clampf(
        ray_t * stepsize_portion,
        2 * (float)SQRT3 * fminf(bound, 1.f) / diagonal_n_steps,
        2 * (float)SQRT3 * bound * inv_grid_res
    );
}

inline __device__ std::uint32_t mip_from_xyz(float x, float y, float z, std:: uint32_t K) {
    if (K == 1) { return 0; }
    float const max_coord = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    int exponent;
    frexpf(max_coord, &exponent);
    return static_cast<std::uint32_t>(clampi(exponent, 0, K-1));
}

// return the finest cascade of occupancy grids that has cell side-length larger than ds (appendix E.2)
inline __device__ std::uint32_t mip_from_ds(float ds, std::uint32_t G, std::uint32_t K) {
    if (K == 1) { return 0; }
    int exponent;
    frexpf(ds * G, &exponent);
    return static_cast<std::uint32_t>(clampi(exponent, 0, K-1));
}

inline __device__ std::uint32_t expand_bits(std::uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __device__ std::uint32_t __morton3D(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    std::uint32_t const xx = expand_bits(x);
    std::uint32_t const yy = expand_bits(y);
    std::uint32_t const zz = expand_bits(z);
    return xx | (yy << 1) | (zz << 2);
}

inline __device__ std::uint32_t __morton3D_invert(std::uint32_t x) {
    x = x & 0x49249249;
    x = (x | (x >> 2)) & 0xc30c30c3;
    x = (x | (x >> 4)) & 0x0f00f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & 0x0000ffff;
    return x;
}

// kernel, REF: <https://github.com/ashawkey/torch-ngp/blob/b6e080468925f0bb44827b4f8f0ed08291dcf8a9/raymarching/src/raymarching.cu#L312>
__global__ void march_rays_kernel(
    // static
    std::uint32_t const n_rays
    , std::uint32_t const total_samples
    , std::uint32_t const diagonal_n_steps
    , std::uint32_t const K
    , std::uint32_t const G
    , float const bound
    , float const stepsize_portion

    // inputs
    , float const * const __restrict__ rays_o  // [n_rays, 3]
    , float const * const __restrict__ rays_d  // [n_rays, 3]
    , float const * const __restrict__ t_starts  // [n_rays]
    , float const * const __restrict__ t_ends  // [n_rays]
    , float const * const __restrict__ noises  // [n_rays]
    , std::uint8_t const * const __restrict__ occupancy_bitfield  // [K*G*G*G//8]

    // accumulator for writing a compact output samples array
    , std::uint32_t * const __restrict__ counter

    // outputs
    , std::uint32_t * const __restrict__ rays_n_samples  // [n_rays]
    , std::uint32_t * const __restrict__ rays_sample_startidx  // [n_rays]
    , float * const __restrict__ xyzs  // [total_samples, 3]
    , float * const __restrict__ dirs  // [total_samples, 3]
    , float * const __restrict__ dss  // [total_samples]
    , float * const __restrict__ z_vals  // [total_samples]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // inputs
    float const ox = rays_o[i * 3 + 0];
    float const oy = rays_o[i * 3 + 1];
    float const oz = rays_o[i * 3 + 2];
    float const dx = rays_d[i * 3 + 0];
    float const dy = rays_d[i * 3 + 1];
    float const dz = rays_d[i * 3 + 2];
    float const ray_t_start = t_starts[i];  // [] (a scalar has no shape)
    float const ray_t_end = t_ends[i];  // [] (a scalar has no shape)

    if (ray_t_end <= ray_t_start) { return; }

    float const ray_noise = noises[i];  // [] (a scalar has no shape)

    // precompute
    std::uint32_t const G3 = G*G*G;
    float const idx = 1.f / dx, idy = 1.f / dy, idz = 1.f / dz;
    float const iG = 1.f / G;

    // actually march rays
    /// but not writing the samples to output!  Writing is done in another marching pass below
    std::uint32_t ray_n_samples = 0;
    float ray_t = ray_t_start;
    ray_t += calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps) * ray_noise;
    while (ray_n_samples < diagonal_n_steps * bound && ray_t < ray_t_end) {
        float const x = ox + ray_t * dx;
        float const y = oy + ray_t * dy;
        float const z = oz + ray_t * dz;

        float const ds = calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);

        // among the grids covering xyz, the finest one with cell side-length larger than Δ𝑡 is
        // queried.
        std::uint32_t const cascade = max(
            mip_from_xyz(x, y, z, K),
            mip_from_ds(ds, G, K)
        );

        // the bound of this mip is [-mip_bound, mip_bound]
        float const mip_bound = fminf(scalbnf(1.f, cascade), bound);

        // round down
        std::uint32_t const grid_x = clampi((int)(.5f * (x / mip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_y = clampi((int)(.5f * (y / mip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_z = clampi((int)(.5f * (z / mip_bound + 1) * G), 0, G-1);

        std::uint32_t const occupancy_grid_idx = cascade * G3 + __morton3D(grid_x, grid_y, grid_z);
        bool const occupied = occupancy_bitfield[occupancy_grid_idx >> 3] & (1 << (occupancy_grid_idx & 7u));  // (x>>3)==(int)(x/8), (x&7)==(x%8)

        if (occupied) {
            ++ray_n_samples;
            ray_t += ds;
        } else {
            float const tx = (((grid_x + .5f + .5f * signf(dx)) * iG * 2 - 1) * mip_bound - x) * idx;
            float const ty = (((grid_y + .5f + .5f * signf(dy)) * iG * 2 - 1) * mip_bound - y) * idy;
            float const tz = (((grid_z + .5f + .5f * signf(dz)) * iG * 2 - 1) * mip_bound - z) * idz;

            // distance to next voxel
            float const tt = ray_t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                ray_t += calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);
            } while (ray_t < tt);
        }
    }

    // can safely return here because before launching the kernel we have `memset`ed every output
    // array to zeros
    if (ray_n_samples == 0) { return; }

    // record how many samples are generated along this ray
    rays_n_samples[i] = ray_n_samples;

    // record the index of the first generated sample on this ray
    std::uint32_t const ray_sample_startidx = atomicAdd(counter, ray_n_samples);
    if (ray_sample_startidx + ray_n_samples > total_samples) { return; }
    rays_sample_startidx[i] = ray_sample_startidx;

    // output arrays
    float * const __restrict__ ray_xyzs = xyzs + ray_sample_startidx * 3;
    float * const __restrict__ ray_dirs = dirs + ray_sample_startidx * 3;
    float * const __restrict__ ray_dss = dss + ray_sample_startidx;
    float * const __restrict__ ray_z_vals = z_vals + ray_sample_startidx;

    // march rays again, this time write sampled points to output
    std::uint32_t steps = 0;
    ray_t = ray_t_start;
    ray_t += calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps) * ray_noise;
    // NOTE:
    //  we still need the condition (ray_t < ray_t_end) because if a ray never hits an occupied grid
    //  cell, its `steps` won't increment, adding this condition avoids infinite loops.
    while (steps < ray_n_samples && ray_t < ray_t_end) {
        float const x = ox + ray_t * dx;
        float const y = oy + ray_t * dy;
        float const z = oz + ray_t * dz;

        float const ds = calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);

        // among the grids covering xyz, the finest one with cell side-length larger than Δ𝑡 is
        // queried.
        std::uint32_t const cascade = max(
            mip_from_xyz(x, y, z, K),
            mip_from_ds(ds, G, K)
        );

        // the bound of this mip is [-mip_bound, mip_bound]
        float const mip_bound = fminf(scalbnf(1.f, cascade), bound);
        float const imip_bound = 1.f / mip_bound;

        // round down
        std::uint32_t const grid_x = clampi((int)(.5f * (x * imip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_y = clampi((int)(.5f * (y * imip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_z = clampi((int)(.5f * (z * imip_bound + 1) * G), 0, G-1);

        std::uint32_t const occupancy_grid_idx = cascade * G3 + __morton3D(grid_x, grid_y, grid_z);
        bool const occupied = occupancy_bitfield[occupancy_grid_idx >> 3] & (1 << (occupancy_grid_idx & 7u));  // (x>>3)==(int)(x/8), (x&7)==(x%8)

        if (occupied) {
            ray_xyzs[steps * 3 + 0] = x;
            ray_xyzs[steps * 3 + 1] = y;
            ray_xyzs[steps * 3 + 2] = z;
            ray_dirs[steps * 3 + 0] = dx;
            ray_dirs[steps * 3 + 1] = dy;
            ray_dirs[steps * 3 + 2] = dz;
            ray_dss[steps] = ds;
            ray_z_vals[steps] = ray_t;

            ++steps;
            ray_t += ds;
        } else {
            float const tx = (((grid_x + .5f + .5f * signf(dx)) * iG * 2 - 1) * mip_bound - x) * idx;
            float const ty = (((grid_y + .5f + .5f * signf(dy)) * iG * 2 - 1) * mip_bound - y) * idy;
            float const tz = (((grid_z + .5f + .5f * signf(dz)) * iG * 2 - 1) * mip_bound - z) * idz;

            // distance to next voxel
            float const tt = ray_t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                ray_t += calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);
            } while (ray_t < tt);
        }
    }
}


__global__ void march_rays_inference_kernel(
    std::uint32_t const n_total_rays
    , std::uint32_t const n_rays
    , std::uint32_t const diagonal_n_steps
    , std::uint32_t const K
    , std::uint32_t const G
    , std::uint32_t const march_steps_cap
    , float const bound
    , float const stepsize_portion

    , float const * const __restrict__ rays_o  // [n_total_rays, 3]
    , float const * const __restrict__ rays_d  // [n_total_rays, 3]
    , float const * const __restrict__ t_starts  // [n_total_rays]
    , float const * const __restrict__ t_ends  // [n_total_rays]
    , std::uint8_t const * const __restrict__ occupancy_bitfield  // [K*G*G*G//8]
    , bool const * const __restrict__ terminated  // [n_rays]
    , std::uint32_t const * const __restrict__ indices_in  // [n_rays]

    , std::uint32_t * const __restrict__ counter  // [1]
    , std::uint32_t * const __restrict__ indices_out  // [n_rays]
    , std::uint32_t * const __restrict__ n_samples  // [n_rays]
    , float * const __restrict__ t_starts_out  // [n_rays]
    , float * const __restrict__ xyzdirs  // [n_rays, march_steps_cap, 6]
    , float * const __restrict__ dss  // [n_rays, march_steps_cap]
    , float * const __restrict__ z_vals  // [n_rays, march_steps_cap]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    std::uint32_t const ray_idx = terminated[i] ? atomicAdd(counter, 1u) : indices_in[i];
    indices_out[i] = ray_idx;
    if (ray_idx >= n_total_rays) { return; }

    float const ox = rays_o[ray_idx * 3 + 0];
    float const oy = rays_o[ray_idx * 3 + 1];
    float const oz = rays_o[ray_idx * 3 + 2];
    float const dx = rays_d[ray_idx * 3 + 0];
    float const dy = rays_d[ray_idx * 3 + 1];
    float const dz = rays_d[ray_idx * 3 + 2];
    float const ray_t_start = t_starts[ray_idx];
    float const ray_t_end = t_ends[ray_idx];

    if (ray_t_end <= ray_t_start) { return; }

    float * const __restrict__ ray_xyzdirs = xyzdirs + i * march_steps_cap * 6;
    float * const __restrict__ ray_dss = dss + i * march_steps_cap;
    float * const __restrict__ ray_z_vals = z_vals + i * march_steps_cap;

    std::uint32_t const G3 = G*G*G;
    float const idx = 1.f / dx, idy = 1.f / dy, idz = 1.f / dz;
    float const iG = 1.f / G;

    std::uint32_t steps = 0;
    float ray_t = ray_t_start;
    while (steps < march_steps_cap && ray_t < ray_t_end) {
        float const x = ox + ray_t * dx;
        float const y = oy + ray_t * dy;
        float const z = oz + ray_t * dz;

        float const ds = calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);

        // among the grids covering xyz, the finest one with cell side-length larger than Δ𝑡 is
        // queried.
        std::uint32_t const cascade = max(
            mip_from_xyz(x, y, z, K),
            mip_from_ds(ds, G, K)
        );

        // the bound of this mip is [-mip_bound, mip_bound]
        float const mip_bound = fminf(scalbnf(1.f, cascade), bound);
        float const imip_bound = 1.f / mip_bound;

        // round down
        std::uint32_t const grid_x = clampi((int)(.5f * (x * imip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_y = clampi((int)(.5f * (y * imip_bound + 1) * G), 0, G-1);
        std::uint32_t const grid_z = clampi((int)(.5f * (z * imip_bound + 1) * G), 0, G-1);

        std::uint32_t const occupancy_grid_idx = cascade * G3 + __morton3D(grid_x, grid_y, grid_z);
        bool const occupied = occupancy_bitfield[occupancy_grid_idx >> 3] & (1 << (occupancy_grid_idx & 7u));  // (x>>3)==(int)(x/8), (x&7)==(x%8)

        if (occupied) {
            ray_xyzdirs[steps * 6 + 0] = x;
            ray_xyzdirs[steps * 6 + 1] = y;
            ray_xyzdirs[steps * 6 + 2] = z;
            ray_xyzdirs[steps * 6 + 3] = dx;
            ray_xyzdirs[steps * 6 + 4] = dy;
            ray_xyzdirs[steps * 6 + 5] = dz;
            ray_dss[steps] = ds;
            ray_z_vals[steps] = ray_t;

            ++steps;
            ray_t += ds;
        } else {
            float const tx = (((grid_x + .5f + .5f * signf(dx)) * iG * 2 - 1) * mip_bound - x) * idx;
            float const ty = (((grid_y + .5f + .5f * signf(dy)) * iG * 2 - 1) * mip_bound - y) * idy;
            float const tz = (((grid_z + .5f + .5f * signf(dz)) * iG * 2 - 1) * mip_bound - z) * idz;

            // distance to next voxel
            float const tt = ray_t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                ray_t += calc_ds(ray_t, stepsize_portion, bound, iG, diagonal_n_steps);
            } while (ray_t < tt);
        }
    }
    n_samples[i] = steps;
    t_starts_out[i] = ray_t;
}

__global__ void morton3d_kernel(
    // inputs
    /// static
    std::uint32_t const length

    /// array
    , std::uint32_t const * const __restrict__ xyzs  // [length, 3]

    // outputs
    , std::uint32_t * const __restrict__ idcs  // [length]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) { return; }

    idcs[i] = __morton3D(xyzs[i*3+0], xyzs[i*3+1], xyzs[i*3+2]);
}

__global__ void morton3d_invert_kernel(
    // inputs
    /// static
    std::uint32_t const length

    /// array
    , std::uint32_t const * const __restrict__ idcs  // [length]

    // outputs
    , std::uint32_t * const __restrict__ xyzs  // [length, 3]
) {
    std::uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) { return; }

    xyzs[i*3+0] = __morton3D_invert(idcs[i] >> 0);
    xyzs[i*3+1] = __morton3D_invert(idcs[i] >> 1);
    xyzs[i*3+2] = __morton3D_invert(idcs[i] >> 2);
}

void march_rays_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    MarchingDescriptor const &desc = *deserialize<MarchingDescriptor>(opaque, opaque_len);
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const total_samples = desc.total_samples;
    std::uint32_t const diagonal_n_steps = desc.diagonal_n_steps;
    std::uint32_t const K = desc.K;
    std::uint32_t const G = desc.G;
    float const bound = desc.bound;
    float const stepsize_portion = desc.stepsize_portion;

    /// arrays
    float const * const __restrict__ rays_o = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ rays_d = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ t_starts = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ t_ends = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ noises = static_cast<float *>(next_buffer());  // [n_rays]
    std::uint8_t const * const __restrict__ occupancy_bitfield = static_cast<std::uint8_t *>(next_buffer());  // [K*G*G*G//8]

    // helper
    std::uint32_t * const __restrict__ counter = static_cast<std::uint32_t *>(next_buffer());

    // outputs
    std::uint32_t * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t * const __restrict__ rays_sample_startidx = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float * const __restrict__ xyzs = static_cast<float *>(next_buffer());  // [total_samples, 3]
    float * const __restrict__ dirs = static_cast<float *>(next_buffer());  // [total_samples, 3]
    float * const __restrict__ dss = static_cast<float *>(next_buffer());  // [total_samples]
    float * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [total_samples]

    // reset helper coutner and outputs to zeros
    CUDA_CHECK_THROW(cudaMemsetAsync(counter, 0x00, sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_n_samples, 0x00, n_rays * sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(rays_sample_startidx, 0x00, n_rays * sizeof(int), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(xyzs, 0x00, total_samples * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dirs, 0x00, total_samples * 3 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dss, 0x00, total_samples * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(z_vals, 0x00, total_samples * sizeof(float), stream));

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    march_rays_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static
        n_rays
        , total_samples
        , diagonal_n_steps
        , K
        , G
        , bound
        , stepsize_portion

        // inputs
        , rays_o
        , rays_d
        , t_starts
        , t_ends
        , noises
        , occupancy_bitfield

        , counter

        // outputs
        , rays_n_samples
        , rays_sample_startidx
        , xyzs
        , dirs
        , dss
        , z_vals
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void march_rays_inference_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    MarchingInferenceDescriptor const &desc = *deserialize<MarchingInferenceDescriptor>(opaque, opaque_len);
    std::uint32_t const n_total_rays = desc.n_total_rays;
    std::uint32_t const n_rays = desc.n_rays;
    std::uint32_t const diagonal_n_steps = desc.diagonal_n_steps;
    std::uint32_t const K = desc.K;
    std::uint32_t const G = desc.G;
    std::uint32_t const march_steps_cap = desc.march_steps_cap;
    float const bound = desc.bound;
    float const stepsize_portion = desc.stepsize_portion;

    /// arrays
    float const * const __restrict__ rays_o = static_cast<float *>(next_buffer());  // [n_total_rays, 3]
    float const * const __restrict__ rays_d = static_cast<float *>(next_buffer());  // [n_total_rays, 3]
    float const * const __restrict__ t_starts = static_cast<float *>(next_buffer());  // [n_total_rays]
    float const * const __restrict__ t_ends = static_cast<float *>(next_buffer());  // [n_total_rays]
    std::uint8_t const * const __restrict__ occupancy_bitfield = static_cast<std::uint8_t *>(next_buffer());  // [K*G*G*G//8]
    std::uint32_t const * const __restrict__ counter_in = static_cast<std::uint32_t *>(next_buffer());  // [1]
    bool const * const __restrict__ terminated = static_cast<bool *>(next_buffer());  // [n_rays]
    std::uint32_t const * const __restrict__ indices_in = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]

    // outputs
    std::uint32_t * const __restrict__ counter = static_cast<std::uint32_t *>(next_buffer());  // [1]
    std::uint32_t * const __restrict__ indices_out = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    std::uint32_t * const __restrict__ n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    float * const __restrict__ t_starts_out = static_cast<float *>(next_buffer());  // [n_rays]
    float * const __restrict__ xyzdirs = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap, 6]
    float * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]
    float * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays, march_steps_cap]

    // copy input counter value to output counter
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter, counter_in, sizeof(std::uint32_t), cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

    // initialize output arrays to zeros
    CUDA_CHECK_THROW(cudaMemsetAsync(indices_out, 0x00, n_rays * sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(n_samples, 0x00, n_rays * sizeof(std::uint32_t), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(t_starts_out, 0x00, n_rays * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(xyzdirs, 0x00, n_rays * march_steps_cap * 6 * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(dss, 0x00, n_rays * march_steps_cap * sizeof(float), stream));
    CUDA_CHECK_THROW(cudaMemsetAsync(z_vals, 0x00, n_rays * march_steps_cap * sizeof(float), stream));

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (n_rays + blockSize - 1) / blockSize;
    march_rays_inference_kernel<<<numBlocks, blockSize, 0, stream>>>(
        n_total_rays
        , n_rays
        , diagonal_n_steps
        , K
        , G
        , march_steps_cap
        , bound
        , stepsize_portion

        , rays_o
        , rays_d
        , t_starts
        , t_ends
        , occupancy_bitfield
        , terminated
        , indices_in

        , counter
        , indices_out
        , n_samples
        , t_starts_out
        , xyzdirs
        , dss
        , z_vals
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void morton3d_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    Morton3DDescriptor const &desc = *deserialize<Morton3DDescriptor>(opaque, opaque_len);

    /// array
    std::uint32_t const * const __restrict__ xyzs = static_cast<std::uint32_t *>(next_buffer());  // [length, 3]

    // output
    std::uint32_t * const __restrict__ idcs = static_cast<std::uint32_t *>(next_buffer());  // [length]

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (desc.length + blockSize - 1) / blockSize;
    morton3d_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // inputs
        /// static
        desc.length
        /// array
        , xyzs
        /// output
        , idcs
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void morton3d_invert_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto const next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    Morton3DDescriptor const &desc = *deserialize<Morton3DDescriptor>(opaque, opaque_len);

    /// array
    std::uint32_t const * const __restrict__ idcs = static_cast<std::uint32_t *>(next_buffer());  // [length]

    /// output
    std::uint32_t * const __restrict__ xyzs = static_cast<std::uint32_t *>(next_buffer());

    // kernel launch
    std::uint32_t static constexpr blockSize = 512;
    std::uint32_t const numBlocks = (desc.length + blockSize - 1) / blockSize;
    morton3d_invert_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // inputs
        /// static
        desc.length
        /// array
        , idcs
        /// output
        , xyzs
    );
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

void march_rays_inference(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaqlne_len
) {
    march_rays_inference_launcher(stream, buffers, opaque, opaqlne_len);
}

void morton3d(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    morton3d_launcher(stream, buffers, opaque, opaque_len);
}

void morton3d_invert(
    cudaStream_t stream,
    void **buffers,
    char const *opaque,
    std::size_t opaque_len
) {
    morton3d_invert_launcher(stream, buffers, opaque, opaque_len);
}

}
