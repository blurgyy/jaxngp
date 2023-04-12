#include "volrend.h"
#include "../serde.h"

namespace volrendjax {

namespace {

// this is the same function as `calc_dt` in ngp_pl's implementation
inline __device__ float calc_ds(float ray_t, float stepsize_portion, float bound, std::uint32_t grid_res, std::uint32_t max_steps) {
    // from appendix E.1 of the NGP paper (the paper sets stepsize_portion=0 for synthetic scenes
    // and 1/256 for others)
    return clampf(
        ray_t * stepsize_portion,
        2 * (float)SQRT3 / max_steps,
        2 * (float)SQRT3 * bound / grid_res
    );
}

inline __device__ std::uint32_t mip_from_xyz(float x, float y, float z, std:: uint32_t K) {
    float max_coord = fmaxf(fabsf(x), fmaxf(fabsf(y), fabsf(z)));
    int exponent;
    frexpf(max_coord, &exponent);
    return clampf(exponent, 0, K-1);
}

// return the finest cascade of occupancy grids that has cell side-length larger than ds (appendix E.2)
inline __device__ std::uint32_t mip_from_ds(float ds, std::uint32_t G, std::uint32_t K) {
    int exponent;
    frexpf(ds * G, &exponent);
    return clampf(exponent, 0, K-1);
}

inline __device__ std::uint32_t expand_bits(std::uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline __device__ std::uint32_t __morton3D(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
    uint32_t xx = expand_bits(x);
    uint32_t yy = expand_bits(y);
    uint32_t zz = expand_bits(z);
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
    std::uint32_t n_rays
    , std::uint32_t max_n_samples
    , std::uint32_t max_steps
    , std::uint32_t K
    , std::uint32_t G
    , float bound
    , float stepsize_portion

    // inputs
    , float const * const __restrict__ rays_o  // [n_rays]
    , float const * const __restrict__ rays_d  // [n_rays]
    , float const * const __restrict__ t_starts  // [n_rays]
    , float const * const __restrict__ t_ends  // [n_rays]
    , float const * const __restrict__ noises  // [n_rays]
    , std::uint8_t const * const __restrict__ occupancy_bitfield  // [K*G*G*G//8]

    // outputs
    , std::uint32_t * const __restrict__ rays_n_samples  // [n_rays]
    , bool * const __restrict__ valid_mask  // [n_rays, max_n_samples]
    , float * const __restrict__ xyzs  // [n_rays, max_n_samples, 3]
    , float * const __restrict__ dirs  // [n_rays, max_n_samples, 3]
    , float * const __restrict__ dss  // [n_rays, max_n_samples]
    , float * const __restrict__ z_vals  // [n_rays, max_n_samples]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays) { return; }

    // input arrays
    float const * const __restrict__ ray_o = rays_o + i * 3;  // [3]
    float const * const __restrict__ ray_d = rays_d + i * 3;  // [3]
    float const ray_t_start = t_starts[i];  // [] (a scalar has no shape)
    float const ray_t_end = t_ends[i];  // [] (a scalar has no shape)
    float const ray_noise = noises[i];  // [] (a scalar has no shape)

    // output arrays
    bool * const __restrict__ ray_valid_mask = valid_mask + i * max_n_samples;
    float * const __restrict__ ray_xyzs = xyzs + i * max_n_samples * 3;
    float * const __restrict__ ray_dirs = dirs + i * max_n_samples * 3;
    float * const __restrict__ ray_dss = dss + i * max_n_samples;
    float * const __restrict__ ray_z_vals = z_vals + i * max_n_samples;

    // if ray does not intersect with scene bounding box, no need to generate samples
    if (ray_t_start <= 0) { return; }
    if (ray_t_start >= ray_t_end) { return; }

    // precompute
    std::uint32_t G3 = G*G*G;

    // actually march rays
    std::uint32_t sample_idx = 0;
    float ray_t = ray_t_start;
    ray_t += calc_ds(ray_t, stepsize_portion, bound, G, max_steps) * ray_noise;
    while(sample_idx < max_n_samples && ray_t < ray_t_end) {
        float const x = ray_o[0] + ray_t * ray_d[0];
        float const y = ray_o[1] + ray_t * ray_d[1];
        float const z = ray_o[2] + ray_t * ray_d[2];

        float const ds = calc_ds(ray_t, stepsize_portion, bound, G, max_steps);

        // among the grids covering xyz, the finest one with cell side-length larger than Δ𝑡 is
        // queried.
        std::uint32_t cascade = max(
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
        bool const occupied = occupancy_bitfield[occupancy_grid_idx / 8] & (1 << (occupancy_grid_idx & 7));  // (x&7) is the same as (x%8)

        if (occupied) {
            ray_valid_mask[sample_idx] = true;  // set true
            ray_xyzs[sample_idx * 3 + 0] = x;
            ray_xyzs[sample_idx * 3 + 1] = y;
            ray_xyzs[sample_idx * 3 + 2] = z;
            ray_dirs[sample_idx * 3 + 0] = ray_d[0];
            ray_dirs[sample_idx * 3 + 1] = ray_d[1];
            ray_dirs[sample_idx * 3 + 2] = ray_d[2];
            ray_dss[sample_idx] = ds;
            ray_z_vals[sample_idx] = ray_t;

            ++sample_idx;
            ray_t += ds;
        } else {
            float const tx = (((grid_x + .5f + .5f * signf(ray_d[0])) / G * 2 - 1) * mip_bound - x) / ray_d[0];
            float const ty = (((grid_y + .5f + .5f * signf(ray_d[1])) / G * 2 - 1) * mip_bound - y) / ray_d[1];
            float const tz = (((grid_z + .5f + .5f * signf(ray_d[2])) / G * 2 - 1) * mip_bound - z) / ray_d[2];

            // distance to next voxel
            float const tt = ray_t + fmaxf(0.0f, fminf(tx, fminf(ty, tz)));
            // step until next voxel
            do { 
                ray_t += calc_ds(ray_t, stepsize_portion, bound, G, max_steps);
            } while (ray_t < tt);
        }
    }
    rays_n_samples[i] = sample_idx;
}

__global__ void morton3d_kernel(
    // inputs
    /// static
    std::uint32_t length

    /// array
    , std::uint32_t const * const __restrict__ xyzs  // [length, 3]

    // outputs
    , std::uint32_t * const __restrict__ idcs  // [length]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) { return; }

    idcs[i] = __morton3D(xyzs[i*3+0], xyzs[i*3+1], xyzs[i*3+2]);
}

__global__ void morton3d_invert_kernel(
    // inputs
    /// static
    std::uint32_t length

    /// array
    , std::uint32_t const * const __restrict__ idcs  // [length]

    // outputs
    , std::uint32_t * const __restrict__ xyzs  // [length, 3]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) { return; }

    xyzs[i*3+0] = __morton3D_invert(idcs[i] >> 0);
    xyzs[i*3+1] = __morton3D_invert(idcs[i] >> 1);
    xyzs[i*3+2] = __morton3D_invert(idcs[i] >> 2);
}

void march_rays_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    MarchingDescriptor const &desc = *deserialize<MarchingDescriptor>(opaque, opaque_len);
    std::uint32_t n_rays = desc.n_rays;
    std::uint32_t max_n_samples = desc.max_n_samples;
    std::uint32_t max_steps = desc.max_steps;
    std::uint32_t K = desc.K;
    std::uint32_t G = desc.G;
    float bound = desc.bound;
    float stepsize_portion = desc.stepsize_portion;

    /// arrays
    float const * const __restrict__ rays_o = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ rays_d = static_cast<float *>(next_buffer());  // [n_rays, 3]
    float const * const __restrict__ t_starts = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ t_ends = static_cast<float *>(next_buffer());  // [n_rays]
    float const * const __restrict__ noises = static_cast<float *>(next_buffer());  // [n_rays]
    std::uint8_t const * const __restrict__ occupancy_bitfield = static_cast<std::uint8_t *>(next_buffer());  // [K*G*G*G//8]

    // outputs
    std::uint32_t * const __restrict__ rays_n_samples = static_cast<std::uint32_t *>(next_buffer());  // [n_rays]
    bool * const __restrict__ valid_mask = static_cast<bool *>(next_buffer());  // [n_rays * max_n_samples]
    float * const __restrict__ xyzs = static_cast<float *>(next_buffer());  // [n_rays * max_n_samples, 3]
    float * const __restrict__ dirs = static_cast<float *>(next_buffer());  // [n_rays * max_n_samples, 3]
    float * const __restrict__ dss = static_cast<float *>(next_buffer());  // [n_rays * max_n_samples]
    float * const __restrict__ z_vals = static_cast<float *>(next_buffer());  // [n_rays * max_n_samples]

    // reset outputs to zeros
    cudaMemset(rays_n_samples, 0x00, n_rays * sizeof(std::uint32_t));
    cudaMemset(valid_mask, false, n_rays * max_n_samples * sizeof(bool));
    cudaMemset(xyzs, 0x00, n_rays * max_n_samples * 3 * sizeof(float));
    cudaMemset(dirs, 0x00, n_rays * max_n_samples * 3 * sizeof(float));
    cudaMemset(dss, 0x00, n_rays * max_n_samples * sizeof(float));
    cudaMemset(z_vals, 0x00, n_rays * max_n_samples * sizeof(float));

    // kernel launch
    int blockSize = 256;
    int numBlocks = (n_rays + blockSize - 1) / blockSize;
    march_rays_kernel<<<numBlocks, blockSize, 0, stream>>>(
        // static
        n_rays
        , max_n_samples
        , max_steps
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

        // outputs
        , rays_n_samples
        , valid_mask
        , xyzs
        , dirs
        , dss
        , z_vals
    );

    // abort on error
    CUDA_CHECK_THROW(cudaGetLastError());
}

void morton3d_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // buffer indexing helper
    std::uint32_t __buffer_idx = 0;
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    Morton3DDescriptor const &desc = *deserialize<Morton3DDescriptor>(opaque, opaque_len);

    /// array
    std::uint32_t const * const __restrict__ xyzs = static_cast<std::uint32_t *>(next_buffer());  // [length, 3]

    // output
    std::uint32_t * const __restrict__ idcs = static_cast<std::uint32_t *>(next_buffer());  // [length]

    // kernel launch
    int blockSize = 256;
    int numBlocks = (desc.length + blockSize - 1) / blockSize;
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
    auto next_buffer = [&]() { return buffers[__buffer_idx++]; };

    // inputs
    /// static
    Morton3DDescriptor const &desc = *deserialize<Morton3DDescriptor>(opaque, opaque_len);

    /// array
    std::uint32_t const * const __restrict__ idcs = static_cast<std::uint32_t *>(next_buffer());  // [length]

    /// output
    std::uint32_t * const __restrict__ xyzs = static_cast<std::uint32_t *>(next_buffer());

    // kernel launch
    int blockSize = 256;
    int numBlocks = (desc.length + blockSize - 1) / blockSize;
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
