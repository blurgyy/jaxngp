#ifdef __CUDACC__
#define FE_INLINE __host__ __device__
#else
#define FE_INLINE inline
#endif

#include <cstdint>

#include "spherical_harmonics_encoding.h"
#include "../serde.h"


namespace shjax {

namespace {

__inline__ void check_throw(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// debugging kernel for inspecting data passed to custom op
__global__ void copy_left_to_right(std::uint32_t length, float const *lhs, float * const rhs) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll
    for (int i = index; i < length; i += stride) {
        rhs[i] = lhs[i];
    }
}

template <typename real_t>
__device__ __inline__ void sh_enc(
    std::uint32_t const degree,
    float const x,
    float const y,
    float const z,
    real_t * const __restrict__ o
) {
    // adapted from <https://github.com/NVlabs/tiny-cuda-nn/blob/39df2387a684e4fe0cfa33542aebf5eab237716b/include/tiny-cuda-nn/encodings/spherical_harmonics.h#L52-L123>

    float xy=x*y, xz=x*z, yz=y*z, x2=x*x, y2=y*y, z2=z*z;
    float x4=x2*x2, y4=y2*y2, z4=z2*z2;
    float x6=x4*x2, y6=y4*y2, z6=z4*z2;

    o[0] = (real_t)(0.28209479177387814f);                          // 1/(2*sqrt(pi))
    if (degree <= 1) { return; }

    o[1] = (real_t)(-0.48860251190291987f*y);                               // -sqrt(3)*y/(2*sqrt(pi))
    o[2] = (real_t)(0.48860251190291987f*z);                                // sqrt(3)*z/(2*sqrt(pi))
    o[3] = (real_t)(-0.48860251190291987f*x);                               // -sqrt(3)*x/(2*sqrt(pi))
    if (degree <= 2) { return; }

    o[4] = (real_t)(1.0925484305920792f*xy);                                // sqrt(15)*xy/(2*sqrt(pi))
    o[5] = (real_t)(-1.0925484305920792f*yz);                               // -sqrt(15)*yz/(2*sqrt(pi))
    o[6] = (real_t)(0.94617469575755997f*z2 - 0.31539156525251999f);                         // sqrt(5)*(3*z2 - 1)/(4*sqrt(pi))
    o[7] = (real_t)(-1.0925484305920792f*xz);                               // -sqrt(15)*xz/(2*sqrt(pi))
    o[8] = (real_t)(0.54627421529603959f*x2 - 0.54627421529603959f*y2);                              // sqrt(15)*(x2 - y2)/(4*sqrt(pi))
    if (degree <= 3) { return; }

    o[9] = (real_t)(0.59004358992664352f*y*(-3.0f*x2 + y2));                         // sqrt(70)*y*(-3*x2 + y2)/(8*sqrt(pi))
    o[10] = (real_t)(2.8906114426405538f*xy*z);                             // sqrt(105)*xy*z/(2*sqrt(pi))
    o[11] = (real_t)(0.45704579946446572f*y*(1.0f - 5.0f*z2));                                // sqrt(42)*y*(1 - 5*z2)/(8*sqrt(pi))
    o[12] = (real_t)(0.3731763325901154f*z*(5.0f*z2 - 3.0f));                         // sqrt(7)*z*(5*z2 - 3)/(4*sqrt(pi))
    o[13] = (real_t)(0.45704579946446572f*x*(1.0f - 5.0f*z2));                                // sqrt(42)*x*(1 - 5*z2)/(8*sqrt(pi))
    o[14] = (real_t)(1.4453057213202769f*z*(x2 - y2));                              // sqrt(105)*z*(x2 - y2)/(4*sqrt(pi))
    o[15] = (real_t)(0.59004358992664352f*x*(-x2 + 3.0f*y2));                                // sqrt(70)*x*(-x2 + 3*y2)/(8*sqrt(pi))
    if (degree <= 4) { return; }

    o[16] = (real_t)(2.5033429417967046f*xy*(x2 - y2));                             // 3*sqrt(35)*xy*(x2 - y2)/(4*sqrt(pi))
    o[17] = (real_t)(1.7701307697799304f*yz*(-3.0f*x2 + y2));                                // 3*sqrt(70)*yz*(-3*x2 + y2)/(8*sqrt(pi))
    o[18] = (real_t)(0.94617469575756008f*xy*(7.0f*z2 - 1.0f));                               // 3*sqrt(5)*xy*(7*z2 - 1)/(4*sqrt(pi))
    o[19] = (real_t)(0.66904654355728921f*yz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*yz*(3 - 7*z2)/(8*sqrt(pi))
    o[20] = (real_t)(-3.1735664074561294f*z2 + 3.7024941420321507f*z4 + 0.31735664074561293f);                                // 3*(-30*z2 + 35*z4 + 3)/(16*sqrt(pi))
    o[21] = (real_t)(0.66904654355728921f*xz*(3.0f - 7.0f*z2));                               // 3*sqrt(10)*xz*(3 - 7*z2)/(8*sqrt(pi))
    o[22] = (real_t)(0.47308734787878004f*(x2 - y2)*(7.0f*z2 - 1.0f));                                // 3*sqrt(5)*(x2 - y2)*(7*z2 - 1)/(8*sqrt(pi))
    o[23] = (real_t)(1.7701307697799304f*xz*(-x2 + 3.0f*y2));                                // 3*sqrt(70)*xz*(-x2 + 3*y2)/(8*sqrt(pi))
    o[24] = (real_t)(-3.7550144126950569f*x2*y2 + 0.62583573544917614f*x4 + 0.62583573544917614f*y4);                         // 3*sqrt(35)*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    if (degree <= 5) { return; }

    o[25] = (real_t)(0.65638205684017015f*y*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(154)*y*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    o[26] = (real_t)(8.3026492595241645f*xy*z*(x2 - y2));                           // 3*sqrt(385)*xy*z*(x2 - y2)/(4*sqrt(pi))
    o[27] = (real_t)(-0.48923829943525038f*y*(3.0f*x2 - y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*y*(3*x2 - y2)*(9*z2 - 1)/(32*sqrt(pi))
    o[28] = (real_t)(4.7935367849733241f*xy*z*(3.0f*z2 - 1.0f));                              // sqrt(1155)*xy*z*(3*z2 - 1)/(4*sqrt(pi))
    o[29] = (real_t)(0.45294665119569694f*y*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*y*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    o[30] = (real_t)(0.1169503224534236f*z*(-70.0f*z2 + 63.0f*z4 + 15.0f));                            // sqrt(11)*z*(-70*z2 + 63*z4 + 15)/(16*sqrt(pi))
    o[31] = (real_t)(0.45294665119569694f*x*(14.0f*z2 - 21.0f*z4 - 1.0f));                             // sqrt(165)*x*(14*z2 - 21*z4 - 1)/(16*sqrt(pi))
    o[32] = (real_t)(2.3967683924866621f*z*(x2 - y2)*(3.0f*z2 - 1.0f));                               // sqrt(1155)*z*(x2 - y2)*(3*z2 - 1)/(8*sqrt(pi))
    o[33] = (real_t)(-0.48923829943525038f*x*(x2 - 3.0f*y2)*(9.0f*z2 - 1.0f));                         // -sqrt(770)*x*(x2 - 3*y2)*(9*z2 - 1)/(32*sqrt(pi))
    o[34] = (real_t)(2.0756623148810411f*z*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(-6*x2*y2 + x4 + y4)/(16*sqrt(pi))
    o[35] = (real_t)(0.65638205684017015f*x*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(154)*x*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    if (degree <= 6) { return; }

    o[36] = (real_t)(1.3663682103838286f*xy*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                               // sqrt(6006)*xy*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    o[37] = (real_t)(2.3666191622317521f*yz*(10.0f*x2*y2 - 5.0f*x4 - y4));                            // 3*sqrt(2002)*yz*(10*x2*y2 - 5*x4 - y4)/(32*sqrt(pi))
    o[38] = (real_t)(2.0182596029148963f*xy*(x2 - y2)*(11.0f*z2 - 1.0f));                             // 3*sqrt(91)*xy*(x2 - y2)*(11*z2 - 1)/(8*sqrt(pi))
    o[39] = (real_t)(-0.92120525951492349f*yz*(3.0f*x2 - y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*yz*(3*x2 - y2)*(11*z2 - 3)/(32*sqrt(pi))
    o[40] = (real_t)(0.92120525951492349f*xy*(-18.0f*z2 + 33.0f*z4 + 1.0f));                           // sqrt(2730)*xy*(-18*z2 + 33*z4 + 1)/(32*sqrt(pi))
    o[41] = (real_t)(0.58262136251873131f*yz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*yz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    o[42] = (real_t)(6.6747662381009842f*z2 - 20.024298714302954f*z4 + 14.684485723822165f*z6 - 0.31784601133814211f);                         // sqrt(13)*(105*z2 - 315*z4 + 231*z6 - 5)/(32*sqrt(pi))
    o[43] = (real_t)(0.58262136251873131f*xz*(30.0f*z2 - 33.0f*z4 - 5.0f));                            // sqrt(273)*xz*(30*z2 - 33*z4 - 5)/(16*sqrt(pi))
    o[44] = (real_t)(0.46060262975746175f*(x2 - y2)*(11.0f*z2*(3.0f*z2 - 1.0f) - 7.0f*z2 + 1.0f));                               // sqrt(2730)*(x2 - y2)*(11*z2*(3*z2 - 1) - 7*z2 + 1)/(64*sqrt(pi))
    o[45] = (real_t)(-0.92120525951492349f*xz*(x2 - 3.0f*y2)*(11.0f*z2 - 3.0f));                               // -sqrt(2730)*xz*(x2 - 3*y2)*(11*z2 - 3)/(32*sqrt(pi))
    o[46] = (real_t)(0.50456490072872406f*(11.0f*z2 - 1.0f)*(-6.0f*x2*y2 + x4 + y4));                          // 3*sqrt(91)*(11*z2 - 1)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    o[47] = (real_t)(2.3666191622317521f*xz*(10.0f*x2*y2 - x4 - 5.0f*y4));                            // 3*sqrt(2002)*xz*(10*x2*y2 - x4 - 5*y4)/(32*sqrt(pi))
    o[48] = (real_t)(10.247761577878714f*x2*y4 - 10.247761577878714f*x4*y2 + 0.6831841051919143f*x6 - 0.6831841051919143f*y6);                         // sqrt(6006)*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    if (degree <= 7) { return; }

    o[49] = (real_t)(0.70716273252459627f*y*(-21.0f*x2*y4 + 35.0f*x4*y2 - 7.0f*x6 + y6));                              // 3*sqrt(715)*y*(-21*x2*y4 + 35*x4*y2 - 7*x6 + y6)/(64*sqrt(pi))
    o[50] = (real_t)(5.2919213236038001f*xy*z*(-10.0f*x2*y2 + 3.0f*x4 + 3.0f*y4));                             // 3*sqrt(10010)*xy*z*(-10*x2*y2 + 3*x4 + 3*y4)/(32*sqrt(pi))
    o[51] = (real_t)(-0.51891557872026028f*y*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + 5.0f*x4 + y4));                          // -3*sqrt(385)*y*(13*z2 - 1)*(-10*x2*y2 + 5*x4 + y4)/(64*sqrt(pi))
    o[52] = (real_t)(4.1513246297620823f*xy*z*(x2 - y2)*(13.0f*z2 - 3.0f));                           // 3*sqrt(385)*xy*z*(x2 - y2)*(13*z2 - 3)/(8*sqrt(pi))
    o[53] = (real_t)(-0.15645893386229404f*y*(3.0f*x2 - y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*y*(3*x2 - y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    o[54] = (real_t)(0.44253269244498261f*xy*z*(-110.0f*z2 + 143.0f*z4 + 15.0f));                              // 3*sqrt(70)*xy*z*(-110*z2 + 143*z4 + 15)/(32*sqrt(pi))
    o[55] = (real_t)(0.090331607582517306f*y*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*y*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    o[56] = (real_t)(0.068284276912004949f*z*(315.0f*z2 - 693.0f*z4 + 429.0f*z6 - 35.0f));                              // sqrt(15)*z*(315*z2 - 693*z4 + 429*z6 - 35)/(32*sqrt(pi))
    o[57] = (real_t)(0.090331607582517306f*x*(-135.0f*z2 + 495.0f*z4 - 429.0f*z6 + 5.0f));                              // sqrt(105)*x*(-135*z2 + 495*z4 - 429*z6 + 5)/(64*sqrt(pi))
    o[58] = (real_t)(0.07375544874083044f*z*(x2 - y2)*(143.0f*z2*(3.0f*z2 - 1.0f) - 187.0f*z2 + 45.0f));                         // sqrt(70)*z*(x2 - y2)*(143*z2*(3*z2 - 1) - 187*z2 + 45)/(64*sqrt(pi))
    o[59] = (real_t)(-0.15645893386229404f*x*(x2 - 3.0f*y2)*(13.0f*z2*(11.0f*z2 - 3.0f) - 27.0f*z2 + 3.0f));                              // -3*sqrt(35)*x*(x2 - 3*y2)*(13*z2*(11*z2 - 3) - 27*z2 + 3)/(64*sqrt(pi))
    o[60] = (real_t)(1.0378311574405206f*z*(13.0f*z2 - 3.0f)*(-6.0f*x2*y2 + x4 + y4));                         // 3*sqrt(385)*z*(13*z2 - 3)*(-6*x2*y2 + x4 + y4)/(32*sqrt(pi))
    o[61] = (real_t)(-0.51891557872026028f*x*(13.0f*z2 - 1.0f)*(-10.0f*x2*y2 + x4 + 5.0f*y4));                          // -3*sqrt(385)*x*(13*z2 - 1)*(-10*x2*y2 + x4 + 5*y4)/(64*sqrt(pi))
    o[62] = (real_t)(2.6459606618019f*z*(15.0f*x2*y4 - 15.0f*x4*y2 + x6 - y6));                               // 3*sqrt(10010)*z*(15*x2*y4 - 15*x4*y2 + x6 - y6)/(64*sqrt(pi))
    o[63] = (real_t)(0.70716273252459627f*x*(-35.0f*x2*y4 + 21.0f*x4*y2 - x6 + 7.0f*y6));
}

// kernel
template <typename real_t>
__global__ void spherical_harmonics_encoding_kernel(
    std::uint32_t n,
    std::uint32_t degree,
    float const *xyz,
    real_t * const __restrict__ output
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll
    for (int i = index; i < n; i += stride) {
        real_t * const o = output + i * degree * degree;
        float x = xyz[i*3], y = xyz[i*3+1], z = xyz[i*3+2];

        sh_enc<real_t>(degree, x, y, z, o);
    }
}

template <typename real_t>
void spherical_harmonics_encoding_launcher(cudaStream_t stream, void **buffers, char const *opaque, std::size_t opaque_len) {
    // inputs
    SphericalHarmonicsEncodingDescriptor const &desc =
        *deserialize<SphericalHarmonicsEncodingDescriptor>(opaque, opaque_len);
    std::uint32_t const n = desc.n;
    std::uint32_t const degree = desc.degree;
    float const *xyz = static_cast<float const *>(buffers[0]);  // [length, 3]

    // outputs
    real_t * const out = static_cast<real_t *>(buffers[1]);  // [length, dim * 2 * n_levels]

    int blockSize = 1024;
    int numBlocks = (n + blockSize - 1) / blockSize;
    numBlocks = std::min<int>(1024, numBlocks);
    spherical_harmonics_encoding_kernel<real_t><<<numBlocks, blockSize, 0, stream>>>(
            n,
            degree,
            xyz,
            out
        );
    check_throw(cudaGetLastError());
}

}  // namespace

void spherical_harmonics_encoding_cuda_f32(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
) {
    spherical_harmonics_encoding_launcher<float>(stream, buffers, opaque, opaque_len);
}

}  // namespace shjax
