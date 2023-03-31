#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace shjax {

struct SphericalHarmonicsEncodingDescriptor {
    std::uint32_t n;
    std::uint32_t degree;
};

void spherical_harmonics_encoding_cuda_f32(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

}  // namespace shjax
