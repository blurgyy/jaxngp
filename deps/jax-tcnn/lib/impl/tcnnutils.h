#include <cstdint>
#include <driver_types.h>

#ifdef __CUDACC__
#include <tiny-cuda-nn/encodings/grid.h>
#endif


namespace jaxtcnn {

struct HashGridDescriptor {
    // number of input coordinates to be encoded
    std::uint32_t n_coords;

    // number of levels (2), "n_levels" in tcnn
    std::uint32_t L;

    // number of features that each level should output (2), "n_features_per_level" in tcnn
    std::uint32_t F;

    // coarsest resolution (16), "base_resolution" in tcnn
    std::uint32_t N_min;

    // scale factor between consecutive levels
    // float const per_level_scale() const {
    //     return std::exp2f(std::log2f(this->N_max) - std::log2f(this->N_min)) / (this->L - 1);
    // }
    float per_level_scale;
};

void hashgrid_encode(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

void hashgrid_encode_backward(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

void hashgrid_encode_inference(
    cudaStream_t stream,
    void **buffers,
    const char *opaque,
    std::size_t opaque_len
);

}  // namespace jaxtcnn
