#include <cstdint>

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <fmt/format.h>
#include <serde-helper/serde.h>

#include "impl/tcnnutils.h"

namespace jaxtcnn {

template <typename T>
pybind11::bytes to_pybind11_bytes(T const &descriptor) {
    return pybind11::bytes(serialize<T>(descriptor));
}

template <typename T>
pybind11::capsule encapsulate_function(T *fn) {
    return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

// expose gpu function
namespace {

pybind11::dict get_hashgrid_registrations() {
    pybind11::dict dict;
    dict["hashgrid_encode"] = encapsulate_function(hashgrid_encode);
    dict["hashgrid_encode_backward"] = encapsulate_function(hashgrid_encode_backward);
    return dict;
}

PYBIND11_MODULE(tcnnutils, m) {
    m.def("get_hashgrid_registrations", &get_hashgrid_registrations);
    m.def("make_hashgrid_descriptor",
          [](std::uint32_t const n_coords
             , std::uint32_t const L
             , std::uint32_t const F
             , std::uint32_t const N_min
             , float const per_level_scale) {
            return to_pybind11_bytes(HashGridDescriptor{
                .n_coords = n_coords,
                .L = L,
                .F = F,
                .N_min = N_min,
                .per_level_scale = per_level_scale
            });
          },
          "Static arguments passed to the `pack_density_into_bits` function.\n\n"
          "Args:\n"
          "    n_coords: number of input coordinates to be encoded\n"
          "    L: number of levels, 'n_levels' in tcnn\n"
          "    F: number of features that each level should output (2), 'n_features_per_level' in tcnn\n"
          "    N_min: coarsest resolution (16), 'base_resolution' in tcnn\n"
          "    per_level_scale: scale factor between consecutive levels\n"
          );
};

}  // namespace

}  // namespace jax_tcnn
