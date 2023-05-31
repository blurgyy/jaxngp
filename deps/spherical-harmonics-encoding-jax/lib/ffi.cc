#include <cstdint>
#include <stdexcept>

#include <fmt/format.h>
#include <pybind11/pybind11.h>
#include <serde-helper/serde.h>

#include "impl/spherical_harmonics_encoding.h"

namespace shjax {

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

pybind11::dict get_registrations() {
    pybind11::dict dict;

    dict["spherical_harmonics_encoding_cuda_f32"] = encapsulate_function(spherical_harmonics_encoding_cuda_f32);

    return dict;
}

PYBIND11_MODULE(cudaops, m) {
    m.def("get_registrations", &get_registrations);
    m.def("make_spherical_harmonics_encoding_descriptor",
          [](std::uint32_t n, std::uint8_t degree) {
            if (degree < 1 || degree > 8) {
                throw std::runtime_error(fmt::format("Spherical harmonics encoding supports degrees 1 to 8 (inclusive) but got {}", degree));
            }
            return to_pybind11_bytes(SphericalHarmonicsEncodingDescriptor{
                .n = n,
                .degree = degree,
            });
          },
          "Description of the data passed to the spherical harmonics encoding function.\n\n"
          "Args:\n"
          "    n: specifies how many inputs are to be encoded\n"
          "    degree: specifies the highest degree of the output encoding, supports integers 1..8\n"
          "\n"
          "Returns:\n"
          "    Serialized bytes that can be passed as the opaque parameter for spherical_harmonics_encoding_cuda"
          );
};

}  // namespace

}  // namespace shjax
