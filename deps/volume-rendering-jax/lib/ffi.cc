#include <cstdint>

#include <pybind11/pybind11.h>
#include <stdexcept>
#include <fmt/format.h>

#include "impl/integrating.h"
#include "serde.h"

namespace volrendjax {

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

    dict["integrate_rays"] = encapsulate_function(integrate_rays);
    dict["integrate_rays_backward"] = encapsulate_function(integrate_rays_backward);

    return dict;
}

PYBIND11_MODULE(integrating, m) {
    m.def("get_registrations", &get_registrations);
    m.def("make_ray_integrating_descriptor",
          [](std::uint32_t n_rays, std::uint32_t total_samples) {
            return to_pybind11_bytes(RayIntegratingDescriptor{
                .n_rays = n_rays,
                .total_samples = total_samples,
            });
          },
          "Description of the data passed to the `integrate_rays` or `integrate_rays_backward` function.\n\n"
          "Args:\n"
          "    n_rays: number of rays\n"
          "    total_samples: sum of number of samples on each ray\n"
          "\n"
          "Returns:\n"
          "    Serialized bytes that can be passed as the opaque parameter to `integrate_rays`\n"
          "    or `integrate_rays_backward`"
          );
};

}  // namespace

}  // namespace volrendjax
