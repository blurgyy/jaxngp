#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#undef __noinline__  // REF: <https://github.com/NVIDIA/thrust/issues/1703#issuecomment-1197072268>
#include <fmt/format.h>
#define __noinline__ noinline

template <class To, class From>
std::enable_if_t<
    sizeof(To) == sizeof(From) &&
    std::is_trivially_copyable_v<From> &&
    std::is_trivially_copyable_v<To>,
    To>
// constexpr support needs compiler magic
bit_cast(const From& src) noexcept
{
    static_assert(std::is_trivially_constructible_v<To>,
        "This implementation additionally requires "
        "destination type to be trivially constructible");
 
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

template <typename T>
std::string serialize(T const &descriptor) {
    return std::string{bit_cast<char const *>(&descriptor), sizeof(T)};
}

template <typename T>
T const *deserialize(char const *opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error(fmt::format("deserialize: Invalid opaque object size, expected {}, got {}", sizeof(T), opaque_len));
    }
    return bit_cast<T const *>(opaque);
}
