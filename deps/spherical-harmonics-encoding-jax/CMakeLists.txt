cmake_minimum_required(VERSION 3.23)
project(spherical_harmonics_encoding_jax LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "all")

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
    cudaops
    ${CMAKE_CURRENT_LIST_DIR}/lib/impl/spherical_harmonics_encoding.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cc
)
target_link_libraries(cudaops PRIVATE fmt::fmt)

install(TARGETS cudaops DESTINATION shjax)
