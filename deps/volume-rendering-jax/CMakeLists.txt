cmake_minimum_required(VERSION 3.23)
project(volume_rendering_jax LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "all")

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
    volrendutils_cuda
    ${CMAKE_CURRENT_LIST_DIR}/lib/impl/packbits.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/impl/marching.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/impl/integrating.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cc
)
target_link_libraries(volrendutils_cuda PRIVATE fmt::fmt)

install(TARGETS volrendutils_cuda DESTINATION volrendjax)
