cmake_minimum_required(VERSION 3.23)
project(volume_rendering_jax LANGUAGES CXX CUDA)
# use `cmake -DCMAKE_CUDA_ARCHITECTURES=61;62;75` to build for compute capabilities 61, 62, and 75
# set(CMAKE_CUDA_ARCHITECTURES "all")
message(STATUS "Enabled CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
pybind11_add_module(
    tcnnutils
    ${CMAKE_CURRENT_LIST_DIR}/lib/impl/hashgrid.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cc
)

# e.g. `cmake -DTCNN_MIN_GPU_ARCH=61`
message(STATUS "TCNN_MIN_GPU_ARCH=${TCNN_MIN_GPU_ARCH}")
target_compile_definitions(tcnnutils PUBLIC -DTCNN_MIN_GPU_ARCH=${TCNN_MIN_GPU_ARCH})

target_link_libraries(tcnnutils PRIVATE tiny-cuda-nn fmt::fmt)

install(TARGETS tcnnutils DESTINATION jaxtcnn)
