cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_FINDPYTHON ON)
add_subdirectory(extern/pybind11)
set(CMAKE_CUDA_ARCHITECTURES 80)
project(Tensor VERSION 0.1.0
               DESCRIPTION "Tensor library for the PKU Programming in AI course"
               LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(tensor STATIC src/tensor.cu src/tensor-init.cu src/tensor-arith.cu
            src/tensor-reduce.cu src/matmul.cu src/conv2d.cu src/conv2d-back.cu
            src/pooling.cu src/cross-entropy.cu)
target_include_directories(tensor PUBLIC include)
set_target_properties(tensor PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(tensor PUBLIC CUDA::curand CUDA::cublas)
target_compile_options(tensor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --extended-lambda
                       >)

pybind11_add_module(pytensor src/pybind.cc src/tensor-numpy.cc)
target_link_libraries(pytensor PUBLIC tensor)
target_include_directories(pytensor PUBLIC include)
install(TARGETS pytensor DESTINATION .)

# C++ test binary
add_executable(tensor_test test/tensor_test.cc)
target_include_directories(tensor_test PUBLIC include)
target_link_libraries(tensor_test PUBLIC tensor)
