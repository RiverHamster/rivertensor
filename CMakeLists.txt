cmake_minimum_required(VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 80)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# set(PYBIND11_NEWPYTHON ON)

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

pybind11_add_module(base src/pybind.cc src/tensor-numpy.cc)
target_link_libraries(base PUBLIC tensor)
target_include_directories(base PUBLIC include)
install(TARGETS base DESTINATION rivertensor)

# C++ test binary
add_executable(tensor_test test/tensor_test.cc)
target_include_directories(tensor_test PUBLIC include)
target_link_libraries(tensor_test PUBLIC tensor)
