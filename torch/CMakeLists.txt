# Check: https://github.com/godweiyang/NN-CUDA-Example
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(bt_dense LANGUAGES CXX CUDA)

find_package(Python REQUIRED)
find_package(CUDA REQUIRED)

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)

set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# modify to your own python path, or delete it if ok
include_directories(/opt/conda/envs/pt20/include/python3.9)
include_directories(../bytetransformer/include)
include_directories(include)
include_directories(..)
include_directories(../3rdparty/cutlass/include)
include_directories(../3rdparty/cutlass/tools/util/include)
include_directories(../cutlass_contrib/include)

set(SRCS
  gemm_ops.cc
  ../bytetransformer/src/gemm.cu
  ../bytetransformer/src/gemm_bias_act.cu
  ../bytetransformer/src/layernorm.cu
)
add_library(bt SHARED ${SRCS})

target_link_libraries(bt "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" -lcublas -lcudart)