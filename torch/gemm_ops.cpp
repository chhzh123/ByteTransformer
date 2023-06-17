#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include "../bytetransformer/include/gemm.h"

void bt_dense(const torch::Tensor &in,
              const torch::Tensor &weight,
              torch::Tensor &out) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
    bytetransformer::dense_layer_kernel_launcher((const __half *)in.data_ptr(),
                (const __half *)weight.data_ptr(),
                (__half *)out.data_ptr(),
                in.size(0),
                in.size(1),
                weight.size(1),
                cublas_handle,
                stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bt_dense",
          &bt_dense,
          "BT dense warpper");
}

TORCH_LIBRARY(bt_dense, m) {
    m.def("bt_dense", bt_dense);
}