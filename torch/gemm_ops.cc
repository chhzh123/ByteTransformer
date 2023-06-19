#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include "bt.h"

void dense(const torch::Tensor &in, const torch::Tensor &weight, torch::Tensor &out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  int n_dim = in.dim();
  int M = 1;
  for (int i = 0; i < n_dim - 1; ++i)
    M *= in.size(i);
  int K = in.size(n_dim - 1);
  int N = weight.size(weight.dim() - 1);
  // https://pytorch.org/cppdocs/notes/tensor_creation.html
  if (in.dtype() == torch::kFloat16) {
    bytetransformer::dense_layer_kernel_launcher(
        (const __half *)in.data_ptr(), (const __half *)weight.data_ptr(), (__half *)out.data_ptr(),
        M, K, N, cublas_handle, stream);
  } else {
    bytetransformer::dense_layer_kernel_launcher(
        (const float *)in.data_ptr(), (const float *)weight.data_ptr(), (float *)out.data_ptr(), M,
        K, N, cublas_handle, stream);
  }
}

void gemm_bias_gelu(const torch::Tensor &in, const torch::Tensor &weight, torch::Tensor &bias,
                    torch::Tensor &out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  int n_dim = in.dim();
  int M = 1;
  for (int i = 0; i < n_dim - 1; ++i)
    M *= in.size(i);
  int K = in.size(n_dim - 1);
  int N = weight.size(weight.dim() - 1);
  if (in.dtype() == torch::kFloat16) {
    bytetransformer::gemm_bias_gelu(
        (const __half *)in.data_ptr(), (const __half *)weight.data_ptr(), (__half *)out.data_ptr(),
        (const __half *)bias.data_ptr(), M, K, N, stream, cublas_handle, 99, 70);
  } else {
    bytetransformer::gemm_bias_gelu((const float *)in.data_ptr(), (const float *)weight.data_ptr(),
                                    (float *)out.data_ptr(), (const float *)bias.data_ptr(), M, K,
                                    N, stream, cublas_handle, -1 /*float algo*/, 70);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense", &dense, "BT dense warpper");
  m.def("gemm_bias_gelu", &gemm_bias_gelu, "BT gemm bias gelu warpper");
}

TORCH_LIBRARY(bt, m) {
  m.def("dense", dense);
  m.def("gemm_bias_gelu", gemm_bias_gelu);
}