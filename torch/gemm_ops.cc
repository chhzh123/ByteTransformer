#include <torch/extension.h>
#include "ATen/cuda/CUDAContext.h"
#include "bt.h"

torch::Tensor dense(const torch::Tensor &in, const torch::Tensor &weight) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  auto out = torch::empty({in.size(0), weight.size(1)}, in.options());
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
  return out;
}

torch::Tensor gemm_bias_gelu(const torch::Tensor &in, const torch::Tensor &weight,
                             const torch::Tensor &bias) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  cublasHandle_t cublas_handle = at::cuda::getCurrentCUDABlasHandle();
  std::vector<int64_t> out_size = in.sizes().vec();
  out_size[out_size.size() - 1] = weight.size(1);
  auto out = torch::empty(out_size, in.options());
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
  return out;
}

torch::Tensor add_bias_layernorm(const torch::Tensor &in, const torch::Tensor &bias,
                                 const torch::Tensor &gamma, const torch::Tensor &beta) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto out = torch::zeros_like(in);
  int n_dim = in.dim();
  int M = 1;
  for (int i = 0; i < n_dim - 1; ++i)
    M *= in.size(i);
  int N = in.size(n_dim - 1);
  if (in.dtype() == torch::kFloat16) {
    // see bert_transformer.cu
    int hidden_dim = N / 2;
    bytetransformer::add_bias_input_layernorm_kernel_launcher(
        (__half *)out.data_ptr(), (const __half *)in.data_ptr(), (const __half *)bias.data_ptr(),
        (const void *)gamma.data_ptr(), (const void *)beta.data_ptr(), M, N, hidden_dim, stream,
        false);
  } else {
    int hidden_dim = N;
    bytetransformer::add_bias_input_layernorm_kernel_launcher(
        (float *)out.data_ptr(), (const float *)in.data_ptr(), (const float *)bias.data_ptr(),
        (const void *)gamma.data_ptr(), (const void *)beta.data_ptr(), M, N, hidden_dim, stream,
        true);
  }
  return out;
}

torch::Tensor add_residual_bias_layernorm(torch::Tensor &in, const torch::Tensor &residual,
                                          const torch::Tensor &bias, const torch::Tensor &gamma,
                                          const torch::Tensor &beta) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  auto out2 = torch::zeros_like(in);
  int n_dim = in.dim();
  int M = 1;
  for (int i = 0; i < n_dim - 1; ++i)
    M *= in.size(i);
  int N = in.size(n_dim - 1);
  if (in.dtype() == torch::kFloat16) {
    // see bert_transformer.cu
    int hidden_dim = N / 2;
    bytetransformer::add_bias_input_out_layernorm_kernel_launcher(
        (__half *)in.data_ptr(), (const __half *)residual.data_ptr(),
        (const __half *)bias.data_ptr(), (__half *)out2.data_ptr(), (const void *)gamma.data_ptr(),
        (const void *)beta.data_ptr(), M, N, hidden_dim, stream, false);
  } else {
    int hidden_dim = N;
    bytetransformer::add_bias_input_out_layernorm_kernel_launcher(
        (float *)in.data_ptr(), (const float *)residual.data_ptr(), (const float *)bias.data_ptr(),
        (float *)out2.data_ptr(), (const void *)gamma.data_ptr(), (const void *)beta.data_ptr(),
        M, N, hidden_dim, stream, true);
  }
  return in;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense", &dense, "BT dense warpper");
  m.def("gemm_bias_gelu", &gemm_bias_gelu, "BT gemm bias gelu warpper");
  m.def("add_bias_layernorm", &add_bias_layernorm, "BT add bias layernorm warpper");
  m.def("add_residual_bias_layernorm", &add_residual_bias_layernorm,
        "BT add residual bias layernorm warpper");
}

TORCH_LIBRARY(bt, m) {
  m.def("dense", dense);
  m.def("gemm_bias_gelu", gemm_bias_gelu);
  m.def("add_bias_layernorm", add_bias_layernorm);
  m.def("add_residual_bias_layernorm", add_residual_bias_layernorm);
}