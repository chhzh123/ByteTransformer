#include "common.h"
#include "reduce.h"
#include "../../torch/include/bt.h"

namespace bytetransformer {

__inline__ __device__ void layernorm(float local_out, const void *gamma, const void *beta,
                                     float *out_ptr, int n, float *s_) {
  float sum = blockReduceSum<float>(local_out);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(sum, n);
  __syncthreads();

  local_out -= s_[0];
  float variance = blockReduceSum<float>(local_out * local_out);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  *out_ptr = local_out * s_[1] * __ldg(&((float *)gamma)[threadIdx.x]) +
             __ldg(&((float *)beta)[threadIdx.x]);
}

__inline__ __device__ void layernorm(half2 local_out, const void *gamma, const void *beta,
                                     half2 *out_ptr, int n, float *s_, bool use_fp32) {
  float2 local_out_fp2 = __half22float2(local_out);
  float t_sum = local_out_fp2.x + local_out_fp2.y;
  float sum = blockReduceSum<float>(t_sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(sum, n);
  __syncthreads();

  local_out_fp2.x -= s_[0];
  local_out_fp2.y -= s_[0];
  float variance = 0.0f;
  if (threadIdx.x < n / 2)
    variance = local_out_fp2.x * local_out_fp2.x + local_out_fp2.y * local_out_fp2.y;
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  if (threadIdx.x < n / 2) {
    float2 gamma_val, beta_val;
    if (use_fp32) {
      gamma_val = __ldg(&((const float2 *)gamma)[threadIdx.x]);
      beta_val = __ldg(&((const float2 *)beta)[threadIdx.x]);
    } else {
      gamma_val = __half22float2(__ldg(&((const half2 *)gamma)[threadIdx.x]));
      beta_val = __half22float2(__ldg(&((const half2 *)beta)[threadIdx.x]));
    }

    local_out_fp2.x = local_out_fp2.x * s_[1] * gamma_val.x + beta_val.x;
    local_out_fp2.y = local_out_fp2.y * s_[1] * gamma_val.y + beta_val.y;
    *out_ptr = __float22half2_rn(local_out_fp2);
  }
}

template <const int ite>
__inline__ __device__ void layernorm_v2(float *local_out, float sum, const void *gamma,
                                        const void *beta, float *out_ptr, int n, float *s_) {
  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(mean, n);
  __syncthreads();

  float var = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out[i] -= s_[0];
    var += local_out[i] * local_out[i];
  }

  float variance = blockReduceSum<float>(var);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    out_ptr[col_id] =
        local_out[i] * s_[1] * __ldg(&((float *)gamma)[col_id]) + __ldg(&((float *)beta)[col_id]);
  }
}

template <const int ite>
__inline__ __device__ void layernorm_v2(float2 *local_out_fp2, float sum, const void *gamma,
                                        const void *beta, half2 *out_ptr, int n, float *s_,
                                        bool use_fp32) {
  float mean = blockReduceSum<float>(sum);
  if (threadIdx.x == 0)
    s_[0] = __fdividef(mean, n);
  __syncthreads();

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x -= s_[0];
    local_out_fp2[i].y -= s_[0];
    variance += local_out_fp2[i].x * local_out_fp2[i].x + local_out_fp2[i].y * local_out_fp2[i].y;
  }

  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0)
    s_[1] = rsqrtf(__fdividef(variance, n) + 1e-6f);
  __syncthreads();

  float2 gamma_val[ite], beta_val[ite];
  if (use_fp32) {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      gamma_val[i] = __ldg(&((const float2 *)gamma)[col_id]);
      beta_val[i] = __ldg(&((const float2 *)beta)[col_id]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < ite; i++) {
      int col_id = i * blockDim.x + threadIdx.x;
      gamma_val[i] = __half22float2(__ldg(&((const half2 *)gamma)[col_id]));
      beta_val[i] = __half22float2(__ldg(&((const half2 *)beta)[col_id]));
    }
  }

#pragma unroll
  for (int i = 0; i < ite; i++) {
    local_out_fp2[i].x = local_out_fp2[i].x * s_[1] * gamma_val[i].x + beta_val[i].x;
    local_out_fp2[i].y = local_out_fp2[i].y * s_[1] * gamma_val[i].y + beta_val[i].y;
    out_ptr[i * blockDim.x + threadIdx.x] = __float22half2_rn(local_out_fp2[i]);
  }
}

template <const int ite>
__global__ void add_bias_input_layernorm_v2(float *out, const float *input, const float *bias,
                                            const void *gamma, const void *beta, int n,
                                            bool use_fp32) {
  int offset = blockIdx.x * n;

  float local_out[ite];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out[i] = (float)(out[id] + __ldg(&input[id]) + __ldg(&bias[col_id]));
    sum += local_out[i];
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out, sum, gamma, beta, out + offset, n, s_);
}

template <const int ite>
__global__ void add_bias_input_layernorm_v2(__half *out, const __half *input, const __half *bias,
                                            const void *gamma, const void *beta, int n,
                                            bool use_fp32) {
  half2 *out_ptr = (half2 *)out;
  const half2 *input_ptr = (const half2 *)input;
  const half2 *bias_ptr = (const half2 *)bias;

  int offset = blockIdx.x * n / 2;

  float sum = 0.0f;
  float2 local_out_fp2[ite];
#pragma unroll
  for (int i = 0; i < ite; i++) {
    int col_id = i * blockDim.x + threadIdx.x;
    int id = offset + col_id;
    local_out_fp2[i] = __half22float2(
        __hadd2(__hadd2(out_ptr[id], __ldg(&input_ptr[id])), __ldg(&bias_ptr[col_id])));
    sum += local_out_fp2[i].x + local_out_fp2[i].y;
  }

  __shared__ float s_[2];
  layernorm_v2<ite>(local_out_fp2, sum, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

__global__ void add_bias_input_layernorm(float *out, const float *input, const float *bias,
                                                const void *gamma, const void *beta, int n,
                                                bool use_fp32) {
  int offset = blockIdx.x * n + threadIdx.x;

  float local_out = (float)(out[offset] + __ldg(&input[offset]) + __ldg(&bias[threadIdx.x]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, out + offset, n, s_);
}

__global__ void add_bias_input_layernorm(__half *out, const __half *input,
                                                 const __half *bias, const void *gamma,
                                                 const void *beta, int n, bool use_fp32) {
  int offset = blockIdx.x * n / 2 + threadIdx.x;

  half2 local_out((__half)0.0f, (__half)0.0f);
  if (threadIdx.x < n / 2)
    local_out = __hadd2(__hadd2(((half2 *)out)[offset], __ldg(&((const half2 *)input)[offset])),
                        __ldg(&((const half2 *)bias)[threadIdx.x]));

  __shared__ float s_[2];
  layernorm(local_out, gamma, beta, ((half2 *)out) + offset, n, s_, use_fp32);
}

template <>
void add_bias_input_layernorm_kernel_launcher<float>(float *output, const float *input, const float *bias,
                                              const void *gamma, const void *beta, int m, int n,
                                              int hidden_dim, cudaStream_t stream, bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_input_layernorm_v2<2>
          <<<grid, block.x / 2, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
    else
      add_bias_input_layernorm_v2<4>
          <<<grid, block.x / 4, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
  } else {
    if (block.x < 32)
      block.x = 32;
    add_bias_input_layernorm<<<grid, block, 0, stream>>>(output, input, bias, gamma, beta, n,
                                                         use_fp32);
  }
}

template <>
void add_bias_input_layernorm_kernel_launcher<__half>(__half *output, const __half *input, const __half *bias,
                                              const void *gamma, const void *beta, int m, int n,
                                              int hidden_dim, cudaStream_t stream, bool use_fp32) {
  dim3 grid(m), block(hidden_dim);
  if (m >= 80 && n % 128 == 0) {
    if (n % 256 != 0)
      add_bias_input_layernorm_v2<2>
          <<<grid, block.x / 2, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
    else
      add_bias_input_layernorm_v2<4>
          <<<grid, block.x / 4, 0, stream>>>(output, input, bias, gamma, beta, n, use_fp32);
  } else {
    if (block.x < 32)
      block.x = 32;
    add_bias_input_layernorm<<<grid, block, 0, stream>>>(output, input, bias, gamma, beta, n,
                                                         use_fp32);
  }
}

} // namespace bytetransformer