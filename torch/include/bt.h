// Copyright 2023 Bytedance Ltd. and/or its affiliates.
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#pragma once
#include "../../bytetransformer/include/common.h"

namespace bytetransformer {
void dense_layer_kernel_launcher(const float *in, const float *weight, float *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = -1);

void dense_layer_kernel_launcher(const __half *in, const __half *weight, __half *out, const int M,
                                 const int K, const int N, cublasHandle_t cublas_handle,
                                 cudaStream_t stream, int cublasAlgo = 99);

template <typename T>
void gemm_bias_gelu(const T *A_, const T *B_, T *C_, const T *bias_, int m_, int k_, int n_,
                    cudaStream_t stream, cublasHandle_t cublas_handle, int cublasAlgo, int arch);

template <typename T>
void add_bias_input_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                              const void *gamma, const void *beta, int m, int n,
                                              int hidden_dim, cudaStream_t stream, bool use_fp32);

template <typename T>
void add_bias_input_out_layernorm_kernel_launcher(T *output, const T *input, const T *bias,
                                                  T *output2, const void *gamma, const void *beta,
                                                  int m, int n, int hidden_dim,
                                                  cudaStream_t stream, bool use_fp32);
}  // namespace bytetransformer
