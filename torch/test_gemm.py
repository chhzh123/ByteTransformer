import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# c = a + b (shape: [n])
M, N, K = 1024, 1024, 1024
# M, N, K = 16, 16, 16
dtype = torch.float32
a = torch.rand((M, K), dtype=dtype, device="cuda:0")
b = torch.rand((K, N), dtype=dtype, device="cuda:0")
bias = torch.rand((N,), dtype=dtype, device="cuda:0")
cuda_c = torch.zeros((M, N), dtype=dtype, device="cuda:0")

ntest = 100


def show_time(func):
    times = list()
    res = None
    # GPU warm up
    for _ in range(10):
        res = func()
    for _ in range(ntest):
        # sync the threads to get accurate cuda running time
        torch.cuda.synchronize(device="cuda:0")
        start_time = time.time()
        func()
        torch.cuda.synchronize(device="cuda:0")
        end_time = time.time()
        times.append((end_time - start_time) * 1e6)
    return times, res


def run_cuda():
    torch.ops.bt.dense(a, b, cuda_c)
    return cuda_c
    # torch.ops.bt.gemm_bias_gelu(a, b, bias, cuda_c)
    # return cuda_c


def run_torch():
    return torch.matmul(a, b)
    c = F.linear(a, b.transpose(1, 0), bias)
    d = F.gelu(c)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.ops.load_library("build/libbt.so")
    cuda_res = run_cuda()
    torch_res = run_torch()
    print(cuda_res)
    print(torch_res)
    torch.testing.assert_close(cuda_res, torch_res)
    print("Kernel test passed.")

    print("Running cuda...")
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))
