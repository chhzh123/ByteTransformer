import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

bs = 1
seq = 512
hs = 1024
# M = bs * seq
# K = hs
# N = hs * 3
dtype = torch.float16
a = torch.rand((bs, seq, hs), dtype=dtype, device="cuda:0")
b = torch.rand((hs * 4, hs), dtype=dtype, device="cuda:0")
bias = torch.rand((hs * 4,), dtype=dtype, device="cuda:0")

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
    return torch.ops.bt.gemm_bias_gelu(a, b, bias)


def run_torch():
    c = F.linear(a, b, bias)
    d = F.gelu(c)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.ops.load_library("build/libbt.so")

    print("Running torch...")
    torch_time, torch_res = show_time(run_torch)
    print("Torch time:  {:.3f}us".format(np.mean(torch_time)))

    print("Running cuda...")
    # Need to prepare the weight in advance
    b = b.transpose(1, 0).contiguous()
    cuda_time, cuda_res = show_time(run_cuda)
    print("Cuda time:  {:.3f}us".format(np.mean(cuda_time)))

    print(cuda_res)
    print(torch_res)
    torch.testing.assert_close(cuda_res, torch_res)
    print("Kernel test passed.")
