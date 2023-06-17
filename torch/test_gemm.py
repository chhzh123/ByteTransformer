import time
import argparse
import numpy as np
import torch

# c = a + b (shape: [n])
# M, N, K = 512, 1024, 512
M, N, K = 16, 16, 16
a = torch.rand((M, K), dtype=torch.float16, device="cuda:0")
b = torch.rand((K, N), dtype=torch.float16, device="cuda:0")
cuda_c = torch.zeros((M, N), dtype=torch.float16, device="cuda:0")

ntest = 10

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
        times.append((end_time-start_time)*1e6)
    return times, res

def run_cuda():
    torch.ops.bt_dense.bt_dense(a, b, cuda_c)
    return cuda_c

def run_torch():
    c = torch.matmul(a, b)
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    torch.ops.load_library("build/libbt_dense.so")
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
