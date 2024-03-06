import os

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.contrib import cublas
target = tvm.target.Target('cuda')

shape_A = (10, 32, 16384)
shape_B = (10, 16384, 33)
shape_C = (10, 32, 33)
A = te.placeholder(shape_A, name='A', dtype='float32')
B = te.placeholder(shape_B, name='kernel', dtype='float32')
C = cublas.batch_matmul(A, B, False, False, dtype='float32')

sch = te.create_schedule(C.op)
args = [A, B, C]
func = tvm.build(sch, args, target)

# Check correctness
data_np = np.random.uniform(size=shape_A).astype(np.float32)
weight_np = np.random.uniform(size=shape_B).astype(np.float32)

ctx = tvm.cuda()
data_tvm = tvm.nd.array(data_np, ctx)
weight_tvm = tvm.nd.array(weight_np, ctx)
out_tvm = tvm.nd.array(np.zeros(shape_C, dtype=C.dtype), ctx)
func(data_tvm, weight_tvm, out_tvm)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, number=100, repeat=10)
time = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results)
print("shape", data_np.shape, weight_np.shape)
print("Execution time of this operator: %.3f ms" % (time * 1000))
# print("Speed: %.3f TFLOPS" % (2 * (M*N*K) / time / 1e12))
