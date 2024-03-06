from compute_bound_profiler import profile_conv, profile_gemm
import onnx_graphsurgeon as gs
from kernel_profiler import profile_main
from math import prod
import subprocess
import os
import re
import onnx

null = open(os.devnull, "w")
pattern = re.compile(r"mean = (.*?) ms")

def trt_profile(onnx_path, trt_path="trt.log"):
    subprocess.run([
        "trtexec",
        "--separateProfileRun",
        f"--onnx={onnx_path}",
        "--dumpProfile",
        "--iterations=100",
        "--duration=0",
        "--device=1"],
        stdout=open(trt_path, "w"), stderr=null)
    for line in open(trt_path):
        if "GPU Compute Time: m" in line:
            return float(pattern.findall(line)[0])
    raise TypeError

def profile(onnx_path, params, DEVICE, TARGET, RPC_CONFIG, TARGET_DEVICE, WORK_DIR="./tune_kernels/", enable_trt=False) -> float:
    if not os.path.exists(WORK_DIR):
        os.makedirs(WORK_DIR)
    candidate_kernel = onnx.load(onnx_path)
    result = trt_profile(onnx_path, WORK_DIR + "trt.log") if enable_trt else 100000
    if params is None:
        tvm_result = list(profile_main(DEVICE, TARGET, RPC_CONFIG, WORK_DIR=WORK_DIR, onnx_model=candidate_kernel, DB_WIPE=False).values())
        if len(tvm_result) != 0:
            result = min(result, float(tvm_result[0][0]) * 1000)

    if params is not None and TARGET_DEVICE in ["v100", "a100"]:  # compute bound case
        type = params.pop("type")
        if type == "conv":
            for i in range(8):
                params["algo"] = i
                cur_t = profile_conv(**params)
                if cur_t > 0:
                    result = min(result, cur_t)
        elif type == "matmul":
            shape_a, shape_b = tuple(params["shapea"]), tuple(params["shapeb"])
            if len(shape_a) != len(shape_b):
                shape_a = (1, prod(shape_a[:-1]), shape_a[-1])
                shape_b = (1, shape_b[0], prod(shape_b[1:]))
            else:
                shape_a = (prod(shape_a[:-2]), shape_a[-2], shape_a[-1])
                shape_b = (prod(shape_b[:-2]), shape_b[-2], shape_b[-1])
                assert shape_a[0] == shape_b[0]
            
            time = profile_gemm(shape_a[0], shape_a[1], shape_b[2], shape_a[2], params["transa"], params["transb"], TARGET_DEVICE == "a100")
            result = min(time, result)

    return result
