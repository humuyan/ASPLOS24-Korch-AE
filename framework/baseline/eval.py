import sys
sys.path.insert(0, '../../..')

import sched
from google.protobuf.json_format import MessageToDict
import logging
import numpy as np
import onnx
import os
import argparse

import numpy as np  # type: ignore
import shutil

import pandas as pd

import tvm
from tvm.contrib import graph_executor
from tvm.contrib.graph_executor import GraphModule
from tvm import relay, runtime
from tvm import meta_schedule as ms
from tvm.relay.frontend import from_onnx
from tvm.meta_schedule.relay_integration import extract_tasks
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from utils import *

from kernel_profiler import benchmark_module, get_input_node_info

NUM_TRIALS_PER_ITER = 64
NUM_TRIALS = 25000
BACKEND = "graph"

rpc_host = "0.0.0.0"
rpc_port = 9090
rpc_key = "v100"


def get_input_node_info(onnx_model):
    # TVM from_onnx() requires shape_dict to be a dictionary of node name: List of dimensions
    shape_dict = {}
    input_name = ""
    DTYPE = ""
    input_shape = []

    for _input in onnx_model.graph.input:
        # ONNX format returns graph nodes as protobuf object
        m_dict = MessageToDict(_input)
        print("input_name : ", m_dict["name"])
        print("input_shape: ", m_dict["type"]["tensorType"]["shape"])
        print("input_dtype: ", m_dict["type"]["tensorType"]["elemType"])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]
        input_shape = [int(d.get("dimValue")) for d in dim_info]
        input_name = m_dict["name"]
        shape_dict[input_name] = input_shape

        # TODO: Convert enum elemType to required datatype
        DTYPE = (
            "float32" if m_dict["type"]["tensorType"]["elemType"] == 1 else "float32"
        )

    return shape_dict, input_name, input_shape, DTYPE


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    mod = GraphModule(rt_mod["default"](device))
    for input_name, input_value in input_data.items():
        mod.set_input(input_name, input_value)
    evaluator = mod.module.time_evaluator(
        "run",
        device,
        min_repeat_ms=500,
        repeat=3,
    )
    print(evaluator())


def profile_main(
    onnx_path="",
    onnx_model=None,
    WORK_DIR=None,
    DB_WIPE=True,
    USE_GE_BENCHMARK=True,
    BENCHMARK_NUM_RUNS=1,
    E2E=False,
):
    assert onnx_model is None and onnx_path != ""
    
    if onnx_model is None:
        onnx_model = onnx.load(onnx_path)
    
    shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)
    
    # TVM ONNX to TensorIR parser
    if MODEL == "candy-9":
        relay_mod, params = from_onnx(onnx_model)
    else:
        relay_mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)

    DEVICE = tvm.cuda()
    TARGET = tvm.target.cuda(
        arch="sm_70",
        options="-max_threads_per_block=1024 -max_shared_memory_per_block=96000",
    )
    
    input_data = tvm.nd.array(np.random.randn(*input_shape).astype(DTYPE))

    # if not MODEL.startswith("yolo"):
    #     relay_mod = convert_conv2d_layout(relay_mod, {"nn.conv2d": ["NHWC", "OHWI"]})
    # relay_mod = relay.transform.ToMixedPrecision("float16")(relay_mod)
    # relay_mod = rewrite_reshape_gelu(relay_mod)
    # tasks = extract_tasks(relay_mod, target=TARGET, params=params)

    rpc_confg = ms.runner.RPCConfig(
        tracker_host=rpc_host,
        tracker_port=rpc_port,
        tracker_key=rpc_key,
        session_timeout_sec=60,
    )
    runner = ms.runner.RPCRunner(rpc_confg)
    
    # db = ms.relay_integration.tune_relay(
    #     mod=relay_mod,
    #     params=params,
    #     target=TARGET,
    #     # config=ms.TuneConfig(
    #     #     strategy="evolutionary",
    #     #     num_trials_per_iter=NUM_TRIALS_PER_ITER,
    #     #     max_trials_per_task=NUM_TRIALS,
    #     #     max_trials_global=NUM_TRIALS,
    #     #     adaptive_training=ADAPTIVE_TRAINING,
    #     # ),
    #     strategy="evolutionary",
    #     num_trials_per_iter=NUM_TRIALS_PER_ITER,
    #     max_trials_per_task=NUM_TRIALS,
    #     max_trials_global=NUM_TRIALS,
    #     work_dir=WORK_DIR,
    #     runner=runner
    #     # backend=BACKEND
    # )
    path_tuning_record = os.path.join(WORK_DIR, "database_tuning_record.json")
    path_workload = os.path.join(WORK_DIR, "database_workload.json")
    db = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record)
    rt_mod = ms.relay_integration.compile_relay(
        mod=relay_mod, database=db, target=TARGET, params=params, backend=BACKEND
    )
    if USE_GE_BENCHMARK:
        print("Benchmarking using graph_executor.benchmark()")
        run_module_via_rpc(
            rpc_config=rpc_confg,
            lib=rt_mod,
            dev_type=TARGET.kind.name,
            args={input_name: input_data} if MODEL != "candy-9" else {},
            continuation=f_measurement,
        )
    else:
        print("Extracting from simulated annealing database")
        # # Extract from simulated annealing database
        # profile_results = {}
        # for i, task in enumerate(tasks):
        #     profile_results[task.task_name] = tuning_records[i].run_secs

    # if DB_WIPE:
    #     # Wiping all records before next kernel profiling
    #     os.remove("tune_kernels/database_workload.json")
    #     os.remove("tune_kernels/database_tuning_record.json")
    #     shutil.rmtree("tune_kernels/logs/")

    return profile_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="candy-9",
        help="Model to profile",
    )
    MODEL = parser.parse_args().model
    # onnx_path = "./models/" + MODEL + ".onnx"
    onnx_path = MODEL

    # USE_GE_BENCHMARK - True: Uses the graph_executor benchmark function, profiles BENCHMARK_NUM_RUNS times
    #                    False: Uses original simulated annealing measurement result
    # BENCHMARK_NUM_RUNS - Number of times to benchmark
    # E2E - Sets whether benchmarking should include data transfer overheads
    # DB_WIPE - True: Wipes the records clean every run
    profile_results = profile_main(
        onnx_path=onnx_path,
        WORK_DIR="./tune_kernels/" + MODEL + "/",
        DB_WIPE=False,
        USE_GE_BENCHMARK=True,
        BENCHMARK_NUM_RUNS=50,
        E2E=True,
    )

    print(profile_results)
