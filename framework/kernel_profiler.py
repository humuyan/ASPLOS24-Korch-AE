#%%
import sched
import sys
from google.protobuf.json_format import MessageToDict
import logging
import numpy as np
import onnx
import os

import pandas as pd

import shutil

from utils import configure_target

import tvm
from tvm import rpc
from tvm import relay
from tvm.tir import Schedule

from tvm.contrib import graph_executor
from tvm.ir import IRModule

from tvm.ir.transform import PassContext
from tvm.meta_schedule.database.json_database import JSONDatabase
from tvm.relay.frontend import from_onnx
from tvm.runtime import Module, NDArray, vm
from tvm.support import describe
from tvm.target import Target

from tvm import meta_schedule as ms
from tvm.relay.frontend import from_onnx
from tvm.support import describe

from tvm.meta_schedule.relay_integration import extracted_tasks_to_tune_contexts, extract_tasks

# %%
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

NUM_RUNS = 20
NUM_REPEAT = 3
MIN_REPEAT_MS = 150
CPU_FLUSH = True

NUM_TRIALS_PER_ITER = 64
NUM_TRIALS = 1024
ADAPTIVE_TRAINING = True
BACKEND = "graph"

NUM_RUNS = 2
NUM_REPEAT = 1
MIN_REPEAT_MS = 150
CPU_FLUSH = True

NUM_TRIALS_PER_ITER = 3
NUM_TRIALS = 10
ADAPTIVE_TRAINING = True
BACKEND = "graph"

# describe()

ndk_path = '/opt/android-sdk-linux/ndk/24.0.8215888'
clang_path = os.path.join(ndk_path, 'toolchains', 'llvm', 'prebuilt', 'linux-x86_64', 'bin', 'aarch64-linux-android31-clang++')

os.environ['TVM_NDK_CC'] = clang_path
    
# %%
def get_input_node_info(onnx_model):
    # TVM from_onnx() requires shape_dict to be a dictionary of node name: List of dimensions
    shape_dict = {}
    input_name = ""
    DTYPE = ""
    input_shape = []
    
    for _input in onnx_model.graph.input:
        # ONNX format returns graph nodes as protobuf object
        m_dict = MessageToDict(_input)
        print("input_name : ", m_dict['name'])
        print("input_shape: ", m_dict["type"]["tensorType"]['shape'])
        print("input_dtype: ", m_dict["type"]["tensorType"]['elemType'])
        dim_info = m_dict["type"]["tensorType"]["shape"]["dim"]
        input_shape = [int(d.get("dimValue")) for d in dim_info]
        input_name = m_dict["name"]
        shape_dict[input_name] = input_shape
        
        # TODO: Convert enum elemType to required datatype
        DTYPE = "float32" if m_dict["type"]["tensorType"]['elemType'] == 1 else "float32"
        
    return shape_dict, input_name, input_shape, DTYPE


def get_output(data, lib, dev, input_name):
        module = graph_executor.GraphModule(lib["default"](dev))
        module.set_input(input_name, data)
        module.run()
        return module.get_output(0).numpy()

# %%
def benchmark_module(data, rt_mod: Module, dev, input_name, numRuns, end_to_end=False):
    module = graph_executor.GraphModule(rt_mod["default"](dev))
    module.set_input(input_name, data)
    results_df = pd.DataFrame(columns=["Mean", "Std_Dev", "Median", "Max", "Min"])
    
    # rt_mod.lib.entry_func()
    timing_results = module.benchmark(device=dev, func_name='run', number=numRuns, end_to_end=end_to_end)
    
    results_df.loc[len(results_df.index)] = [timing_results.mean, timing_results.std, timing_results.median, timing_results.max, timing_results.min]
    
    return results_df


def generate_cuda_code(DATABASE, MODEL_FILENAME, tuning_records, WORK_DIR):
    path_lib = WORK_DIR + "lib_export/" + MODEL_FILENAME
    
    # Assumes only one record exists in the database. Unable to query database properly with initial IRModule
    # Database appears to store transformed IRModule from tune_relay() that is unavailable to top level script
    best_schedule = DATABASE.query(tuning_records[0].workload.mod, target=TARGET, kind="schedule").mod

    built_mod = tvm.build(inputs=best_schedule, target=TARGET, name="main")
    
    # Exports the compiied binary as a shared library object
    built_mod.export_library(path_lib + ".so", cc="nvcc")

    # Exports the lowered CUDA code as a .cu file
    dev_module = built_mod.imported_modules[0]
    
    print("Saving " + MODEL_FILENAME + ".cu" + " at path: " + path_lib + ".cu\n")
    
    with open(path_lib + ".cu", "w") as outfile:
        outfile.write(dev_module.get_source())
        
    outfile.close()


# %%
def profile_main(DEVICE, TARGET, RPC_CONFIG, onnx_path="", onnx_model=None, WORK_DIR="./tune_kernels/",
                 DB_WIPE=True, USE_GE_BENCHMARK=True, BENCHMARK_NUM_RUNS= 1, E2E=False, CHECK_CORRECTNESS=False,
                 SAVE_LIB=False, MODEL_FILENAME="kernel"):
    """_summary_

    Args:
        onnx_path (str): The path to the candidate cases stored as onnx files
        onnx_model (ModelProto, optional): Pass in an onnx model object directly is desired
        WORK_DIR (str, optional): location to store kernel tuning logs
        DB_WIPE (bool, optional): To disable wiping of database every iteration of kernel profiler. Defaults to True.
        TODO: The wiping of database is temporary until a better solution is found to reconcile task.task_name and Workload hash

    Returns:
        _type_: _description_
    """
    assert onnx_model is not None or onnx_path != ""

    profile_results = {}

    if onnx_model is None:
        onnx_model = onnx.load(onnx_path)

    shape_dict, input_name, input_shape, DTYPE = get_input_node_info(onnx_model)

    # shape_dict = {'data': [1, 3, 224, 224]}

    # TVM ONNX to TensorIR parser
    mod, params = from_onnx(onnx_model, shape_dict, freeze_params=True)

    # Reference: https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # Page 22 Compute Capabilities table

    data = tvm.nd.array(np.random.randn(*input_shape).astype(DTYPE), DEVICE)

    # extracted_tasks = extract_task_from_relay(mod, TARGET, params)
    extracted_tasks = ms.relay_integration.extract_tasks(mod, target=TARGET, params=params)
    tir_mod = extracted_tasks[0].dispatched[0]
    # Disallow profiler to continue if extracted_tasks > 1
    if len(extracted_tasks) > 1:
        print("Number of extracted tasks > 1. Returning empty profile_results dictionary")
        return profile_results
    
    DATABASE = JSONDatabase(path_workload=WORK_DIR+"database_workload.json", path_tuning_record=WORK_DIR+"database_tuning_record.json", work_dir=WORK_DIR)
    if(DATABASE.has_workload(tir_mod)):
        print("Fetching results from database")
        tuning_record = DATABASE.query_tuning_record(tir_mod, TARGET, "")
        profile_results = {}
        try:
            profile_results[extracted_tasks[0].task_name] = tuning_record.run_secs
        except AttributeError:
            print("Profile results  corrupted!!")
        return profile_results
    
    else:
        print("Task not found in database; performing kernel profiling")
        # Return to top level: latency and CUDA code file location
        with ms.Profiler() as profiler:
            if(len(RPC_CONFIG) > 0):
                rpc_confg = ms.runner.RPCConfig(
                    tracker_host=RPC_CONFIG[0],
                    tracker_port=RPC_CONFIG[1],
                    tracker_key=RPC_CONFIG[2],
                    session_timeout_sec=60,
                )
                runner = ms.runner.RPCRunner(rpc_confg)
            else:
                runner = "local"
            
            print("runner: ", runner)
                
            DATABASE = ms.relay_integration.tune_relay(
                mod=mod,
                params=params,
                target=TARGET,
                database=DATABASE,
                strategy="evolutionary",
                num_trials_per_iter=NUM_TRIALS_PER_ITER,
                max_trials_per_task=NUM_TRIALS,
                max_trials_global=NUM_TRIALS,
                work_dir=WORK_DIR,
                runner=runner
                )
            
        rt_mod1 = ms.relay_integration.compile_relay(
            database=DATABASE,
            mod=mod,
            target=TARGET,
            params=params,
            backend=BACKEND,
        )
        
        # Build original graph without optimizations
        with tvm.transform.PassContext(opt_level=0):
            rt_mod2 = relay.build(mod, target=TARGET, params=params)

        # Check correctness
        if(CHECK_CORRECTNESS):
            actual_output = get_output(data, rt_mod1, DEVICE, input_name)
            expected_output = get_output(data, rt_mod2, DEVICE, input_name)
            assert np.allclose(actual_output, expected_output, rtol=1e-4, atol=2e-4, equal_nan=True)
    
        # Extract tuning records from database file
        tuning_records = DATABASE.get_all_tuning_records()
        
        # Extract from simulated annealing database
        profile_results = {}
        for task in extracted_tasks:
            record = DATABASE.query_tuning_record(task.dispatched[0], TARGET, "")
            profile_results[task.task_name] = record.run_secs
                

        return profile_results

        # if USE_GE_BENCHMARK:
        #     print("Benchmarking using graph_executor.benchmark()")
        #     # Benchmark using graph_executor.benchmark()
        #     profile_results = benchmark_module(data, rt_mod1, DEVICE, input_name, BENCHMARK_NUM_RUNS, E2E)
        # else:
        #     print("Extracting from simulated annealing database")
        #     # Extract from simulated annealing database
        #     profile_results = {}
        #     for i, task in enumerate(extracted_tasks):
        #         profile_results[task.task_name] = tuning_records[i].run_secs
        
        # if SAVE_LIB:
        #     # Save library object as shared library file
        #     # Extract CUDA source file and save to file
        #     if not os.path.exists(WORK_DIR + "lib_export/"): os.makedirs(WORK_DIR + "lib_export/")
            
        #     generate_cuda_code(DATABASE, MODEL_FILENAME, tuning_records, WORK_DIR)
        
        
        # if DB_WIPE:
        #     # Wiping all records before next kernel profiling
        #     os.remove("tune_kernels/database_workload.json")
        #     os.remove("tune_kernels/database_tuning_record.json")
        #     shutil.rmtree('tune_kernels/logs/')
            
        # return profile_results, rt_mod1, DATABASE


if __name__=="__main__":
    model_name = "vgg16-7"
    onnx_path = "../cases/mobile_cases/" + model_name + ".onnx"

    if(len(sys.argv) == 2):
        TARGET_DEVICE = sys.argv[1]
    else:
        TARGET_DEVICE = 'cuda'
        
    DEVICE, TARGET, RPC_CONFIG = configure_target(TARGET_DEVICE)
    
    print("device ", DEVICE)
    # print("remote", remote)
    print("target ", TARGET)
    print("rpc_config ", RPC_CONFIG)
    # USE_GE_BENCHMARK - True: Uses the graph_executor benchmark function, profiles BENCHMARK_NUM_RUNS times
    #                    False: Uses original simulated annealing measurement result
    # BENCHMARK_NUM_RUNS - Number of times to benchmark
    # E2E - Sets whether benchmarking should include data transfer overheads
    # DB_WIPE - True: Wipes the records clean every run
    
    
    # TARGET = tvm.target.Target("llvm -num-cores 8 -keys=arm_cpu,cpu -mtriple=aarch64-linux-android")
    profile_results = profile_main(DEVICE, TARGET, RPC_CONFIG, onnx_path=onnx_path, DB_WIPE=False,
                                   USE_GE_BENCHMARK=False, BENCHMARK_NUM_RUNS= 50, E2E=True)

    print(profile_results)

# %%
