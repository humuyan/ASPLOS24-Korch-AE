import sched
from google.protobuf.json_format import MessageToDict
import logging
import numpy as np
import onnx
import os

import numpy as np  # type: ignore
import shutil

import pandas as pd

import tvm

# from tvm.contrib import graph_executor
# from tvm.contrib.graph_executor import GraphModule
# from tvm import relay, runtime
# from tvm import meta_schedule as ms
# from tvm.relay.frontend import from_onnx
# from tvm.meta_schedule.relay_integration import extract_tasks
# from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from codes.utils import *

from tvm.script import tir as T


# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(p0: T.Buffer[(T.int64(12), T.int64(16384), T.int64(16)), "float32"], p1: T.Buffer[(T.int64(12), T.int64(1), T.int64(16)), "float32"], T_batch_matmul_NT: T.Buffer[(T.int64(12), T.int64(16384), T.int64(1)), "float32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True, "layout_free_buffers": [1]})
        # body
        # with T.block("root")
        for b, i, j, k in T.grid(T.int64(12), T.int64(16384), T.int64(1), T.int64(16)):
            with T.block("T_batch_matmul_NT"):
                v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k])
                T.reads(p0[v_b, v_i, v_k], p1[v_b, v_j, v_k])
                T.writes(T_batch_matmul_NT[v_b, v_i, v_j])
                with T.init():
                    T_batch_matmul_NT[v_b, v_i, v_j] = T.float32(0)
                T_batch_matmul_NT[v_b, v_i, v_j] = T_batch_matmul_NT[v_b, v_i, v_j] + p0[v_b, v_i, v_k] * p1[v_b, v_j, v_k]


sch = tvm.tir.Schedule(Module, debug_mask="all")

b0 = sch.get_block(name="T_batch_matmul_NT", func_name="main")
b1 = sch.get_block(name="root", func_name="main")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
l2, l3, l4, l5 = sch.get_loops(block=b0)
v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 1, 2, 6, 1])
l11, l12, l13, l14, l15 = sch.split(loop=l2, factors=[v6, v7, v8, v9, v10], preserve_unit_iters=True)
v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 16, 64, 1])
l21, l22, l23, l24, l25 = sch.split(loop=l3, factors=[v16, v17, v18, v19, v20], preserve_unit_iters=True)
v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
l31, l32, l33, l34, l35 = sch.split(loop=l4, factors=[v26, v27, v28, v29, v30], preserve_unit_iters=True)
v36, v37, v38 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64, decision=[16, 1, 1])
l39, l40, l41 = sch.split(loop=l5, factors=[v36, v37, v38], preserve_unit_iters=True)
sch.reorder(l11, l21, l31, l12, l22, l32, l13, l23, l33, l39, l40, l14, l24, l34, l41, l15, l25, l35)
l42 = sch.fuse(l11, l21, l31, preserve_unit_iters=True)
sch.bind(loop=l42, thread_axis="blockIdx.x")
l43 = sch.fuse(l12, l22, l32, preserve_unit_iters=True)
sch.bind(loop=l43, thread_axis="vthread.x")
l44 = sch.fuse(l13, l23, l33, preserve_unit_iters=True)
sch.bind(loop=l44, thread_axis="threadIdx.x")
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
sch.annotate(block_or_loop=b0, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
b45 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
sch.reverse_compute_at(block=b45, loop=l44, preserve_unit_loops=True, index=-1)
b46 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b0])
sch.compute_at(block=b46, loop=l39, preserve_unit_loops=True, index=-1)
l47, l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b46)
l54 = sch.fuse(l51, l52, l53, preserve_unit_iters=True)
v55 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
sch.annotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch", ann_val=v55)
b56 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b0])
sch.compute_at(block=b56, loop=l39, preserve_unit_loops=True, index=-1)
l57, l58, l59, l60, l61, l62, l63 = sch.get_loops(block=b56)
l64 = sch.fuse(l61, l62, l63, preserve_unit_iters=True)
v65 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
sch.annotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
v66 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v66)
sch.enter_postproc()
sch.unannotate(block_or_loop=b46, ann_key="meta_schedule.cooperative_fetch")
l67, l68, l69, l70, l71 = sch.get_loops(block=b46)
l72, l73, l74 = sch.split(loop=l71, factors=[None, 32, 3], preserve_unit_iters=True)
sch.vectorize(loop=l74)
sch.bind(loop=l73, thread_axis="threadIdx.x")
sch.unannotate(block_or_loop=b56, ann_key="meta_schedule.cooperative_fetch")
l75, l76, l77, l78, l79 = sch.get_loops(block=b56)
l80, l81, l82 = sch.split(loop=l79, factors=[None, 32, 2], preserve_unit_iters=True)
sch.vectorize(loop=l82)
sch.bind(loop=l81, thread_axis="threadIdx.x")
b83 = sch.get_block(name="root", func_name="main")
sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.unroll_explicit")
b84, b85, b86, b87 = sch.get_child_blocks(b83)
l88, l89, l90, l91, l92, l93, l94 = sch.get_loops(block=b84)
sch.annotate(block_or_loop=l88, ann_key="pragma_auto_unroll_max_step", ann_val=16)
sch.annotate(block_or_loop=l88, ann_key="pragma_unroll_explicit", ann_val=1)
l95, l96, l97, l98, l99, l100, l101 = sch.get_loops(block=b85)
sch.annotate(block_or_loop=l95, ann_key="pragma_auto_unroll_max_step", ann_val=16)
sch.annotate(block_or_loop=l95, ann_key="pragma_unroll_explicit", ann_val=1)
l102, l103, l104, l105, l106, l107, l108, l109, l110, l111, l112, l113 = sch.get_loops(block=b86)
sch.annotate(block_or_loop=l102, ann_key="pragma_auto_unroll_max_step", ann_val=16)
sch.annotate(block_or_loop=l102, ann_key="pragma_unroll_explicit", ann_val=1)
l114, l115, l116, l117, l118, l119 = sch.get_loops(block=b87)
sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=16)
sch.annotate(block_or_loop=l114, ann_key="pragma_unroll_explicit", ann_val=1)
b120 = sch.get_block(name="T_batch_matmul_NT", func_name="main")
l121, l122, l123, l124, l125, l126, l127, l128, l129, l130, l131, l132 = sch.get_loops(block=b120)
b133 = sch.decompose_reduction(block=b120, loop=l124)

print(sch.mod.script())
# print("OK")
f = tvm.build(sch.mod, target="cuda")
print(f.imported_modules[0].get_source())

device = tvm.cuda(0)
p0 = tvm.nd.array(np.random.randn(12, 16384, 16).astype("float32"), device)
p1 = tvm.nd.array(np.random.randn(12, 1, 16).astype("float32"), device)
p2 = tvm.nd.array(np.random.randn(12, 16384, 1).astype("float32"), device)
f(p0, p1, p2)

