import onnx
import onnx_graphsurgeon as gs
import sys
from typing import Optional, Tuple
from sortedcontainers import SortedSet
from tqdm import tqdm
import pulp
from profiler import profile
import os
import shutil

from utils import configure_target


assert len(sys.argv) in [3, 4]
onnx_graph = gs.import_onnx(onnx.load(sys.argv[1]))
if sys.argv[1].startswith("segformer"):
    cut_points = [
        (["input.1"], ["/model/segformer/encoder/Transpose_output_0"]),
        (["/model/segformer/encoder/Transpose_output_0"], ["/model/segformer/encoder/Transpose_1_output_0"]),
        (["/model/segformer/encoder/Transpose_1_output_0"], ["/model/segformer/encoder/Transpose_2_output_0"]),
        (["/model/segformer/encoder/Transpose_2_output_0"], ["/model/decode_head/linear_c.3/proj/MatMul_output_0"]),
        (["/model/segformer/encoder/Transpose_output_0"], ["/model/decode_head/linear_c.0/proj/MatMul_output_0"]),
        (["/model/segformer/encoder/Transpose_1_output_0"], ["/model/decode_head/linear_c.1/proj/MatMul_output_0"]),
        (["/model/segformer/encoder/Transpose_2_output_0"], ["/model/decode_head/linear_c.2/proj/MatMul_output_0"]),
        (["/model/decode_head/linear_c.0/proj/MatMul_output_0", "/model/decode_head/linear_c.1/proj/MatMul_output_0", "/model/decode_head/linear_c.2/proj/MatMul_output_0", "/model/decode_head/linear_c.3/proj/MatMul_output_0"], ["/model/decode_head/Concat_4_output_0"]),
        (["/model/decode_head/Concat_4_output_0"], ["1558"])
    ]
    full_graph = [7]
elif sys.argv[1].startswith("yolox"):
    cut_points = [
        (["onnx::Slice_0"], ["/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"]),
        (["/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"], ["/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"]),
        (["/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"], ["/backbone/lateral_conv0/act/Mul_output_0"]),
        (["/backbone/lateral_conv0/act/Mul_output_0", "/backbone/backbone/dark4/dark4.1/conv3/act/Mul_output_0"], ["/backbone/reduce_conv1/act/Mul_output_0"]),
        (["/backbone/reduce_conv1/act/Mul_output_0", "/backbone/backbone/dark3/dark3.1/conv3/act/Mul_output_0"], ["/backbone/C3_p3/conv3/act/Mul_output_0"]),
        (["/backbone/C3_p3/conv3/act/Mul_output_0", "/backbone/reduce_conv1/act/Mul_output_0"], ["/backbone/C3_n3/conv3/act/Mul_output_0"]),
        (["/backbone/C3_n3/conv3/act/Mul_output_0", "/backbone/lateral_conv0/act/Mul_output_0"], [f"/head/{i}_preds.2/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        (["/backbone/C3_n3/conv3/act/Mul_output_0"], [f"/head/{i}_preds.1/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        (["/backbone/C3_p3/conv3/act/Mul_output_0"], [f"/head/{i}_preds.0/Conv_output_0" for i in ["reg", "obj", "cls"]]),
        ([f"/head/{j}_preds.{i}/Conv_output_0" for i in range(3) for j in ["reg", "obj", "cls"]], ["1171"])
    ]
    full_graph = [9]
elif sys.argv[1].startswith("yolov4"):
    cut_points = [
        (["input.1"], ["/down3/conv5/conv.2/Mul_output_0"]),
        (["/down3/conv5/conv.2/Mul_output_0"], ["/down4/conv5/conv.2/Mul_output_0"]),
        (["/down4/conv5/conv.2/Mul_output_0"], ["/neck/conv6/conv.2/LeakyRelu_output_0"]),
        (["/down4/conv5/conv.2/Mul_output_0", "/neck/conv6/conv.2/LeakyRelu_output_0"], ["/neck/conv13/conv.2/LeakyRelu_output_0"]),
        (["/neck/conv13/conv.2/LeakyRelu_output_0", "/down3/conv5/conv.2/Mul_output_0"], ["/neck/conv20/conv.2/LeakyRelu_output_0"]),
        (["/neck/conv20/conv.2/LeakyRelu_output_0"], ["/head/conv2/conv.0/Conv_output_0"]),
        (["/neck/conv13/conv.2/LeakyRelu_output_0", "/neck/conv20/conv.2/LeakyRelu_output_0", "/neck/conv6/conv.2/LeakyRelu_output_0"], ["/head/conv10/conv.0/Conv_output_0", "/head/conv18/conv.0/Conv_output_0"]),
        (["/head/conv2/conv.0/Conv_output_0", "/head/conv10/conv.0/Conv_output_0", "/head/conv18/conv.0/Conv_output_0"], ["1979"])#[tensor.name for tensor in onnx_graph.outputs])#["1965"])
    ]
    full_graph = [7]
else:
    cut_points = []
    full_graph = []

TARGET_DEVICE = sys.argv[2]
DEVICE, TARGET, RPC_CONFIG = configure_target(TARGET_DEVICE)

if len(cut_points) == 0:
    cut_points = [([tensor.name for tensor in onnx_graph.inputs], [tensor.name for tensor in onnx_graph.outputs])]

specified_subgraph = int(sys.argv[3]) if len(sys.argv) == 4 else None

latencies = []
for subgraph_id, (inputs, outputs) in enumerate(cut_points):
    if specified_subgraph is not None and subgraph_id != specified_subgraph:
        continue
    onnx_dir = f"./{sys.argv[1]}_onnx_{subgraph_id}/"
    if not os.path.exists(onnx_dir):
        os.mkdir(onnx_dir)

    graph = onnx_graph.copy()
    tensors = graph.tensors()
    graph.inputs = [tensors[input] for input in inputs]
    graph.outputs = [tensors[output] for output in outputs]
    graph.cleanup(True, True, True)
    # onnx.save(gs.export_onnx(graph), f"yolov4_{i}.onnx")
    # continue
    # if i != 7:
        # continue

    print(f"Graph loaded, number of nodes: {len(graph.nodes)}")

    consts = set()
    for node in graph.nodes:
        if node.op == "Constant":
            consts.add(node.outputs[0].name)

    K = []
    if subgraph_id in full_graph:
        K.append((graph, None))
        onnx.save(gs.export_onnx(graph), f"{onnx_dir}{len(K)}.onnx")
    else:
        ### Calculate execution states ###
        cur = set()  # current execution state
        for tensor in graph.inputs:
            cur.add(tensor.name)
        execution_states = SortedSet({str(sorted(cur))})
        output2node = dict()

        def dfs(depth: int):
            for node in graph.nodes:
                if len(node.inputs) == 0:
                    continue
                prepared = True
                for tensor in node.inputs:
                    if not isinstance(tensor, gs.Constant) and tensor.name not in consts and tensor.name not in cur:
                        prepared = False
                        break
                if prepared:  # all dependencies have been calculated
                    assert len(node.outputs) == 1  # only consider operators with 1 output for now
                    tensor = node.outputs[0]
                    if tensor.name not in cur:
                        output2node[tensor.name] = node
                        cur.add(tensor.name)
                        s = str(sorted(cur))
                        if s not in execution_states:
                            execution_states.add(s)
                            dfs(depth + 1)
                        cur.remove(tensor.name)

        dfs(1)
        print(f"Number of execution states: {len(execution_states)}")
        D = [set(eval(x)) for x in execution_states]

        ### Calculate candidate kernels ###
        def pruning_check(candidate_kernel: set) -> Tuple[bool, list]:
            """
            Return whether the given kernel should be preserved.
            If it is a valid compute-bound kernel, `dict` of parameters will be returned.
            """
            compute_bound = 0
            for tensor in candidate_kernel:
                if output2node[tensor].op in ["MatMul", "Conv", "Gemm"]:
                    compute_bound += 1
                    op = output2node[tensor].op
                    if compute_bound > 1:
                        return False, None
            if compute_bound == 1:
                if op == "Conv":
                    if len(candidate_kernel) > 1:  # TODO: conv + relu
                        return False, None
                    conv = output2node[next(iter(candidate_kernel))]  # conv must be the first node
                    kernel_size = conv.attrs["kernel_shape"]
                    assert kernel_size[0] == kernel_size[1]
                    stride = conv.attrs["strides"]
                    assert stride[0] == stride[1]
                    pad = conv.attrs["pads"]
                    assert pad[0] == pad[1] == pad[2] == pad[3]
                    dilation = conv.attrs["dilations"]
                    assert dilation[0] == dilation[1]
                    params = {
                        "type": "conv", "batch_size": conv.inputs[0].shape[0],
                        "in_channels": conv.inputs[0].shape[1], "in_height": conv.inputs[0].shape[2],
                        "in_width": conv.inputs[0].shape[3], "out_channels": conv.inputs[1].shape[0],
                        "kernel_size": kernel_size[0], "stride": stride[0], "padding": pad[0],
                        "dilation": dilation[0], "groups": conv.attrs["group"],
                        "mode": 0 if len(conv.inputs) == 2 else 1}
                    return True, params
                elif op == "MatMul":
                    # TODO: more cases of matmul
                    if len(candidate_kernel) > 1:
                        return False, None
                    matmul = output2node[next(iter(candidate_kernel))]
                    shape_a, shape_b = [x.shape for x in matmul.inputs]
                    params = {"type": "matmul", "transa": False, "transb": False, "shapea": shape_a, "shapeb": shape_b}
                    return True, params
                elif op == "Gemm":
                    gemm = output2node[next(iter(candidate_kernel))]
                    if len(candidate_kernel) > 1:
                        return False, None
                    shape_a, shape_b = [gemm.inputs[i].shape for i in range(2)]
                    transa = False if "transA" not in gemm.attrs else gemm.attrs["transA"] == 1
                    transb = False if "transB" not in gemm.attrs else gemm.attrs["transB"] == 1
                    params = {"type": "matmul", "transa": transa, "transb": transb, "shapea": shape_a, "shapeb": shape_b}
                    return True, params
                else:
                    return False, None
            return True, None

        def export_candidate_kernel_graph(candidate_kernel: set) -> Optional[gs.Graph]:
            """
                If given candidate kernel is valid, return its `gs.Graph`. Otherwise return `None`.
            """
            subgraph = graph.copy()
            subgraph.inputs, subgraph.outputs = [], []
            nodes = []
            tensor_map = subgraph.tensors()
            for tensor_name in candidate_kernel:
                # find the corresponding node
                for node in subgraph.nodes:
                    if node.outputs[0].name == tensor_name:
                        cur_node = node
                        break
                nodes.append(cur_node)
                # find inputs
                for tensor in cur_node.inputs:
                    if not isinstance(tensor, gs.Constant) and tensor.name not in consts and tensor.name not in candidate_kernel and tensor not in subgraph.inputs:
                        subgraph.inputs.append(tensor)
                if cur_node.outputs[0].name not in candidate_kernel:
                    subgraph.outputs.append(tensor)
            # find outputs
            for tensor_name in candidate_kernel:
                is_output = True
                for node in nodes:
                    for tensor in node.inputs:
                        if tensor.name == tensor_name:
                            is_output = False
                if is_output:
                    if len(subgraph.outputs) != 0:  # multiple outputs
                        return None
                    subgraph.outputs.append(tensor_map[tensor_name])
            subgraph.cleanup(True, True, True)
            return subgraph

        for i in tqdm(range(len(D))):
            for j in range(len(D)):
                if i != j and D[i].issubset(D[j]):
                    k = D[j] - D[i]
                    preserve, params = pruning_check(k)
                    if preserve:
                        candidate_kernel_graph = export_candidate_kernel_graph(k)
                        if candidate_kernel_graph is not None:
                            K.append((candidate_kernel_graph, params))
                            onnx.save(gs.export_onnx(candidate_kernel_graph), f"{onnx_dir}{len(K)}.onnx")

    print(f"Number of candidate kernels: {len(K)}")

    ### Profile latency of candidate kernels

    latency = []
    for i, (_, params) in enumerate(tqdm(K)):
        latency.append(profile(f"{onnx_dir}{i + 1}.onnx", params, DEVICE, TARGET, RPC_CONFIG, TARGET_DEVICE, f"./{sys.argv[1]}{subgraph_id}/"))
        # print(candidate_kernel, latency[-1], sep='\n', file=open("latency.txt", "a"))
    print(latency, file=open("latency.txt", "a"))
    # exit()

    ### Build and solve binary programming ###

    name2id = dict()
    def visit(name: str) -> int:
        if name not in name2id:
            name2id[name] = len(name2id)
        return name2id[name]

    def get_inputs(graph):
        ret = []
        for tensor in graph.inputs:
            if tensor.name not in consts:
                ret.append(visit(tensor.name))
        return ret

    graph_inputs = get_inputs(graph)

    inputs = []
    outputs = []
    for candidate_kernel, _ in K:
        inputs.append(get_inputs(candidate_kernel))
        outputs.append(visit(candidate_kernel.outputs[0].name))

    bp = pulp.LpProblem("BinaryProgramming")
    a = [pulp.LpVariable(f"a{i}", cat="Binary") for i in range(len(K))]
    bp += (sum([latency[i] * a[i] for i in range(len(K))]))
    b = []
    for _, id in name2id.items():
        if id in graph_inputs:
            b.append(1)
        else:
            b.append(0)
            for i in range(len(K)):
                if outputs[i] == id:
                    b[id] += a[i]
    for tensor in graph.outputs:
        bp += (b[name2id[tensor.name]] >= 1)
    for name, id in name2id.items():
        for i in range(len(K)):
            if id in inputs[i]:
                bp += (b[id] >= a[i])
    bp.solve(pulp.PULP_CBC_CMD(maxSeconds=1000, msg=1, fracGap=0))
    for v in bp.variables():
        if v.varValue == 1:
            id = int(v.name[1:])
            print(id, [x.name for x in K[id][0].nodes], K[id][0].outputs[0].name, latency[id])
    latencies.append(bp.objective.value())
    print("Current latencies:", latencies)

print("Overall latency:", sum(latencies))
