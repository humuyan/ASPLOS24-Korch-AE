# Korch AE

## Environment preparation

### Set up Python Environment

```bash
source activate pytorch
pip install nvidia-pyindex
pip install tornado psutil 'xgboost<1.6.0' cloudpickle onnx onnx-graphsurgeon transformers netron sortedcontainers pulp==2.7.0
```

### Install TVM

```bash
git clone --recursive https://github.com/balamurugan15/tvm-kernel-mapper.git tvm
export TVM_HOME=`realpath tvm`
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .
cmake ..
make -j4
python -c "import tvm; print(tvm.__version__)"
```

Showing `0.13.dev0` means that TVM has been installed correctly.

### Install TensorRT

Download [TensorRT 8.2 EA for Linux x86_64 and CUDA 11.0, CUDA 11.1, CUDA 11.2, 11.3 and 11.4 TAR Package](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.0/tars/tensorrt-8.2.0.6.linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz) as `trt.tar.gz`. Extract it with `` will get a `TensorRT-8.2.0.6` folder.

```bash
tar -zxvf trt.tar.gz
export PATH=`realpath TensorRT-8.2.0.6/targets/x86_64-linux-gnu/bin`:$PATH
export LD_LIBRARY_PATH=`realpath TensorRT-8.2.0.6/targets/x86_64-linux-gnu/lib`:$LD_LIBRARY_PATH
trtexec
```

Showing `&&&& PASSED TensorRT.trtexec [TensorRT v8200] # trtexec` means that TensorRT has been installed correctly.

### Clone Korch Repo and compile Korch's compute-bound profiler

```bash
git clone https://github.com/humuyan/ASPLOS24-Korch-AE.git korch
cd korch/operators
./build.sh
cp compute_bound_profiler.cpython-39-x86_64-linux-gnu.so ../framework
```

## Run TensorRT baseline

```bash
cd korch/cases
trtexec --onnx=candy.onnx
trtexec --onnx=segformer.onnx
```

After each run, `GPU Compute Time: min = XXX ms, max = XXX ms, mean = XXX ms, median = XXX ms, percentile(99%) = XXX ms` will show in the `Performance summary` at the end of `trtexec` output. We use mean time as TensorRT's end-to-end latency.

## Run Korch

### Operator fission

```bash
cd ../framework
python operator_fission.py ../cases/candy.onnx candy_fission.onnx
python operator_fission.py ../cases/segformer.onnx segformer_fission.onnx
```

You can use `netron` to visualize the ONNX graph to see the difference after operator fission.

Run `trtexec --onnx=segformer_fission.onnx` to get the result of adaption study over TensorRT. The numbers should be similar with Figure 7 in the paper.

### Kernel orchestration

```bash
python calc.py candy_fission.onnx v100
python calc.py segformer_fission.onnx v100
```

