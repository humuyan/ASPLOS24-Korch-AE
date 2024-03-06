nvcc -gencode arch=compute_70,code=sm_70 -O3 compute_bound_profiler.cu -lcudnn -lcublas -std=c++17 -shared -Xcompiler -fPIC $(python3 -m pybind11 --includes) -o compute_bound_profiler$(python3-config --extension-suffix) -L/usr/local/cuda/lib -I/usr/local/cuda/include
# nvcc -gencode arch=compute_80,code=sm_80 -O3 compute_bound_profiler.cu -lcudnn -lcublas -std=c++17 -shared -Xcompiler -fPIC $(python3 -m pybind11 --includes) -o compute_bound_profiler$(python3-config --extension-suffix) -L/usr/local/cuda/lib -I/usr/local/cuda/include
