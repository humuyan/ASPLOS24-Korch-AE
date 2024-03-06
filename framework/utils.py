import tvm
import os
from tvm import rpc

def configure_target(TARGET_DEVICE):
    # Configure target device for kernel profiling
    if TARGET_DEVICE == 'v100':
        DEVICE = tvm.cuda()
        TARGET = tvm.target.cuda(arch='sm_70' , 
                                options="-max_threads_per_block=1024 \
                                -max_shared_memory_per_block=96000")
        RPC_CONFIG = []
        
        return DEVICE, TARGET, RPC_CONFIG
            
    elif TARGET_DEVICE == 'a100':
        DEVICE = tvm.cuda()
        TARGET = tvm.target.cuda(arch='sm_80' , 
                                options="-max_threads_per_block=1024 \
                                -max_shared_memory_per_block=164000")
        RPC_CONFIG = []
        
        return DEVICE, TARGET, RPC_CONFIG
            
    else:
        assert(f'TARGET_DEVICE {TARGET_DEVICE} not supported')