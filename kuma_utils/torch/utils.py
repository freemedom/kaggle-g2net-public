import os
import random
import subprocess
import numpy as np
import torch
import time
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA = True
except ModuleNotFoundError:
    XLA = False


def freeze_module(module):
    # 中文：冻结模型参数，关闭梯度计算（常用于微调时固定 backbone）
    for i, param in enumerate(module.parameters()):
        param.requires_grad = False


def fit_state_dict(state_dict, model):
    '''
    Ignore size mismatch when loading state_dict
    '''
    # 中文：加载预训练权重时，遇到形状不匹配的参数直接跳过（打印提示）
    for name, param in model.named_parameters():
        if name in state_dict.keys():
            new_param = state_dict[name]
        else:
            continue
        if new_param.size() != param.size():
            print(f'Size mismatch in {name}: {new_param.shape} -> {param.shape}')
            state_dict.pop(name)


def get_device(arg):
    # 中文：根据传入参数决定计算设备与 device_ids：
    # - torch.device / xla_device：直接使用
    # - None / list / tuple：自动选择可用 GPU，否则 CPU；XLA 优先
    # - 字符串：直接转为 torch.device（'xla' 需环境支持 XLA）
    if isinstance(arg, torch.device) or \
        (XLA and isinstance(arg, xm.xla_device)):
        device = arg
    elif arg is None or isinstance(arg, (list, tuple)):
        if XLA:
            device = xm.xla_device()
        else:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(arg, str):
        if arg == 'xla' and XLA:
            device = xm.xla_device()
        else:
            device = torch.device(arg)
    
    if isinstance(arg, (list, tuple)):
        if isinstance(arg[0], int):
            device_ids = list(arg)
        elif isinstance(arg[0], str) and arg[0].isnumeric():
             device_ids = [ int(a) for a in arg ]
        else:
            raise ValueError(f'Invalid device: {arg}')
    else:
        if device.type == 'cuda':
            assert torch.cuda.is_available()
            if device.index is None:
                device_count = torch.cuda.device_count()
                if device_count > 1:
                    device_ids = list(range(device_count))
                else:
                    device_ids = [0]
            else:
                device_ids = [device.index]
        else:
            device_ids = [device.index]
    
    return device, device_ids


def seed_everything(random_state=0, deterministic=False):
    # 中文：设置 Python/NumPy/PyTorch 的随机种子，可选 deterministic 以获得可复现结果
    random.seed(random_state)
    os.environ['PYTHONHASHSEED'] = str(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False


def get_gpu_memory():
    """
    Code borrowed from: 
    https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_time(time_format='%H:%M:%S'):
    # 中文：返回当前本地时间的格式化字符串
    return time.strftime(time_format, time.localtime())
