import sys

# Priority 1: PyTorch (For Training/GPU)
try:
    import torch
    xp = torch
    BACKEND = 'torch'
except ImportError:
    # Priority 2: NumPy (Legacy/Lightweight)
    try:
        import numpy as xp
        BACKEND = 'numpy'
    except ImportError:
        raise ImportError("RangeFlow requires PyTorch or NumPy.")

# Priority 3: CuPy (If PyTorch is missing but CUDA is present)
try:
    import cupy as cp
    if cp.cuda.is_available() and BACKEND == 'numpy':
        xp = cp
        BACKEND = 'cupy'
except ImportError:
    pass

def get_backend(): return xp

def to_tensor(d):
    if BACKEND == 'torch':
        if isinstance(d, torch.Tensor): return d
        # Default to CUDA if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.as_tensor(d, device=device)
    return (cp.asarray(d) if BACKEND == 'cupy' else xp.asarray(d))

def to_cpu(d): 
    if BACKEND == 'torch': 
        if isinstance(d, torch.Tensor): return d.detach().cpu().numpy()
        return d
    if BACKEND == 'cupy' and hasattr(d, 'get'): return d.get()
    return xp.asarray(d)

def get_device(): 
    if BACKEND == 'torch': return "GPU" if torch.cuda.is_available() else "CPU"
    return "GPU" if BACKEND == 'cupy' else "CPU"