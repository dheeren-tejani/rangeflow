import sys

try:
    import numpy as xp
    BACKEND = 'numpy'
except ImportError:
    raise ImportError("RangeFlow requires NumPy.")

try:
    import cupy as cp
    if cp.cuda.is_available():
        xp = cp
        BACKEND = 'cupy'
except ImportError:
    pass

def get_backend(): return xp
def to_tensor(d): return (cp.asarray(d) if BACKEND == 'cupy' else xp.asarray(d))
def to_cpu(d): 
    if BACKEND == 'cupy' and hasattr(d, 'get'): return d.get()
    return xp.asarray(d)
def get_device(): return "GPU" if BACKEND == 'cupy' else "CPU"