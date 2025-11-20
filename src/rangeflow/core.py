from typing import Any
from .backend import get_backend, to_tensor, to_cpu

class Symbol:
    def __init__(self, op_name, parents=(), value=None, **kwargs):
        self.op_name = op_name
        self.parents = parents
        self.value = value
        self.kwargs = kwargs
        self._cache = None

class RangeTensor:
    def __init__(self, symbol: Symbol, shape: tuple):
        self.symbol = symbol
        self.shape = shape

    @classmethod
    def from_array(cls, data):
        t = to_tensor(data)
        return cls(Symbol("LEAF", value=t), t.shape)
    
    @classmethod
    def from_range(cls, min_v, max_v):
        t_min, t_max = to_tensor(min_v), to_tensor(max_v)
        return cls(Symbol("LEAF_RANGE", value=(t_min, t_max)), t_min.shape)

    def decay(self):
        from .ops import evaluate_bounds
        return evaluate_bounds(self.symbol)
    
    # Ops
    def __add__(self, o): return _op("add", self, o)
    def __sub__(self, o): return _op("sub", self, o)
    def __mul__(self, o): return _op("mul", self, o)
    def __matmul__(self, o): return _op("matmul", self, o)
    def relu(self): return _op("relu", self)
    def tanh(self): return _op("tanh", self)
    def exp(self): return _op("exp", self)
    def transpose(self, d0, d1): return _op("transpose", self, dim0=d0, dim1=d1)
    def reshape(self, *s): return _op("reshape", self, shape=s)
    
    @property
    def T(self): return self.transpose(-1, -2)

def _op(name, *args, **kwargs):
    from .ops import infer_shape
    clean_args = [a if isinstance(a, RangeTensor) else RangeTensor.from_array(a) for a in args]
    shape = infer_shape(name, [a.shape for a in clean_args], **kwargs)
    return RangeTensor(Symbol(name, tuple(a.symbol for a in clean_args), **kwargs), shape)