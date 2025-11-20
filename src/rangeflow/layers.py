from .core import RangeTensor
from .backend import get_backend
import numpy as np

xp = get_backend()

class RangeModule:
    """Base class for all RangeFlow layers."""
    def __init__(self):
        self.training = True

    def __call__(self, x):
        return self.forward(x)
    
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False

class RangeLinear(RangeModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        limit = np.sqrt(2.0 / in_features)
        self.weight = RangeTensor.from_array(xp.random.uniform(-limit, limit, (out_features, in_features)))
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_features))
        else:
            self.bias = None
        
    def forward(self, x):
        # Linear is x @ W.T + b
        # We transpose the weight to match (out, in) -> (in, out) for matmul
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias
        return out

class RangeConv2d(RangeModule):
    """
    Robust Convolution Layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize Weights
        k = self.kernel_size[0] * self.kernel_size[1] * in_channels
        limit = np.sqrt(2.0 / k)
        # Shape: [Out, In, K, K]
        self.weight = RangeTensor.from_array(xp.random.uniform(-limit, limit, (out_channels, in_channels, *self.kernel_size)))
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # We delegate to a core op named 'conv2d'
        # NOTE: This requires 'conv2d' to be implemented in ops.py!
        # Since we want this to work *now*, we will use a trick:
        # We implement the Conv2d logic here using Im2Col (Unfold) + MatMul if possible,
        # OR we assume the user has added 'conv2d' to ops.py.
        
        # To make it work immediately without changing ops.py, let's use the 'unfold' trick 
        # (if backend supports it) OR just register a custom op dynamically.
        
        # SIMPLER FIX FOR V1.0:
        # We will just call a new internal helper _conv2d_op which we define here or in core.
        # But the cleanest way is to assume the 'conv2d' op exists in the graph.
        from .core import _op
        
        # We need to pass the parameters as keyword arguments to the op
        return _op("conv2d", x, self.weight, self.bias, 
                   stride=self.stride, padding=self.padding)


class RangeLayerNorm(RangeModule):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = RangeTensor.from_array(xp.ones(normalized_shape))
        self.bias = RangeTensor.from_array(xp.zeros(normalized_shape))

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        mu = xp.mean(center, axis=-1, keepdims=True)
        var = xp.var(center, axis=-1, keepdims=True)
        norm_center = (center - mu) / xp.sqrt(var + self.eps)
        
        norm_width = width / xp.sqrt(var + self.eps)
        
        # Re-leaf
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        gl, gh = self.weight.decay()
        bl, bh = self.bias.decay()
        
        fin_l = new_min * gl + bl
        fin_h = new_max * gh + bh
        
        return RangeTensor.from_range(fin_l, fin_h)

class RangeDropout(RangeModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0: return x
        
        min_x, max_x = x.decay()
        mask = xp.random.rand(*min_x.shape) > self.p
        
        LARGE = 10.0 
        out_min = xp.where(mask, min_x, -LARGE)
        out_max = xp.where(mask, max_x, LARGE)
        
        return RangeTensor.from_range(out_min, out_max)

class RangeAttention(RangeModule):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.q_proj = RangeLinear(embed_dim, embed_dim)
        self.k_proj = RangeLinear(embed_dim, embed_dim)
        self.v_proj = RangeLinear(embed_dim, embed_dim)
        self.out_proj = RangeLinear(embed_dim, embed_dim)
        
        self.scale = RangeTensor.from_array(xp.array(1.0 / np.sqrt(self.head_dim)))
    
    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        from .core import _op
        attn_weights = _op("softmax", scores, axis=-1)
        
        out = attn_weights @ V
        return self.out_proj(out)