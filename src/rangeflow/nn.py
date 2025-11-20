from .core import RangeTensor
from .backend import get_backend
import numpy as np

xp = get_backend()

class RangeModule:
    """Base class for all RangeFlow layers."""
    def __init__(self):
        self._parameters = {}
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
        # Initialize weights (He initialization approx)
        limit = np.sqrt(2.0 / in_features)
        self.weight = RangeTensor.from_array(xp.random.uniform(-limit, limit, (out_features, in_features)))
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_features))
        else:
            self.bias = None
        
    def forward(self, x):
        # x: [..., in_features]
        # weight: [out_features, in_features]
        # Linear is x @ W.T + b
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias
        return out

class RangeConv2d(RangeModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Weights: [out, in, k, k]
        k = self.kernel_size[0] * self.kernel_size[1] * in_channels
        limit = np.sqrt(2.0 / k)
        self.weight = RangeTensor.from_array(xp.random.uniform(-limit, limit, (out_channels, in_channels, *self.kernel_size)))
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # For prototype, we implement Conv2d via Unfold + MatMul (Im2Col)
        # or assume the backend supports it. 
        # Since we are framework-agnostic, we rely on the user mapping this to torch 
        # or using a backend that supports 'conv2d' op.
        # For now, we raise NotImplementedError for pure NumPy backend Conv2d without torch
        # UNLESS we add 'conv2d' to ops.py.
        # To keep it simple for this version, we assume the 'core' has an 'op' for it.
        from .core import _op
        return _op("conv2d", x, self.weight, self.bias, stride=self.stride, padding=self.padding)

class RangeLayerNorm(RangeModule):
    """
    The 'Balloon Popper'.
    Normalizes Range Centers and Range Widths to prevent explosion.
    """
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable affine parameters
        self.weight = RangeTensor.from_array(xp.ones(normalized_shape))
        self.bias = RangeTensor.from_array(xp.zeros(normalized_shape))

    def forward(self, x):
        # 1. Calculate Concrete Bounds
        min_x, max_x = x.decay()
        
        # 2. Center and Width
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        # 3. Normalize Center (Standard LayerNorm)
        # Mean/Var across last dimension
        mu = xp.mean(center, axis=-1, keepdims=True)
        var = xp.var(center, axis=-1, keepdims=True)
        norm_center = (center - mu) / xp.sqrt(var + self.eps)
        
        # 4. Normalize Width (The Balloon Pop)
        # We divide width by the standard deviation of the center.
        # This keeps the uncertainty relative to the signal strength.
        norm_width = width / xp.sqrt(var + self.eps)
        
        # 5. Reconstruct RangeTensor (Leaf)
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        # Apply gamma/beta
        gl, gh = self.weight.decay()
        bl, bh = self.bias.decay()
        
        # Gamma is positive usually
        fin_l = new_min * gl + bl
        fin_h = new_max * gh + bh
        
        return RangeTensor.from_range(fin_l, fin_h)

class RangeDropout(RangeModule):
    """
    Uncertainty Dropout.
    Instead of zeroing values, we explode their uncertainty.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0: return x
        
        min_x, max_x = x.decay()
        
        # Create Dropout Mask
        mask = xp.random.rand(*min_x.shape) > self.p
        
        # Where mask is False (Dropped), set range to [-Large, +Large]
        LARGE = 10.0 
        
        # Keep original where mask is 1
        out_min = xp.where(mask, min_x, -LARGE)
        out_max = xp.where(mask, max_x, LARGE)
        
        return RangeTensor.from_range(out_min, out_max)

class RangeAttention(RangeModule):
    """
    Multi-Head Self Attention (Simplified).
    """
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
        # x: [Batch, Seq, Dim]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Score = Q @ K.T
        # Note: Need explicit reshape/transpose for multi-head, 
        # for prototype we assume 1 head or pre-reshaped
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Range Softmax
        from .core import _op
        attn_weights = _op("softmax", scores, axis=-1)
        
        # Aggregation
        out = attn_weights @ V
        return self.out_proj(out)