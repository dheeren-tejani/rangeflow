"""
RangeFlow Neural Network Layers
================================
Complete implementations of range-aware neural network layers.
All layers support both standard tensors and RangeTensors.
"""

import torch
import torch.nn as nn
import numpy as np
from .core import RangeTensor, _op
from .backend import get_backend

xp = get_backend()

class RangeModule(nn.Module):
    """Base class for all RangeFlow layers."""
    def __init__(self):
        super().__init__()
        self.training = True

    def __call__(self, x):
        return self.forward(x)
    
    def train(self, mode=True):
        super().train(mode)
        self.training = mode
        return self
        
    def eval(self):
        super().eval()
        self.training = False
        return self


# ==========================================
# CORE LAYERS
# ==========================================

class RangeLinear(RangeModule):
    """Fully connected layer with range propagation."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Safe Initialization (0.05) to prevent explosion
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x):
        w = RangeTensor.from_array(self.weight)
        # Linear expects x @ W.T
        out = x @ w.transpose(-1, -2)
        
        if self.bias is not None:
            out = out + RangeTensor.from_array(self.bias)
        return out


class RangeConv2d(RangeModule):
    """2D Convolution with range propagation."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Kaiming Init approximation
        k = self.kernel_size[0] * self.kernel_size[1] * in_channels
        std = np.sqrt(1.0 / k)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * std)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        w = RangeTensor.from_array(self.weight)
        b = RangeTensor.from_array(self.bias) if self.bias is not None else None
        return _op("conv2d", x, w, b, stride=self.stride, padding=self.padding)


# ==========================================
# NORMALIZATION LAYERS
# ==========================================

class RangeLayerNorm(RangeModule):
    """Layer Normalization with width stabilization (The Balloon Popper)."""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        # Handle PyTorch dimensions dynamically
        param_len = len(self.weight.shape)
        dims = tuple(range(center.ndim - param_len, center.ndim))
        
        mu = center.mean(dim=dims, keepdim=True)
        var = center.var(dim=dims, keepdim=True, unbiased=False)
        
        norm_center = (center - mu) / torch.sqrt(var + self.eps)
        norm_width = width / torch.sqrt(var + self.eps)
        
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        w = RangeTensor.from_array(self.weight)
        b = RangeTensor.from_array(self.bias)
        
        return RangeTensor.from_range(new_min, new_max) * w + b


class RangeBatchNorm1d(RangeModule):
    """1D Batch Normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        if self.training:
            mu = center.mean(dim=0)
            var = center.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mu = self.running_mean
            var = self.running_var
        
        norm_center = (center - mu) / torch.sqrt(var + self.eps)
        norm_width = width / torch.sqrt(var + self.eps)
        
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        w = RangeTensor.from_array(self.weight)
        b = RangeTensor.from_array(self.bias)
        
        return RangeTensor.from_range(new_min, new_max) * w + b


class RangeBatchNorm2d(RangeModule):
    """2D Batch Normalization."""
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        if self.training:
            mu = center.mean(dim=(0, 2, 3), keepdim=True)
            var = center.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mu = self.running_mean
            var = self.running_var
        
        norm_center = (center - mu) / torch.sqrt(var + self.eps)
        norm_width = width / torch.sqrt(var + self.eps)
        
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        w = RangeTensor.from_array(self.weight)
        b = RangeTensor.from_array(self.bias)
        
        return RangeTensor.from_range(new_min, new_max) * w + b


# ==========================================
# POOLING LAYERS
# ==========================================

class RangeMaxPool2d(RangeModule):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return _op("max_pool2d", x, kernel_size=self.kernel_size, 
                   stride=self.stride, padding=self.padding)


class RangeAvgPool2d(RangeModule):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return _op("avg_pool2d", x, kernel_size=self.kernel_size, 
                   stride=self.stride, padding=self.padding)


# ==========================================
# RECURRENT LAYERS
# ==========================================

class RangeRNN(RangeModule):
    """Vanilla RNN with range propagation."""
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        std = np.sqrt(1.0 / hidden_size)
        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size) * std)
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * std)
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x, h=None):
        w_ih = RangeTensor.from_array(self.weight_ih)
        w_hh = RangeTensor.from_array(self.weight_hh)
        b = RangeTensor.from_array(self.bias)
        
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        
        if h is None:
            batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
            h = RangeTensor.from_array(torch.zeros((batch_size, self.hidden_size), device=self.weight_ih.device))
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            linear_part = (x_t @ w_ih.transpose(-1, -2)) + (h @ w_hh.transpose(-1, -2)) + b
            
            if self.nonlinearity == 'tanh':
                h = linear_part.tanh()
            elif self.nonlinearity == 'relu':
                h = linear_part.relu()
            outputs.append(h)
        return outputs, h


class RangeLSTM(RangeModule):
    """LSTM with range propagation."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        std = np.sqrt(1.0 / hidden_size)
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size) * std)
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * std)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))
    
    def forward(self, x, state=None):
        w_ih = RangeTensor.from_array(self.weight_ih)
        w_hh = RangeTensor.from_array(self.weight_hh)
        b = RangeTensor.from_array(self.bias)
        
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
        
        if state is None:
            h = RangeTensor.from_array(torch.zeros((batch_size, self.hidden_size), device=self.weight_ih.device))
            c = RangeTensor.from_array(torch.zeros((batch_size, self.hidden_size), device=self.weight_ih.device))
        else:
            h, c = state
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            gates = (x_t @ w_ih.transpose(-1, -2)) + (h @ w_hh.transpose(-1, -2)) + b
            
            i = gates[:, :self.hidden_size].sigmoid()
            f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()
            g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()
            o = gates[:, 3*self.hidden_size:].sigmoid()
            
            c = (f * c) + (i * g)
            h = o * c.tanh()
            outputs.append(h)
        return outputs, (h, c)


class RangeGRU(RangeModule):
    """GRU with range propagation."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        std = np.sqrt(1.0 / hidden_size)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size) * std)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size) * std)
        self.bias = nn.Parameter(torch.zeros(3 * hidden_size))
    
    def forward(self, x, h=None):
        w_ih = RangeTensor.from_array(self.weight_ih)
        w_hh = RangeTensor.from_array(self.weight_hh)
        b = RangeTensor.from_array(self.bias)
        
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
        
        if h is None:
            h = RangeTensor.from_array(torch.zeros((batch_size, self.hidden_size), device=self.weight_ih.device))
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            gates = (x_t @ w_ih.transpose(-1, -2)) + (h @ w_hh.transpose(-1, -2)) + b
            
            r = gates[:, :self.hidden_size].sigmoid()
            z = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()
            n = gates[:, 2*self.hidden_size:].tanh()
            
            ones = RangeTensor.from_array(torch.ones((1,), device=self.weight_ih.device))
            h = ((ones - z) * n) + (z * h)
            outputs.append(h)
        return outputs, h


# ==========================================
# ATTENTION LAYERS
# ==========================================

class RangeAttention(RangeModule):
    """Multi-Head Self Attention."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = RangeLinear(embed_dim, embed_dim)
        self.k_proj = RangeLinear(embed_dim, embed_dim)
        self.v_proj = RangeLinear(embed_dim, embed_dim)
        self.out_proj = RangeLinear(embed_dim, embed_dim)
        self.scale = RangeTensor.from_array(torch.tensor(1.0 / np.sqrt(self.head_dim)))
    
    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = _op("softmax", scores, axis=-1)
        out = attn_weights @ V
        return self.out_proj(out)


# ==========================================
# REGULARIZATION & UTILS
# ==========================================

class RangeDropout(RangeModule):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        min_x, max_x = x.decay()
        mask = torch.rand_like(min_x) > self.p
        
        LARGE = 10.0
        out_min = torch.where(mask, min_x, min_x - LARGE)
        out_max = torch.where(mask, max_x, max_x + LARGE)
        return RangeTensor.from_range(out_min, out_max)

class RangeReLU(RangeModule):
    def forward(self, x): return x.relu()

class RangeSigmoid(RangeModule):
    def forward(self, x): return x.sigmoid()

class RangeTanh(RangeModule):
    def forward(self, x): return x.tanh()

class RangeGELU(RangeModule):
    def forward(self, x):
        min_x, max_x = x.decay()
        def gelu(z):
            return 0.5 * z * (1 + torch.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
        corners = [gelu(min_x), gelu(max_x)]
        return RangeTensor.from_range(torch.min(corners[0], corners[1]), torch.max(corners[0], corners[1]))

class RangeFlatten(RangeModule):
    def forward(self, x): return x.flatten()

class RangeSequential(nn.Sequential):
    """Compatible with both RangeModule and nn.Sequential"""
    def __init__(self, *args):
        super().__init__(*args)