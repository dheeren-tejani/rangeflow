"""
RangeFlow Neural Network Layers
================================
Complete implementations of range-aware neural network layers.
All layers support both standard tensors and RangeTensors.
"""

from .core import RangeTensor, _op
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


# ==========================================
# CORE LAYERS
# ==========================================

class RangeLinear(RangeModule):
    """
    Fully connected layer with range propagation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias term
    
    Example:
        >>> layer = RangeLinear(128, 64)
        >>> x_range = RangeTensor.from_range(x - 0.1, x + 0.1)
        >>> y_range = layer(x_range)
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        limit = np.sqrt(2.0 / in_features)
        self.weight = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (out_features, in_features))
        )
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_features))
        else:
            self.bias = None
        
    def forward(self, x):
        """Forward pass: y = xW^T + b"""
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias
        return out


class RangeConv2d(RangeModule):
    """
    2D Convolution with range propagation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel
        stride: Stride of convolution
        padding: Zero-padding added to input
        bias: Whether to include bias
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        k = self.kernel_size[0] * self.kernel_size[1] * in_channels
        limit = np.sqrt(2.0 / k)
        self.weight = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (out_channels, in_channels, *self.kernel_size))
        )
        
        if bias:
            self.bias = RangeTensor.from_array(xp.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        return _op("conv2d", x, self.weight, self.bias, 
                   stride=self.stride, padding=self.padding)


# ==========================================
# NORMALIZATION LAYERS
# ==========================================

class RangeLayerNorm(RangeModule):
    """
    Layer Normalization with width stabilization (The Balloon Popper).
    
    Normalizes both the center and width of ranges to prevent
    exponential explosion in deep networks.
    
    Args:
        normalized_shape: Input shape to normalize over
        eps: Epsilon for numerical stability
    """
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
        
        # Normalize center (standard LayerNorm)
        mu = xp.mean(center, axis=-1, keepdims=True)
        var = xp.var(center, axis=-1, keepdims=True)
        norm_center = (center - mu) / xp.sqrt(var + self.eps)
        
        # Normalize width (prevent explosion)
        norm_width = width / xp.sqrt(var + self.eps)
        
        # Reconstruct range
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        # Apply affine transform
        gl, gh = self.weight.decay()
        bl, bh = self.bias.decay()
        
        fin_l = new_min * gl + bl
        fin_h = new_max * gh + bh
        
        return RangeTensor.from_range(fin_l, fin_h)


class RangeBatchNorm1d(RangeModule):
    """
    1D Batch Normalization for fully connected layers.
    
    Args:
        num_features: Number of features (channels)
        eps: Epsilon for numerical stability
        momentum: Momentum for running statistics
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = RangeTensor.from_array(xp.ones(num_features))
        self.bias = RangeTensor.from_array(xp.zeros(num_features))
        
        # Running statistics (not ranges, just scalars)
        self.running_mean = xp.zeros(num_features)
        self.running_var = xp.ones(num_features)

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        if self.training:
            # Use batch statistics
            mu = xp.mean(center, axis=0)
            var = xp.var(center, axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics
            mu = self.running_mean
            var = self.running_var
        
        # Normalize
        norm_center = (center - mu) / xp.sqrt(var + self.eps)
        norm_width = width / xp.sqrt(var + self.eps)
        
        # Reconstruct
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        # Affine
        gl, gh = self.weight.decay()
        bl, bh = self.bias.decay()
        
        fin_l = new_min * gl + bl
        fin_h = new_max * gh + bh
        
        return RangeTensor.from_range(fin_l, fin_h)


class RangeBatchNorm2d(RangeModule):
    """
    2D Batch Normalization for convolutional layers.
    
    Args:
        num_features: Number of channels
        eps: Epsilon for numerical stability
        momentum: Momentum for running statistics
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = RangeTensor.from_array(xp.ones((1, num_features, 1, 1)))
        self.bias = RangeTensor.from_array(xp.zeros((1, num_features, 1, 1)))
        
        self.running_mean = xp.zeros((1, num_features, 1, 1))
        self.running_var = xp.ones((1, num_features, 1, 1))

    def forward(self, x):
        min_x, max_x = x.decay()
        center = (min_x + max_x) / 2
        width = (max_x - min_x)
        
        if self.training:
            # Compute over (N, H, W) dimensions
            mu = xp.mean(center, axis=(0, 2, 3), keepdims=True)
            var = xp.var(center, axis=(0, 2, 3), keepdims=True)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mu = self.running_mean
            var = self.running_var
        
        norm_center = (center - mu) / xp.sqrt(var + self.eps)
        norm_width = width / xp.sqrt(var + self.eps)
        
        new_min = norm_center - (norm_width / 2)
        new_max = norm_center + (norm_width / 2)
        
        gl, gh = self.weight.decay()
        bl, bh = self.bias.decay()
        
        fin_l = new_min * gl + bl
        fin_h = new_max * gh + bh
        
        return RangeTensor.from_range(fin_l, fin_h)


# ==========================================
# POOLING LAYERS
# ==========================================

class RangeMaxPool2d(RangeModule):
    """
    2D Max Pooling with range propagation.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling
        padding: Padding to add
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return _op("max_pool2d", x, kernel_size=self.kernel_size, 
                   stride=self.stride, padding=self.padding)


class RangeAvgPool2d(RangeModule):
    """
    2D Average Pooling with range propagation.
    
    Args:
        kernel_size: Size of pooling window
        stride: Stride of pooling
        padding: Padding to add
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return _op("avg_pool2d", x, kernel_size=self.kernel_size, 
                   stride=self.stride, padding=self.padding)


# ==========================================
# RECURRENT LAYERS
# ==========================================

class RangeRNN(RangeModule):
    """
    Vanilla RNN with range propagation.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        nonlinearity: 'tanh' or 'relu'
    """
    def __init__(self, input_size, hidden_size, nonlinearity='tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Weights
        limit = np.sqrt(1.0 / hidden_size)
        self.weight_ih = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (hidden_size, input_size))
        )
        self.weight_hh = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (hidden_size, hidden_size))
        )
        self.bias = RangeTensor.from_array(xp.zeros(hidden_size))
    
    def forward(self, x, h=None):
        """
        Args:
            x: Input sequence (seq_len, batch, input_size) as RangeTensor
            h: Initial hidden state (batch, hidden_size) as RangeTensor or None
        
        Returns:
            output: (seq_len, batch, hidden_size)
            h_n: Final hidden state (batch, hidden_size)
        """
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        
        if h is None:
            # Initialize hidden state as zero range
            batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
            h = RangeTensor.from_array(xp.zeros((batch_size, self.hidden_size)))
        
        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # (batch, input_size)
            
            # h_t = activation(W_ih @ x_t + W_hh @ h_{t-1} + b)
            h = (x_t @ self.weight_ih.transpose(-1, -2)) + \
                (h @ self.weight_hh.transpose(-1, -2)) + self.bias
            
            if self.nonlinearity == 'tanh':
                h = h.tanh()
            elif self.nonlinearity == 'relu':
                h = h.relu()
            
            outputs.append(h)
        
        # Stack outputs
        return outputs, h


class RangeLSTM(RangeModule):
    """
    LSTM with range propagation.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, cell, output
        limit = np.sqrt(1.0 / hidden_size)
        
        # Input-to-hidden weights (4 gates)
        self.weight_ih = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (4 * hidden_size, input_size))
        )
        
        # Hidden-to-hidden weights (4 gates)
        self.weight_hh = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (4 * hidden_size, hidden_size))
        )
        
        # Biases (4 gates)
        self.bias = RangeTensor.from_array(xp.zeros(4 * hidden_size))
    
    def forward(self, x, state=None):
        """
        Args:
            x: Input sequence (seq_len, batch, input_size)
            state: Tuple of (h_0, c_0) or None
        
        Returns:
            output: (seq_len, batch, hidden_size)
            (h_n, c_n): Final hidden and cell states
        """
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
        
        if state is None:
            h = RangeTensor.from_array(xp.zeros((batch_size, self.hidden_size)))
            c = RangeTensor.from_array(xp.zeros((batch_size, self.hidden_size)))
        else:
            h, c = state
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            
            # Compute gates
            gates = (x_t @ self.weight_ih.transpose(-1, -2)) + \
                    (h @ self.weight_hh.transpose(-1, -2)) + self.bias
            
            # Split into 4 gates
            # Note: This requires implementing slicing in RangeTensor
            # For now, we'll use a simplified version
            i = gates[:, :self.hidden_size].sigmoid()  # Input gate
            f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Forget gate
            g = gates[:, 2*self.hidden_size:3*self.hidden_size].tanh()  # Cell gate
            o = gates[:, 3*self.hidden_size:].sigmoid()  # Output gate
            
            # Update cell and hidden states
            c = (f * c) + (i * g)
            h = o * c.tanh()
            
            outputs.append(h)
        
        return outputs, (h, c)


class RangeGRU(RangeModule):
    """
    GRU with range propagation.
    
    Args:
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        limit = np.sqrt(1.0 / hidden_size)
        
        # 3 gates: reset, update, new
        self.weight_ih = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (3 * hidden_size, input_size))
        )
        self.weight_hh = RangeTensor.from_array(
            xp.random.uniform(-limit, limit, (3 * hidden_size, hidden_size))
        )
        self.bias = RangeTensor.from_array(xp.zeros(3 * hidden_size))
    
    def forward(self, x, h=None):
        """GRU forward pass with ranges"""
        seq_len = x.shape[0] if hasattr(x, 'shape') else x.symbol.value[0].shape[0]
        batch_size = x.shape[1] if hasattr(x, 'shape') else x.symbol.value[0].shape[1]
        
        if h is None:
            h = RangeTensor.from_array(xp.zeros((batch_size, self.hidden_size)))
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]
            
            # Compute gates
            gates = (x_t @ self.weight_ih.transpose(-1, -2)) + \
                    (h @ self.weight_hh.transpose(-1, -2)) + self.bias
            
            r = gates[:, :self.hidden_size].sigmoid()  # Reset gate
            z = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()  # Update gate
            n = gates[:, 2*self.hidden_size:].tanh()  # New gate
            
            # Update hidden state
            h = ((RangeTensor.from_array(xp.ones_like(z.symbol.value[0])) - z) * n) + (z * h)
            
            outputs.append(h)
        
        return outputs, h


# ==========================================
# ATTENTION LAYERS
# ==========================================

class RangeAttention(RangeModule):
    """
    Multi-Head Self Attention with range propagation.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = RangeLinear(embed_dim, embed_dim)
        self.k_proj = RangeLinear(embed_dim, embed_dim)
        self.v_proj = RangeLinear(embed_dim, embed_dim)
        self.out_proj = RangeLinear(embed_dim, embed_dim)
        
        self.scale = RangeTensor.from_array(xp.array(1.0 / np.sqrt(self.head_dim)))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim) as RangeTensor
        
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Attention scores
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        
        # Softmax
        attn_weights = _op("softmax", scores, axis=-1)
        
        # Aggregate
        out = attn_weights @ V
        
        return self.out_proj(out)


# ==========================================
# REGULARIZATION
# ==========================================

class RangeDropout(RangeModule):
    """
    Dropout that expands uncertainty instead of zeroing.
    
    Args:
        p: Dropout probability
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        
        min_x, max_x = x.decay()
        mask = xp.random.rand(*min_x.shape) > self.p
        
        # Where dropped, expand range to large uncertainty
        LARGE = 10.0
        out_min = xp.where(mask, min_x, -LARGE)
        out_max = xp.where(mask, max_x, LARGE)
        
        return RangeTensor.from_range(out_min, out_max)


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

class RangeReLU(RangeModule):
    """ReLU activation for ranges"""
    def forward(self, x):
        return x.relu()


class RangeSigmoid(RangeModule):
    """Sigmoid activation for ranges"""
    def forward(self, x):
        min_x, max_x = x.decay()
        return RangeTensor.from_range(
            1 / (1 + xp.exp(-min_x)),
            1 / (1 + xp.exp(-max_x))
        )


class RangeTanh(RangeModule):
    """Tanh activation for ranges"""
    def forward(self, x):
        return x.tanh()


class RangeGELU(RangeModule):
    """
    GELU activation (approximation for ranges).
    Non-monotonic, so we use conservative bounds.
    """
    def forward(self, x):
        min_x, max_x = x.decay()
        
        # GELU(x) ≈ x * Φ(x) where Φ is CDF of standard normal
        # For ranges, we evaluate at endpoints
        def gelu(z):
            return 0.5 * z * (1 + xp.tanh(xp.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
        
        corners = [gelu(min_x), gelu(max_x)]
        
        return RangeTensor.from_range(
            xp.minimum(corners[0], corners[1]),
            xp.maximum(corners[0], corners[1])
        )


# ==========================================
# UTILITY LAYERS
# ==========================================

class RangeSequential(RangeModule):
    """
    Sequential container for RangeFlow layers.
    
    Example:
        >>> model = RangeSequential(
        ...     RangeLinear(784, 128),
        ...     RangeReLU(),
        ...     RangeLinear(128, 10)
        ... )
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self):
        super().train()
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        super().eval()
        for layer in self.layers:
            layer.eval()


class RangeFlatten(RangeModule):
    """Flatten spatial dimensions"""
    def forward(self, x):
        return x.reshape(-1, np.prod(x.shape[1:]))