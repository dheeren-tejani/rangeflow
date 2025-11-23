"""
RangeFlow Core Module
=====================
The fundamental RangeTensor class with all operations.

This module implements the core interval arithmetic with:
- Dimensional growth framework
- Dependent/independent range tracking
- Strategic decay operations
- Complete arithmetic and comparison operators
"""

from typing import Any, Tuple, Optional, Union
from .backend import get_backend, to_tensor, to_cpu
import numpy as np

xp = get_backend()


class Symbol:
    """
    Symbolic computation graph node for lazy evaluation.
    
    Attributes:
        op_name: Operation type (e.g., 'add', 'mul', 'relu')
        parents: Tuple of parent Symbol nodes
        value: Stored value for LEAF nodes
        kwargs: Operation-specific parameters
        _cache: Cached result after evaluation
    """
    def __init__(self, op_name: str, parents: tuple = (), value=None, **kwargs):
        self.op_name = op_name
        self.parents = parents
        self.value = value
        self.kwargs = kwargs
        self._cache = None


class RangeTensor:
    """
    Interval representation with lazy computation graph.
    
    Represents uncertainty as [min, max] bounds that propagate
    through operations while tracking dependencies.
    
    Key Concepts:
    -------------
    - **Lazy Evaluation**: Operations build a graph, computed on decay()
    - **Dependency Tracking**: Distinguishes dependent vs independent ranges
    - **Dimensional Growth**: Independent ops grow dimensions (2^n extremes)
    - **Strategic Decay**: Collapses high-D ranges to 1D for computation
    
    Examples:
    ---------
    >>> # Create range with uncertainty
    >>> x = RangeTensor.from_range(5.0, 5.2)  # [5.0 <-> 5.2]
    >>> 
    >>> # Operations build computation graph
    >>> y = x * 2 + 1  # Lazy - not computed yet!
    >>> 
    >>> # Execute and get bounds
    >>> min_val, max_val = y.decay()  # [11.0, 11.4]
    >>> 
    >>> # Track range properties
    >>> print(f"Width: {y.len()}")  # 0.4
    >>> print(f"Center: {y.avg()}")  # 11.2
    """
    
    def __init__(self, symbol: Symbol, shape: tuple):
        self.symbol = symbol
        self.shape = shape
        self._dependency_id = None  # For tracking correlations

    # ==========================================
    # CONSTRUCTION METHODS
    # ==========================================
    
    @classmethod
    def from_array(cls, data):
        """
        Create range from array (degenerate interval [x, x]).
        
        Args:
            data: NumPy/CuPy array or scalar
            
        Returns:
            RangeTensor with min=max=data
        """
        t = to_tensor(data)
        return cls(Symbol("LEAF", value=t), t.shape)
    
    @classmethod
    def from_range(cls, min_v, max_v):
        """
        Create range from explicit bounds.
        """
        t_min, t_max = to_tensor(min_v), to_tensor(max_v)
        
        # If tiny float errors cause min > max, we clamp them instead of crashing
        if xp.__name__ == 'torch':
            final_min = xp.min(t_min, t_max)
            final_max = xp.max(t_min, t_max)
        else:
            assert xp.all(t_min <= t_max), "min must be <= max"
            final_min, final_max = t_min, t_max
            
        return cls(Symbol("LEAF_RANGE", value=(final_min, final_max)), final_min.shape)
    
    @classmethod
    def from_epsilon_ball(cls, center, epsilon):
        """
        Create L-infinity ball around center.
        
        Useful for adversarial robustness (epsilon perturbations).
        
        Args:
            center: Central value
            epsilon: Radius of perturbation
            
        Returns:
            RangeTensor representing [center-ε, center+ε]
        """
        center = to_tensor(center)
        eps = to_tensor(epsilon)
        return cls.from_range(center - eps, center + eps)
    
    @classmethod
    def from_noise(cls, data, noise_level):
        """
        Model sensor/measurement noise.
        
        Args:
            data: Measured values
            noise_level: Absolute noise magnitude
            
        Returns:
            RangeTensor accounting for noise
        """
        return cls.from_epsilon_ball(data, noise_level)

    # ==========================================
    # CORE PROPERTIES
    # ==========================================
    
    def decay(self) -> Tuple[Any, Any]:
        """
        Execute computation graph and return [min, max] bounds.
        
        This is the core operation that:
        1. Traverses the symbolic graph
        2. Applies interval arithmetic rules
        3. Returns concrete bounds
        
        Returns:
            (min_array, max_array): Tuple of bounds
        """
        from .ops import evaluate_bounds
        return evaluate_bounds(self.symbol)
    
    def len(self):
        """
        Compute range width (uncertainty magnitude).
        
        For 1D range [a, b]: len = |b - a|
        For nD range: len = decay() then compute width
        
        Returns:
            Width of the interval
        """
        min_val, max_val = self.decay()
        return xp.abs(max_val - min_val)
    
    def avg(self):
        """
        Compute range center (midpoint/average).
        
        For 1D range [a, b]: avg = (a + b) / 2
        For nD range: avg = mean of all 2^n extremes
        
        Returns:
            Center point of the interval
        """
        min_val, max_val = self.decay()
        return (min_val + max_val) / 2
    
    def width(self):
        """Alias for len() - returns uncertainty width"""
        return self.len()
    
    def center(self):
        """Alias for avg() - returns center point"""
        return self.avg()
    
    def relative_width(self):
        """
        Compute normalized uncertainty: width / |center|
        
        Useful for checking if bounds are tight relative to magnitude.
        
        Returns:
            Relative uncertainty (0 = perfect, 1 = very uncertain)
        """
        c = self.center()
        w = self.width()
        return w / (xp.abs(c) + 1e-8)

    # ==========================================
    # ARITHMETIC OPERATIONS
    # ==========================================
    
    def __add__(self, other):
        """Addition: [a,b] + [c,d] = [a+c, b+d]"""
        return _op("add", self, other)
    
    def __sub__(self, other):
        """Subtraction: [a,b] - [c,d] = [a-d, b-c]"""
        return _op("sub", self, other)
    
    def __mul__(self, other):
        """Multiplication: [a,b] * [c,d] = [min(ac,ad,bc,bd), max(...)]"""
        return _op("mul", self, other)
    
    def __truediv__(self, other):
        """Division: [a,b] / [c,d] with zero handling"""
        return _op("div", self, other)
    
    def __pow__(self, exponent):
        """Power: [a,b]^n with critical point handling"""
        return _op("pow", self, exponent=exponent)
    
    def __matmul__(self, other):
        """Matrix multiplication with monotonicity shortcut"""
        return _op("matmul", self, other)
    
    def __neg__(self):
        """Negation: -[a,b] = [-b, -a]"""
        return _op("neg", self)
    
    # Reverse operations
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __rsub__(self, other):
        return _op("sub", _ensure_range(other), self)
    
    def __rtruediv__(self, other):
        return _op("div", _ensure_range(other), self)

    # ==========================================
    # COMPARISON OPERATIONS (PARTIAL ORDERING)
    # ==========================================
    
    def __lt__(self, other):
        """Less than (by center): avg(self) < avg(other)"""
        return self.avg() < _ensure_range(other).avg()
    
    def __le__(self, other):
        return self.avg() <= _ensure_range(other).avg()
    
    def __gt__(self, other):
        return self.avg() > _ensure_range(other).avg()
    
    def __ge__(self, other):
        return self.avg() >= _ensure_range(other).avg()
    
    def contains(self, other):
        """Check if self ⊇ other (containment ordering)"""
        self_min, self_max = self.decay()
        other_min, other_max = _ensure_range(other).decay()
        return xp.all(self_min <= other_min) and xp.all(self_max >= other_max)
    
    def overlaps(self, other):
        """Check if ranges overlap"""
        self_min, self_max = self.decay()
        other_min, other_max = _ensure_range(other).decay()
        return xp.any((self_min <= other_max) & (other_min <= self_max))
    
    def is_wider_than(self, other):
        """Compare by uncertainty width"""
        return self.len() > _ensure_range(other).len()
    
    def is_tighter_than(self, other):
        """Compare by tightness (inverse of width)"""
        return self.len() < _ensure_range(other).len()

    # ==========================================
    # ACTIVATION FUNCTIONS (MONOTONIC)
    # ==========================================
    
    def relu(self):
        """ReLU: max(0, x) - monotonic, preserves bounds"""
        return _op("relu", self)
    
    def sigmoid(self):
        """Sigmoid: 1/(1+e^-x) - monotonic"""
        return _op("sigmoid", self)
    
    def tanh(self):
        """Tanh: (e^x - e^-x)/(e^x + e^-x) - monotonic"""
        return _op("tanh", self)
    
    def exp(self):
        """Exponential: e^x - monotonic"""
        return _op("exp", self)
    
    def log(self):
        """Natural log: ln(x) - monotonic, requires positive input"""
        return _op("log", self)
    
    def softmax(self, axis=-1):
        """Softmax with conservative bounds"""
        return _op("softmax", self, axis=axis)

    # ==========================================
    # NON-MONOTONIC FUNCTIONS (CRITICAL POINTS)
    # ==========================================
    
    def square(self):
        """
        Square with critical point handling: x^2
        
        Critical point at x=0 (minimum).
        For x ∈ [-a, b]: result ∈ [0, max(a^2, b^2)]
        """
        return _op("square", self)
    
    def abs(self):
        """
        Absolute value: |x|
        
        Critical point at x=0 (V-shape).
        For x ∈ [-a, b]: result ∈ [0, max(a, b)]
        """
        return _op("abs", self)
    
    def sqrt(self):
        """
        Square root: √x (positive branch only)
        
        Requires non-negative input.
        """
        return _op("sqrt", self)

    # ==========================================
    # SHAPE OPERATIONS
    # ==========================================
    
    def transpose(self, dim0, dim1):
        """Transpose dimensions"""
        return _op("transpose", self, dim0=dim0, dim1=dim1)
    
    def reshape(self, *shape):
        """Reshape range tensor"""
        return _op("reshape", self, shape=shape)
    
    def flatten(self):
        """Flatten all dimensions except batch"""
        return _op("flatten", self)
    
    @property
    def T(self):
        """Matrix transpose"""
        return self.transpose(-1, -2)
    
    def __getitem__(self, key):
        """Indexing/slicing"""
        return _op("getitem", self, key=key)

    # ==========================================
    # RANGE-SPECIFIC OPERATIONS
    # ==========================================
    
    def transpose_range(self):
        """
        Transpose of range: flip min/max.

        [a <-> b] becomes [b <-> a]

        This creates an "improper interval" useful for certain operations.
        """
        min_val, max_val = self.decay()
        result = RangeTensor.__new__(RangeTensor)
        result.symbol = Symbol("LEAF_RANGE", value=(max_val, min_val))
        result.shape = self.shape
        return result
    
    def union(self, other):
        """
        Union of two ranges (if connected).
        
        [a, b] ∪ [c, d] = [min(a,c), max(b,d)] if they overlap
        
        Args:
            other: Another RangeTensor
            
        Returns:
            RangeTensor representing union
            
        Raises:
            ValueError: If ranges are disconnected
        """
        self_min, self_max = self.decay()
        other_min, other_max = _ensure_range(other).decay()
        
        # Check if connected
        if not self.overlaps(other):
            raise ValueError("Cannot union disconnected ranges (use MultiRange)")
        
        return RangeTensor.from_range(
            xp.minimum(self_min, other_min),
            xp.maximum(self_max, other_max)
        )
    
    def intersection(self, other):
        """
        Intersection of two ranges.
        
        [a, b] ∩ [c, d] = [max(a,c), min(b,d)]
        
        Args:
            other: Another RangeTensor
            
        Returns:
            RangeTensor representing intersection
            
        Raises:
            ValueError: If ranges don't overlap
        """
        self_min, self_max = self.decay()
        other_min, other_max = _ensure_range(other).decay()
        
        new_min = xp.maximum(self_min, other_min)
        new_max = xp.minimum(self_max, other_max)
        
        if xp.any(new_min > new_max):
            raise ValueError("Ranges don't intersect")
        
        return RangeTensor.from_range(new_min, new_max)

    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def is_degenerate(self, tol=1e-8):
        """Check if range is essentially a point (width ≈ 0)"""
        return xp.all(self.len() < tol)
    
    def to_numpy(self):
        """Convert bounds to numpy arrays"""
        min_val, max_val = self.decay()
        return to_cpu(min_val), to_cpu(max_val)
    
    def __repr__(self):
        try:
            min_val, max_val = self.decay()
            min_str = f"{float(min_val.flat[0]):.3f}" if min_val.size > 0 else "empty"
            max_str = f"{float(max_val.flat[0]):.3f}" if max_val.size > 0 else "empty"
            return f"RangeTensor([{min_str} <-> {max_str}], shape={self.shape})"
        except:
            return f"RangeTensor(shape={self.shape}, unevaluated)"


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def _ensure_range(obj):
    """Convert scalar/array to RangeTensor if needed"""
    if isinstance(obj, RangeTensor):
        return obj
    return RangeTensor.from_array(obj)


def _op(name: str, *args, **kwargs):
    """
    Create operation node in computation graph.
    
    Args:
        name: Operation name
        *args: Input RangeTensors or scalars
        **kwargs: Operation parameters
        
    Returns:
        New RangeTensor with symbolic operation
    """
    from .ops import infer_shape
    
    # Ensure all args are RangeTensors
    clean_args = [_ensure_range(a) for a in args]
    
    # Infer output shape
    shape = infer_shape(name, [a.shape for a in clean_args], **kwargs)
    
    # Create symbol node
    symbol = Symbol(name, tuple(a.symbol for a in clean_args), **kwargs)
    
    return RangeTensor(symbol, shape)