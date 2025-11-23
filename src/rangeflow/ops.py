"""
RangeFlow Operations
====================
Complete implementation of interval arithmetic operations.
Includes all mathematical operations needed for neural networks.
Universal Engine: PyTorch (GPU training) + NumPy/CuPy (legacy support)
"""

from .backend import get_backend
import numpy as np

xp = get_backend()

# Detect backend type
BACKEND = 'torch' if hasattr(xp, 'nn') else 'numpy'


def infer_shape(op, shapes, **kwargs):
    """
    Calculates output shape without running the math (lazy execution).
    
    Args:
        op: Operation name
        shapes: List of input shapes
        **kwargs: Operation-specific parameters
    
    Returns:
        Output shape tuple
    """
    if not shapes:
        return ()
    
    s0 = shapes[0]
    
    # Element-wise operations
    if op in ["add", "sub", "mul", "div", "pow", "square", "sqrt", "abs", "neg"]:
        return s0
    
    # Matrix operations
    if op == "matmul":
        # (..., N, M) @ (..., M, K) -> (..., N, K)
        return s0[:-1] + (shapes[1][-1],)
    
    # Shape operations
    if op == "transpose":
        l = list(s0)
        d0, d1 = kwargs['dim0'], kwargs['dim1']
        l[d0], l[d1] = l[d1], l[d0]
        return tuple(l)
    
    if op == "reshape":
        return kwargs['shape']
    
    if op == "flatten":
        return (s0[0], int(np.prod(s0[1:])))
    
    # Convolution
    if op == "conv2d":
        # Input: (N, C_in, H, W)
        # Weight: (C_out, C_in, K, K)
        # Output: (N, C_out, H_out, W_out)
        N, C_in, H, W = s0
        C_out = shapes[1][0]
        K = shapes[1][2]
        stride = kwargs.get('stride', 1)
        padding = kwargs.get('padding', 0)
        
        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1
        return (N, C_out, H_out, W_out)
    
    # Pooling
    if op in ["max_pool2d", "avg_pool2d"]:
        N, C, H, W = s0
        kernel_size = kwargs['kernel_size']
        stride = kwargs.get('stride', None)
        if stride is None:
            stride = kernel_size  # Default stride = kernel_size
        padding = kwargs.get('padding', 0)
        
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        return (N, C, H_out, W_out)
    
    # Reduction operations
    if op in ["sum", "mean", "max", "min"]:
        axis = kwargs.get('axis', None)
        keepdims = kwargs.get('keepdims', False)
        
        if axis is None:
            return () if not keepdims else tuple(1 for _ in s0)
        
        if isinstance(axis, int):
            axis = (axis,)
        
        new_shape = []
        for i, dim in enumerate(s0):
            if i in axis:
                if keepdims:
                    new_shape.append(1)
            else:
                new_shape.append(dim)
        return tuple(new_shape)
    
    # Indexing/slicing
    if op == "getitem":
        # Simplified - actual implementation would be more complex
        return s0
    
    # Activations and other ops
    if op in ["relu", "sigmoid", "tanh", "exp", "log", "softmax"]:
        return s0
    
    # Default: preserve shape
    return s0


def evaluate_bounds(node):
    """
    The FCD (Flowing Conservative Decay) Execution Engine.
    
    Traverses the symbolic computation graph and computes [Min, Max] bounds
    using interval arithmetic with monotonicity shortcuts.
    
    Universal Backend: Automatically uses PyTorch or NumPy/CuPy.
    
    Args:
        node: Symbol node from computation graph
    
    Returns:
        (min_bound, max_bound): Tuple of tensors representing interval bounds
    """
    # Check cache
    if node._cache is not None:
        return node._cache
    
    # --- LEAF NODES ---
    if node.op_name == "LEAF":
        return node.value, node.value
    
    if node.op_name == "LEAF_RANGE":
        return node.value
    
    # --- RECURSION ---
    parents = [evaluate_bounds(p) for p in node.parents]
    
    rl, rh = None, None
    
    # ==========================================
    # HELPER FUNCTIONS (Universal Backend)
    # ==========================================
    
    def clamp(t, min_v=None, max_v=None):
        """Universal clamp/clip for PyTorch and NumPy"""
        if BACKEND == 'torch':
            return t.clamp(min=min_v, max=max_v)
        else:
            return xp.clip(t, min_v, max_v)
    
    def min_max_stack(tensors):
        """Universal min/max reduction across tensor stack"""
        if BACKEND == 'torch':
            stacked = xp.stack(tensors)
            return xp.min(stacked, dim=0)[0], xp.max(stacked, dim=0)[0]
        else:
            # NumPy/CuPy: manual reduction
            mn, mx = tensors[0], tensors[0]
            for t in tensors[1:]:
                mn = xp.minimum(mn, t)
                mx = xp.maximum(mx, t)
            return mn, mx
    
    # ==========================================
    # ARITHMETIC OPERATIONS
    # ==========================================
    
    if node.op_name == "add":
        (al, ah), (bl, bh) = parents
        rl, rh = al + bl, ah + bh
    
    elif node.op_name == "sub":
        (al, ah), (bl, bh) = parents
        rl, rh = al - bh, ah - bl
    
    elif node.op_name == "mul":
        (al, ah), (bl, bh) = parents
        # All 4 products (handles negative values)
        p1, p2, p3, p4 = al*bl, al*bh, ah*bl, ah*bh
        rl, rh = min_max_stack([p1, p2, p3, p4])
    
    elif node.op_name == "div":
        (al, ah), (bl, bh) = parents
        # Ensure denominator doesn't contain zero
        eps = 1e-8
        if BACKEND == 'torch':
            bl = xp.where(xp.abs(bl) < eps, eps * xp.sign(bl), bl)
            bh = xp.where(xp.abs(bh) < eps, eps * xp.sign(bh), bh)
        else:
            bl = xp.where(xp.abs(bl) < eps, eps * xp.sign(bl), bl)
            bh = xp.where(xp.abs(bh) < eps, eps * xp.sign(bh), bh)
        
        # All 4 divisions
        d1, d2, d3, d4 = al/bl, al/bh, ah/bl, ah/bh
        rl, rh = min_max_stack([d1, d2, d3, d4])
    
    elif node.op_name == "pow":
        (al, ah) = parents[0]
        exponent = node.kwargs.get('exponent', 2)
        
        if exponent % 2 == 0:  # Even power
            # Check if interval contains zero (Quantum Fix)
            contains_zero = xp.any(al <= 0) and xp.any(ah >= 0)
            if contains_zero:
                rl = xp.zeros_like(al)
                rh_candidates = [xp.abs(al)**exponent, xp.abs(ah)**exponent]
                rh = xp.maximum(rh_candidates[0], rh_candidates[1]) if BACKEND != 'torch' else xp.max(xp.stack(rh_candidates), dim=0)[0]
            else:
                p1, p2 = al**exponent, ah**exponent
                rl, rh = min_max_stack([p1, p2])
        else:  # Odd power (monotonic)
            rl, rh = al**exponent, ah**exponent
    
    elif node.op_name == "neg":
        (al, ah) = parents[0]
        rl, rh = -ah, -al
    
    # ==========================================
    # NON-MONOTONIC FUNCTIONS (CRITICAL POINTS)
    # ==========================================
    
    elif node.op_name == "square":
        (al, ah) = parents[0]
        # Critical point at x=0 (Quantum Fix)
        contains_zero = xp.any(al <= 0) and xp.any(ah >= 0)
        if contains_zero:
            # Interval contains zero (minimum of x^2)
            rl = xp.zeros_like(al)
            rh = xp.maximum(al**2, ah**2) if BACKEND != 'torch' else xp.max(xp.stack([al**2, ah**2]), dim=0)[0]
        else:
            # Doesn't contain zero - monotonic
            p1, p2 = al**2, ah**2
            rl, rh = min_max_stack([p1, p2])
    
    elif node.op_name == "sqrt":
        (al, ah) = parents[0]
        # Ensure non-negative
        al = clamp(al, min_v=0)
        ah = clamp(ah, min_v=0)
        rl, rh = xp.sqrt(al), xp.sqrt(ah)
    
    elif node.op_name == "abs":
        (al, ah) = parents[0]
        # Critical point at x=0 (Quantum Fix)
        contains_zero = xp.any(al <= 0) and xp.any(ah >= 0)
        if contains_zero:
            # Interval contains zero
            rl = xp.zeros_like(al)
            rh = xp.maximum(xp.abs(al), xp.abs(ah)) if BACKEND != 'torch' else xp.max(xp.stack([xp.abs(al), xp.abs(ah)]), dim=0)[0]
        else:
            # Doesn't contain zero
            p1, p2 = xp.abs(al), xp.abs(ah)
            rl, rh = min_max_stack([p1, p2])
    
    # ==========================================
    # MATRIX OPERATIONS
    # ==========================================
    
    elif node.op_name == "matmul":
        (al, ah), (bl, bh) = parents
        # Monotonic shortcut (critical optimization!)
        # Note: Removed extra transpose - graph handles this via transpose node
        w_pos = clamp(bh, min_v=0)
        w_neg = clamp(bl, max_v=0)
        rl = (al @ w_pos) + (ah @ w_neg)
        rh = (ah @ w_pos) + (al @ w_neg)
    
    # ==========================================
    # CONVOLUTION (Tensor-Native)
    # ==========================================
    
    elif node.op_name == "conv2d":
        (min_x, max_x) = parents[0]
        (min_w, max_w) = parents[1]
        
        if len(parents) > 2:
            (min_b, max_b) = parents[2]
        else:
            min_b, max_b = 0, 0
        
        stride = node.kwargs.get('stride', 1)
        padding = node.kwargs.get('padding', 0)
        
        # Tensor-Native: Keep as PyTorch tensors (no conversion to NumPy)
        if BACKEND == 'torch':
            import torch.nn.functional as F
            
            # Monotonic convolution
            w_pos = clamp(max_w, min_v=0)
            w_neg = clamp(min_w, max_v=0)
            
            res_min = F.conv2d(min_x, w_pos, stride=stride, padding=padding) + \
                      F.conv2d(max_x, w_neg, stride=stride, padding=padding)
            
            res_max = F.conv2d(max_x, w_pos, stride=stride, padding=padding) + \
                      F.conv2d(min_x, w_neg, stride=stride, padding=padding)
            
            if len(parents) > 2:
                res_min += min_b.view(1, -1, 1, 1)
                res_max += max_b.view(1, -1, 1, 1)
            
            rl, rh = res_min, res_max
        else:
            raise RuntimeError("Conv2d requires PyTorch backend")
    
    # ==========================================
    # POOLING OPERATIONS (Tensor-Native)
    # ==========================================
    
    elif node.op_name == "max_pool2d":
        (min_x, max_x) = parents[0]
        kernel_size = node.kwargs['kernel_size']
        stride = node.kwargs.get('stride', None)
        if stride is None:
            stride = kernel_size  # Default stride = kernel_size
        padding = node.kwargs.get('padding', 0)
        
        if BACKEND == 'torch':
            import torch.nn.functional as F
            
            # For max pooling, we take max of both bounds
            rl = F.max_pool2d(min_x, kernel_size, stride, padding)
            rh = F.max_pool2d(max_x, kernel_size, stride, padding)
        else:
            raise RuntimeError("Pooling requires PyTorch backend")
    
    elif node.op_name == "avg_pool2d":
        (min_x, max_x) = parents[0]
        kernel_size = node.kwargs['kernel_size']
        stride = node.kwargs.get('stride', None)
        if stride is None:
            stride = kernel_size  # Default stride = kernel_size
        padding = node.kwargs.get('padding', 0)
        
        if BACKEND == 'torch':
            import torch.nn.functional as F
            
            # Average is linear, so we average both bounds
            rl = F.avg_pool2d(min_x, kernel_size, stride, padding)
            rh = F.avg_pool2d(max_x, kernel_size, stride, padding)
        else:
            raise RuntimeError("Pooling requires PyTorch backend")
    
    # ==========================================
    # ACTIVATION FUNCTIONS (MONOTONIC)
    # ==========================================
    
    elif node.op_name == "relu":
        (al, ah) = parents[0]
        rl, rh = clamp(al, min_v=0), clamp(ah, min_v=0)
    
    elif node.op_name == "sigmoid":
        (al, ah) = parents[0]
        if BACKEND == 'torch':
            rl = xp.sigmoid(al)
            rh = xp.sigmoid(ah)
        else:
            rl = 1 / (1 + xp.exp(-al))
            rh = 1 / (1 + xp.exp(-ah))
    
    elif node.op_name == "tanh":
        (al, ah) = parents[0]
        rl, rh = xp.tanh(al), xp.tanh(ah)
    
    elif node.op_name == "exp":
        (al, ah) = parents[0]
        rl, rh = xp.exp(al), xp.exp(ah)
    
    elif node.op_name == "log":
        (al, ah) = parents[0]
        # Ensure positive
        al = clamp(al, min_v=1e-8)
        ah = clamp(ah, min_v=1e-8)
        rl, rh = xp.log(al), xp.log(ah)
    
    elif node.op_name == "softmax":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', -1)
        
        # Conservative softmax bounds
        exp_l = xp.exp(al - xp.max(ah, axis=axis, keepdims=True))
        exp_h = xp.exp(ah - xp.min(al, axis=axis, keepdims=True))
        
        sum_h = xp.sum(exp_h, axis=axis, keepdims=True)
        sum_l = xp.sum(exp_l, axis=axis, keepdims=True)
        
        rl = exp_l / sum_h
        rh = exp_h / sum_l
    
    # ==========================================
    # SHAPE OPERATIONS (Now Fully Implemented)
    # ==========================================
    
    elif node.op_name == "reshape":
        (al, ah) = parents[0]
        s = node.kwargs['shape']
        if BACKEND == 'torch':
            rl, rh = al.reshape(s), ah.reshape(s)
        else:
            rl, rh = xp.reshape(al, s), xp.reshape(ah, s)
    
    elif node.op_name == "transpose":
        (al, ah) = parents[0]
        d0, d1 = node.kwargs['dim0'], node.kwargs['dim1']
        if BACKEND == 'torch':
            rl, rh = al.transpose(d0, d1), ah.transpose(d0, d1)
        else:
            rl, rh = xp.swapaxes(al, d0, d1), xp.swapaxes(ah, d0, d1)
    
    elif node.op_name == "flatten":
        (al, ah) = parents[0]
        start_dim = node.kwargs.get('start_dim', 1)
        if BACKEND == 'torch':
            rl = al.flatten(start_dim)
            rh = ah.flatten(start_dim)
        else:
            batch_size = al.shape[0]
            rl = xp.reshape(al, (batch_size, -1))
            rh = xp.reshape(ah, (batch_size, -1))
    
    # ==========================================
    # REDUCTION OPERATIONS
    # ==========================================
    
    elif node.op_name == "sum":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        
        if BACKEND == 'torch':
            if axis is not None:
                rl = xp.sum(al, dim=axis, keepdim=keepdims)
                rh = xp.sum(ah, dim=axis, keepdim=keepdims)
            else:
                rl = xp.sum(al)
                rh = xp.sum(ah)
        else:
            rl = xp.sum(al, axis=axis, keepdims=keepdims)
            rh = xp.sum(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "mean":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        
        if BACKEND == 'torch':
            if axis is not None:
                rl = xp.mean(al, dim=axis, keepdim=keepdims)
                rh = xp.mean(ah, dim=axis, keepdim=keepdims)
            else:
                rl = xp.mean(al)
                rh = xp.mean(ah)
        else:
            rl = xp.mean(al, axis=axis, keepdims=keepdims)
            rh = xp.mean(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "max":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        
        if BACKEND == 'torch':
            if axis is not None:
                rl = xp.max(al, dim=axis, keepdim=keepdims)[0]
                rh = xp.max(ah, dim=axis, keepdim=keepdims)[0]
            else:
                rl = xp.max(al)
                rh = xp.max(ah)
        else:
            rl = xp.max(al, axis=axis, keepdims=keepdims)
            rh = xp.max(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "min":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        
        if BACKEND == 'torch':
            if axis is not None:
                rl = xp.min(al, dim=axis, keepdim=keepdims)[0]
                rh = xp.min(ah, dim=axis, keepdim=keepdims)[0]
            else:
                rl = xp.min(al)
                rh = xp.min(ah)
        else:
            rl = xp.min(al, axis=axis, keepdims=keepdims)
            rh = xp.min(ah, axis=axis, keepdims=keepdims)
    
    # ==========================================
    # INDEXING/SLICING
    # ==========================================
    
    elif node.op_name == "getitem":
        (al, ah) = parents[0]
        key = node.kwargs['key']
        rl, rh = al[key], ah[key]
    
    elif node.op_name == "concatenate":
        # Concatenate multiple ranges
        axis = node.kwargs.get('axis', 0)
        mins = [p[0] for p in parents]
        maxs = [p[1] for p in parents]
        
        if BACKEND == 'torch':
            rl = xp.cat(mins, dim=axis)
            rh = xp.cat(maxs, dim=axis)
        else:
            rl = xp.concatenate(mins, axis=axis)
            rh = xp.concatenate(maxs, axis=axis)
    
    # Error checking
    if rl is None:
        raise NotImplementedError(f"Operation '{node.op_name}' is not implemented in ops.py")
    
    # Cache result
    node._cache = (rl, rh)
    return rl, rh