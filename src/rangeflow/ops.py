"""
RangeFlow Operations
====================
Complete implementation of interval arithmetic operations.
Includes all mathematical operations needed for neural networks.
"""

from .backend import get_backend
import numpy as np

xp = get_backend()


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
        return (s0[0], np.prod(s0[1:]))
    
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
        stride = kwargs.get('stride', kernel_size)
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
        rl = xp.minimum(xp.minimum(p1, p2), xp.minimum(p3, p4))
        rh = xp.maximum(xp.maximum(p1, p2), xp.maximum(p3, p4))
    
    elif node.op_name == "div":
        (al, ah), (bl, bh) = parents
        # Ensure denominator doesn't contain zero
        eps = 1e-8
        bl = xp.where(xp.abs(bl) < eps, eps * xp.sign(bl), bl)
        bh = xp.where(xp.abs(bh) < eps, eps * xp.sign(bh), bh)
        
        # All 4 divisions
        d1, d2, d3, d4 = al/bl, al/bh, ah/bl, ah/bh
        rl = xp.minimum(xp.minimum(d1, d2), xp.minimum(d3, d4))
        rh = xp.maximum(xp.maximum(d1, d2), xp.maximum(d3, d4))
    
    elif node.op_name == "pow":
        (al, ah) = parents[0]
        exponent = node.kwargs.get('exponent', 2)
        
        if exponent % 2 == 0:  # Even power
            # Check if interval contains zero
            if xp.any(al <= 0) and xp.any(ah >= 0):
                rl = xp.zeros_like(al)
                rh = xp.maximum(xp.abs(al)**exponent, xp.abs(ah)**exponent)
            else:
                rl = xp.minimum(al**exponent, ah**exponent)
                rh = xp.maximum(al**exponent, ah**exponent)
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
        # Critical point at x=0
        if xp.any(al <= 0) and xp.any(ah >= 0):
            # Interval contains zero (minimum of x^2)
            rl = xp.zeros_like(al)
            rh = xp.maximum(al**2, ah**2)
        else:
            # Doesn't contain zero - monotonic
            rl = xp.minimum(al**2, ah**2)
            rh = xp.maximum(al**2, ah**2)
    
    elif node.op_name == "sqrt":
        (al, ah) = parents[0]
        # Ensure non-negative
        al = xp.maximum(al, 0)
        ah = xp.maximum(ah, 0)
        rl, rh = xp.sqrt(al), xp.sqrt(ah)
    
    elif node.op_name == "abs":
        (al, ah) = parents[0]
        # Critical point at x=0
        if xp.any(al <= 0) and xp.any(ah >= 0):
            # Interval contains zero
            rl = xp.zeros_like(al)
            rh = xp.maximum(xp.abs(al), xp.abs(ah))
        else:
            # Doesn't contain zero
            rl = xp.minimum(xp.abs(al), xp.abs(ah))
            rh = xp.maximum(xp.abs(al), xp.abs(ah))
    
    # ==========================================
    # MATRIX OPERATIONS
    # ==========================================
    
    elif node.op_name == "matmul":
        (al, ah), (bl, bh) = parents
        # Monotonic shortcut (critical optimization!)
        w_pos = xp.maximum(bh, 0)
        w_neg = xp.minimum(bl, 0)
        rl = (al @ w_pos) + (ah @ w_neg)
        rh = (ah @ w_pos) + (al @ w_neg)
    
    # ==========================================
    # CONVOLUTION
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
        
        # Use PyTorch for optimized convolution
        try:
            import torch
            import torch.nn.functional as F
            
            def to_torch(arr):
                if hasattr(arr, 'get'):
                    arr = arr.get()  # CuPy -> NumPy
                if not isinstance(arr, (int, float)):
                    return torch.from_numpy(np.asarray(arr)).float()
                return torch.tensor(arr).float()
            
            t_min_x = to_torch(min_x)
            t_max_x = to_torch(max_x)
            t_min_w = to_torch(min_w)
            t_max_w = to_torch(max_w)
            
            # Monotonic convolution
            w_pos = torch.clamp(t_max_w, min=0)
            w_neg = torch.clamp(t_min_w, max=0)
            
            res_min = F.conv2d(t_min_x, w_pos, stride=stride, padding=padding) + \
                      F.conv2d(t_max_x, w_neg, stride=stride, padding=padding)
            
            res_max = F.conv2d(t_max_x, w_pos, stride=stride, padding=padding) + \
                      F.conv2d(t_min_x, w_neg, stride=stride, padding=padding)
            
            if len(parents) > 2:
                t_b = to_torch(min_b)
                res_min += t_b.view(1, -1, 1, 1)
                res_max += t_b.view(1, -1, 1, 1)
            
            rl = xp.asarray(res_min.numpy())
            rh = xp.asarray(res_max.numpy())
        
        except ImportError:
            raise RuntimeError("PyTorch required for Conv2d operations")
    
    # ==========================================
    # POOLING OPERATIONS
    # ==========================================
    
    elif node.op_name == "max_pool2d":
        (min_x, max_x) = parents[0]
        kernel_size = node.kwargs['kernel_size']
        stride = node.kwargs.get('stride', kernel_size)
        padding = node.kwargs.get('padding', 0)
        
        try:
            import torch
            import torch.nn.functional as F
            
            def to_torch(arr):
                if hasattr(arr, 'get'):
                    arr = arr.get()
                return torch.from_numpy(np.asarray(arr)).float()
            
            # For max pooling, we take max of both bounds
            rl = F.max_pool2d(to_torch(min_x), kernel_size, stride, padding).numpy()
            rh = F.max_pool2d(to_torch(max_x), kernel_size, stride, padding).numpy()
            
            rl = xp.asarray(rl)
            rh = xp.asarray(rh)
        
        except ImportError:
            raise RuntimeError("PyTorch required for pooling operations")
    
    elif node.op_name == "avg_pool2d":
        (min_x, max_x) = parents[0]
        kernel_size = node.kwargs['kernel_size']
        stride = node.kwargs.get('stride', kernel_size)
        padding = node.kwargs.get('padding', 0)
        
        try:
            import torch
            import torch.nn.functional as F
            
            def to_torch(arr):
                if hasattr(arr, 'get'):
                    arr = arr.get()
                return torch.from_numpy(np.asarray(arr)).float()
            
            # Average is linear, so we average both bounds
            rl = F.avg_pool2d(to_torch(min_x), kernel_size, stride, padding).numpy()
            rh = F.avg_pool2d(to_torch(max_x), kernel_size, stride, padding).numpy()
            
            rl = xp.asarray(rl)
            rh = xp.asarray(rh)
        
        except ImportError:
            raise RuntimeError("PyTorch required for pooling operations")
    
    # ==========================================
    # ACTIVATION FUNCTIONS (MONOTONIC)
    # ==========================================
    
    elif node.op_name == "relu":
        (al, ah) = parents[0]
        rl, rh = xp.maximum(al, 0), xp.maximum(ah, 0)
    
    elif node.op_name == "sigmoid":
        (al, ah) = parents[0]
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
        al = xp.maximum(al, 1e-8)
        ah = xp.maximum(ah, 1e-8)
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
    # SHAPE OPERATIONS
    # ==========================================
    
    elif node.op_name == "reshape":
        (al, ah) = parents[0]
        s = node.kwargs['shape']
        rl, rh = xp.reshape(al, s), xp.reshape(ah, s)
    
    elif node.op_name == "transpose":
        (al, ah) = parents[0]
        d0, d1 = node.kwargs['dim0'], node.kwargs['dim1']
        rl, rh = xp.swapaxes(al, d0, d1), xp.swapaxes(ah, d0, d1)
    
    elif node.op_name == "flatten":
        (al, ah) = parents[0]
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
        rl = xp.sum(al, axis=axis, keepdims=keepdims)
        rh = xp.sum(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "mean":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        rl = xp.mean(al, axis=axis, keepdims=keepdims)
        rh = xp.mean(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "max":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        # For max, we take max of both bounds
        rl = xp.max(al, axis=axis, keepdims=keepdims)
        rh = xp.max(ah, axis=axis, keepdims=keepdims)
    
    elif node.op_name == "min":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', None)
        keepdims = node.kwargs.get('keepdims', False)
        # For min, we take min of both bounds
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
        rl = xp.concatenate(mins, axis=axis)
        rh = xp.concatenate(maxs, axis=axis)
    
    # Error checking
    if rl is None:
        raise NotImplementedError(f"Operation '{node.op_name}' is not implemented in ops.py")
    
    # Cache result
    node._cache = (rl, rh)
    return rl, rh