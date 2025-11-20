from .backend import get_backend
import numpy as np

def infer_shape(op, shapes, **kwargs):
    """Calculates output shape without running the math."""
    if not shapes: return ()
    s0 = shapes[0]
    
    if op in ["add", "sub", "mul", "div"]: return s0
    
    if op == "matmul": 
        # (..., N, M) @ (..., M, K) -> (..., N, K)
        return s0[:-1] + (shapes[1][-1],)
        
    if op == "transpose":
        l = list(s0)
        d0, d1 = kwargs['dim0'], kwargs['dim1']
        l[d0], l[d1] = l[d1], l[d0]
        return tuple(l)
        
    if op == "reshape": return kwargs['shape']
    
    if op == "conv2d":
        # Input: (N, C_in, H, W)
        # Weight: (C_out, C_in, K, K)
        # Output: (N, C_out, H_out, W_out)
        # For simplicity in Lazy Graph, we assume standard formulas or rely on execution
        # A proper library would calculate H_out/W_out based on padding/stride
        # Placeholder: Return input shape with channel swap (Good enough for lazy init)
        return (s0[0], shapes[1][0], s0[2], s0[3]) 

    return s0

def evaluate_bounds(node):
    """
    The FCD Execution Engine.
    Traverses the graph and computes [Min, Max].
    """
    if node._cache is not None: return node._cache
    
    xp = get_backend()
    
    # --- LEAF NODES ---
    if node.op_name == "LEAF": return node.value, node.value
    if node.op_name == "LEAF_RANGE": return node.value
    
    # --- RECURSION ---
    parents = [evaluate_bounds(p) for p in node.parents]
    
    # Initialize results
    rl, rh = None, None
    
    # --- ARITHMETIC ---
    if node.op_name == "add":
        (al, ah), (bl, bh) = parents
        rl, rh = al + bl, ah + bh
        
    elif node.op_name == "sub":
        (al, ah), (bl, bh) = parents
        rl, rh = al - bh, ah - bl
        
    elif node.op_name == "mul":
        (al, ah), (bl, bh) = parents
        p1, p2, p3, p4 = al*bl, al*bh, ah*bl, ah*bh
        rl = xp.minimum(xp.minimum(p1, p2), xp.minimum(p3, p4))
        rh = xp.maximum(xp.maximum(p1, p2), xp.maximum(p3, p4))
        
    elif node.op_name == "matmul":
        (al, ah), (bl, bh) = parents
        # Monotonic shortcut for weights
        w_pos = xp.maximum(bh, 0)
        w_neg = xp.minimum(bl, 0)
        rl = (al @ w_pos) + (ah @ w_neg)
        rh = (ah @ w_pos) + (al @ w_neg)

    # --- CONVOLUTION (The Missing Piece) ---
    elif node.op_name == "conv2d":
        (min_x, max_x) = parents[0]
        (min_w, max_w) = parents[1]
        
        # Handle Bias
        if len(parents) > 2:
            (min_b, max_b) = parents[2]
        else:
            min_b, max_b = 0, 0
            
        stride = node.kwargs.get('stride', 1)
        padding = node.kwargs.get('padding', 0)

        # Implementation Note:
        # Writing a Convolution kernel in raw NumPy is slow/complex.
        # We leverage PyTorch's optimized F.conv2d if available.
        try:
            import torch
            import torch.nn.functional as F
            
            # Helper to convert any backend (numpy/cupy) to torch cpu tensors
            def to_torch(arr):
                if hasattr(arr, 'get'): arr = arr.get() # CuPy -> NumPy
                return torch.from_numpy(arr).float() if hasattr(arr, 'shape') else torch.tensor(arr).float()

            t_min_x = to_torch(min_x)
            t_max_x = to_torch(max_x)
            t_min_w = to_torch(min_w)
            t_max_w = to_torch(max_w)
            
            # Monotonic Conv Logic
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
                
            # Convert back to active backend
            rl = xp.asarray(res_min.numpy())
            rh = xp.asarray(res_max.numpy())

        except ImportError:
             raise RuntimeError("RangeFlow requires PyTorch installed to run Conv2d operations.")

    # --- ACTIVATIONS ---
    elif node.op_name == "relu":
        (al, ah) = parents[0]
        rl, rh = xp.maximum(al, 0), xp.maximum(ah, 0)
        
    elif node.op_name == "tanh":
        (al, ah) = parents[0]
        rl, rh = xp.tanh(al), xp.tanh(ah)
        
    elif node.op_name == "exp":
        (al, ah) = parents[0]
        rl, rh = xp.exp(al), xp.exp(ah)
        
    elif node.op_name == "softmax":
        (al, ah) = parents[0]
        axis = node.kwargs.get('axis', -1)
        ex_l, ex_h = xp.exp(al), xp.exp(ah)
        sum_h = xp.sum(ex_h, axis=axis, keepdims=True)
        sum_l = xp.sum(ex_l, axis=axis, keepdims=True)
        rl = ex_l / sum_h
        rh = ex_h / sum_l

    # --- SHAPE ---
    elif node.op_name == "reshape":
        (al, ah) = parents[0]
        s = node.kwargs['shape']
        rl, rh = xp.reshape(al, s), xp.reshape(ah, s)
        
    elif node.op_name == "transpose":
        (al, ah) = parents[0]
        d0, d1 = node.kwargs['dim0'], node.kwargs['dim1']
        rl, rh = xp.swapaxes(al, d0, d1), xp.swapaxes(ah, d0, d1)
    
    # Error checking
    if rl is None:
        raise NotImplementedError(f"Operation '{node.op_name}' is not implemented in ops.py")

    node._cache = (rl, rh)
    return rl, rh