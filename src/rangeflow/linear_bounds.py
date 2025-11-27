"""
RangeFlow Linear Bound Propagation (CROWN/DeepPoly)
====================================================
Symbolic interval propagation that tracks linear relationships.

This solves the "decorrelation problem" where pure IBP loses
the connection between variables (e.g., "if pixel A↑ then pixel B↓").

Mathematical Foundation:
-----------------------
Instead of [min, max], we propagate:
    y_lower ≥ W_L @ x + b_L
    y_upper ≤ W_U @ x + b_U

This keeps bounds MUCH tighter through deep networks.
"""

import torch
import numpy as np
from .core import RangeTensor
from .backend import get_backend

xp = get_backend()


class LinearRangeTensor:
    """
    Hybrid Symbolic-Concrete Interval Representation.
    
    Instead of just [min, max], we track:
    - Lower bound: linear function of input
    - Upper bound: linear function of input
    
    This enables MUCH tighter bounds through compositions.
    
    Attributes:
        lower_w: Lower bound weight matrix
        lower_b: Lower bound bias
        upper_w: Upper bound weight matrix  
        upper_b: Upper bound bias
        concrete_lower: Concrete lower bound (optional)
        concrete_upper: Concrete upper bound (optional)
    """
    
    def __init__(self, lower_w, lower_b, upper_w, upper_b,
                 concrete_lower=None, concrete_upper=None):
        self.lower_w = lower_w
        self.lower_b = lower_b
        self.upper_w = upper_w
        self.upper_b = upper_b
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        self.shape = lower_b.shape
    
    @classmethod
    def from_range_tensor(cls, range_tensor, input_bounds):
        """
        Convert standard RangeTensor to LinearRangeTensor.
        
        Args:
            range_tensor: Standard RangeTensor
            input_bounds: (min_input, max_input) tuple
        
        Returns:
            LinearRangeTensor with identity linear forms
        """
        min_val, max_val = range_tensor.decay()
        
        # Identity transformation (y = 1*x + 0)
        batch_size, dim = min_val.shape
        lower_w = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
        upper_w = lower_w.clone()
        lower_b = torch.zeros(batch_size, dim)
        upper_b = torch.zeros(batch_size, dim)
        
        return cls(lower_w, lower_b, upper_w, upper_b, min_val, max_val)
    
    @classmethod
    def from_epsilon_ball(cls, center, epsilon, create_symbolic=True):
        """
        Create LinearRangeTensor from epsilon ball.
        
        Args:
            center: Center point
            epsilon: Perturbation radius
            create_symbolic: If True, create symbolic bounds
        
        Returns:
            LinearRangeTensor with stored input bounds
        """
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        
        if len(center.shape) == 1:
            center = center.unsqueeze(0)
        
        batch_size, dim = center.shape
        
        if create_symbolic:
            # Symbolic: y = x (identity)
            lower_w = torch.eye(dim).unsqueeze(0).repeat(batch_size, 1, 1)
            upper_w = lower_w.clone()
            lower_b = -epsilon * torch.ones(batch_size, dim)
            upper_b = epsilon * torch.ones(batch_size, dim)
            
            concrete_lower = center - epsilon
            concrete_upper = center + epsilon
        else:
            # Just concrete bounds
            lower_w = torch.zeros(batch_size, dim, dim)
            upper_w = torch.zeros(batch_size, dim, dim)
            lower_b = center - epsilon
            upper_b = center + epsilon
            concrete_lower = lower_b
            concrete_upper = upper_b
        
        obj = cls(lower_w, lower_b, upper_w, upper_b, concrete_lower, concrete_upper)
        
        # Store input bounds for ReLU propagation
        obj._input_bounds = (concrete_lower, concrete_upper)
        
        return obj
    
    def concretize(self, input_bounds):
        """
        Evaluate concrete bounds given input bounds.
        
        Args:
            input_bounds: (min_input, max_input) tuple
        
        Returns:
            (concrete_lower, concrete_upper) tuple
        """
        x_min, x_max = input_bounds
        
        # For lower bound: y_lower = W_L @ x + b_L
        # We want minimum, so use x_min where W_L > 0, x_max where W_L < 0
        w_l_pos = torch.clamp(self.lower_w, min=0)
        w_l_neg = torch.clamp(self.lower_w, max=0)
        
        if len(x_min.shape) == 1:
            x_min = x_min.unsqueeze(0)
            x_max = x_max.unsqueeze(0)
        
        concrete_lower = torch.bmm(w_l_pos, x_min.unsqueeze(-1)).squeeze(-1) + \
                        torch.bmm(w_l_neg, x_max.unsqueeze(-1)).squeeze(-1) + \
                        self.lower_b
        
        # For upper bound: y_upper = W_U @ x + b_U  
        # We want maximum, so use x_max where W_U > 0, x_min where W_U < 0
        w_u_pos = torch.clamp(self.upper_w, min=0)
        w_u_neg = torch.clamp(self.upper_w, max=0)
        
        concrete_upper = torch.bmm(w_u_pos, x_max.unsqueeze(-1)).squeeze(-1) + \
                        torch.bmm(w_u_neg, x_min.unsqueeze(-1)).squeeze(-1) + \
                        self.upper_b
        
        self.concrete_lower = concrete_lower
        self.concrete_upper = concrete_upper
        
        return concrete_lower, concrete_upper
    
    def decay(self):
        """Get concrete bounds (for compatibility with RangeTensor)"""
        if self.concrete_lower is None or self.concrete_upper is None:
            raise ValueError("Must call concretize() first or provide concrete bounds")
        return self.concrete_lower, self.concrete_upper
    
    def width(self):
        """Get uncertainty width"""
        lower, upper = self.decay()
        return upper - lower
    
    def avg(self):
        """Get center point"""
        lower, upper = self.decay()
        return (lower + upper) / 2


class LinearBoundPropagation:
    """
    CROWN/DeepPoly-style linear bound propagation.
    
    Key Operations:
    1. Linear layers: Simple matrix multiplication
    2. ReLU: Adaptive relaxation (the hard part)
    3. Composition: Chain linear transformations
    """
    
    @staticmethod
    def propagate_linear(x_linear, weight, bias=None):
        """
        Propagate through linear layer: y = Wx + b
        
        This is EASY - just compose the linear functions!
        
        Args:
            x_linear: Input LinearRangeTensor
            weight: Layer weight matrix (out_dim, in_dim)
            bias: Layer bias (out_dim,)
        
        Returns:
            Output LinearRangeTensor with preserved input_bounds
        """
        batch_size = x_linear.lower_w.shape[0]
        
        # Lower bound: y_L = W @ (W_L_x @ x + b_L_x) + b
        #                  = (W @ W_L_x) @ x + (W @ b_L_x + b)
        lower_w_new = torch.bmm(
            weight.unsqueeze(0).repeat(batch_size, 1, 1),
            x_linear.lower_w
        )
        lower_b_new = torch.matmul(
            weight.unsqueeze(0).repeat(batch_size, 1, 1),
            x_linear.lower_b.unsqueeze(-1)
        ).squeeze(-1)
        
        if bias is not None:
            lower_b_new += bias.unsqueeze(0)
        
        # Upper bound: same logic
        upper_w_new = torch.bmm(
            weight.unsqueeze(0).repeat(batch_size, 1, 1),
            x_linear.upper_w
        )
        upper_b_new = torch.matmul(
            weight.unsqueeze(0).repeat(batch_size, 1, 1),
            x_linear.upper_b.unsqueeze(-1)
        ).squeeze(-1)
        
        if bias is not None:
            upper_b_new += bias.unsqueeze(0)
        
        result = LinearRangeTensor(
            lower_w_new, lower_b_new,
            upper_w_new, upper_b_new
        )
        
        # Preserve input bounds for downstream ReLU layers
        if hasattr(x_linear, '_input_bounds'):
            result._input_bounds = x_linear._input_bounds
        
        return result
    
    @staticmethod
    def propagate_relu(x_linear, input_bounds):
        """
        Propagate through ReLU with adaptive relaxation.
        
        This is the CROWN innovation - instead of just clipping,
        we draw optimal lines that bound the ReLU.
        
        Cases for each neuron:
        1. Always positive (l ≥ 0): ReLU(x) = x (identity)
        2. Always negative (u ≤ 0): ReLU(x) = 0 (zero)
        3. Crossing zero (l < 0 < u): Adaptive relaxation
        
        Args:
            x_linear: Input LinearRangeTensor
            input_bounds: (min_input, max_input) for concretization
        
        Returns:
            Output LinearRangeTensor with relaxed ReLU and preserved input_bounds
        """
        # First concretize to get bounds on pre-ReLU activations
        l, u = x_linear.concretize(input_bounds)
        
        batch_size, dim = l.shape
        
        # Initialize new bounds (will be element-wise different)
        lower_w_new = torch.zeros_like(x_linear.lower_w)
        lower_b_new = torch.zeros_like(x_linear.lower_b)
        upper_w_new = torch.zeros_like(x_linear.upper_w)
        upper_b_new = torch.zeros_like(x_linear.upper_b)
        
        for i in range(dim):
            l_i = l[:, i]
            u_i = u[:, i]
            
            # Case 1: Always positive (l ≥ 0)
            always_pos = l_i >= 0
            if always_pos.any():
                lower_w_new[always_pos, i, :] = x_linear.lower_w[always_pos, i, :]
                lower_b_new[always_pos, i] = x_linear.lower_b[always_pos, i]
                upper_w_new[always_pos, i, :] = x_linear.upper_w[always_pos, i, :]
                upper_b_new[always_pos, i] = x_linear.upper_b[always_pos, i]
            
            # Case 2: Always negative (u ≤ 0)
            always_neg = u_i <= 0
            if always_neg.any():
                # Output is zero (w=0, b=0 already initialized)
                pass
            
            # Case 3: Crossing zero (l < 0 < u) - THE HARD CASE
            crossing = (~always_pos) & (~always_neg)
            if crossing.any():
                # CROWN adaptive relaxation
                # Lower bound: Use line from (l, 0) to (u, u)
                # Equation: y = u/(u-l) * (x - l)
                
                lambda_lower = torch.zeros(batch_size)
                lambda_lower[crossing] = u_i[crossing] / (u_i[crossing] - l_i[crossing] + 1e-8)
                
                lower_w_new[crossing, i, :] = lambda_lower[crossing].unsqueeze(-1) * \
                                              x_linear.lower_w[crossing, i, :]
                lower_b_new[crossing, i] = lambda_lower[crossing] * \
                                           (x_linear.lower_b[crossing, i] - l_i[crossing])
                
                # Upper bound: Use min of two lines:
                # Line 1: y = x (identity)
                # Line 2: y = u/(u-l) * (x - l) 
                # We conservatively use the identity line for upper bound
                upper_w_new[crossing, i, :] = x_linear.upper_w[crossing, i, :]
                upper_b_new[crossing, i] = x_linear.upper_b[crossing, i]
        
        result = LinearRangeTensor(
            lower_w_new, lower_b_new,
            upper_w_new, upper_b_new
        )
        
        # Preserve input bounds
        if hasattr(x_linear, '_input_bounds'):
            result._input_bounds = x_linear._input_bounds
        
        return result
    
    @staticmethod
    def propagate_conv2d(x_linear, weight, bias=None, stride=1, padding=0):
        """
        Propagate through Conv2D using Im2Col trick.
        
        Conv2D is just matrix multiplication on unfolded patches,
        so we can reuse propagate_linear!
        """
        # TODO: Implement using torch.nn.functional.unfold
        # For now, fall back to concrete bounds
        raise NotImplementedError("Conv2D linear propagation coming soon")


def hybrid_verification(model, input_center, epsilon, use_linear=True, 
                       branching_depth=0):
    """
    Hybrid verification: Linear bounds + optional branching.
    
    This combines CROWN (tight bounds) with Branch-and-Bound (completeness).
    
    Args:
        model: Neural network
        input_center: Clean input
        epsilon: Perturbation budget
        use_linear: If True, use linear bounds (tighter)
        branching_depth: Max depth for branch-and-bound (0 = no branching)
    
    Returns:
        (is_verified, margin, method_used)
    
    Example:
        >>> is_safe, margin, method = hybrid_verification(
        ...     model, image, epsilon=0.3, 
        ...     use_linear=True, branching_depth=2
        ... )
    """
    device = next(model.parameters()).device
    input_center = input_center.to(device)
    
    if use_linear:
        # Create linear bounds
        x_linear = LinearRangeTensor.from_epsilon_ball(input_center, epsilon)
        
        # Propagate through network
        # (This requires model to support LinearRangeTensor - see next section)
        try:
            output_linear = model(x_linear, return_linear=True)
            min_logits, max_logits = output_linear.concretize(
                (input_center - epsilon, input_center + epsilon)
            )
            method = "CROWN"
        except:
            # Fall back to standard IBP
            from .core import RangeTensor
            x_range = RangeTensor.from_epsilon_ball(input_center, epsilon)
            output_range = model(x_range)
            min_logits, max_logits = output_range.decay()
            method = "IBP"
    else:
        # Standard IBP
        from .core import RangeTensor
        x_range = RangeTensor.from_epsilon_ball(input_center, epsilon)
        output_range = model(x_range)
        min_logits, max_logits = output_range.decay()
        method = "IBP"
    
    # Check verification
    pred_class = torch.argmax(min_logits + max_logits)
    correct_min = min_logits[0, pred_class]
    
    mask = torch.ones_like(min_logits[0], dtype=torch.bool)
    mask[pred_class] = False
    others_max = max_logits[0, mask].max()
    
    margin = float(correct_min - others_max)
    is_verified = margin > 0
    
    # If not verified and branching allowed, try splitting
    if not is_verified and branching_depth > 0:
        # Recursive branching (simplified)
        mid = input_center
        
        # Split into two sub-problems
        is_verified_1, margin_1, _ = hybrid_verification(
            model, mid - epsilon/2, epsilon/2, use_linear, branching_depth-1
        )
        is_verified_2, margin_2, _ = hybrid_verification(
            model, mid + epsilon/2, epsilon/2, use_linear, branching_depth-1
        )
        
        is_verified = is_verified_1 and is_verified_2
        margin = min(margin_1, margin_2)
        method = f"{method}+BaB"
    
    return is_verified, margin, method


# Integration with existing RangeFlow layers
def enable_linear_bounds(model):
    """
    Modify model to support LinearRangeTensor propagation.
    
    This adds a 'return_linear' mode to all layers and stores input_bounds
    in the model state for ReLU layers.
    
    Args:
        model: RangeFlow model
    
    Example:
        >>> model = RobustCNN()
        >>> enable_linear_bounds(model)
        >>> x_linear = LinearRangeTensor.from_epsilon_ball(x, 0.3)
        >>> output = model(x_linear)
    """
    from .layers import RangeLinear, RangeReLU, RangeModule
    
    # Store input bounds in model
    if not hasattr(model, '_linear_bounds_config'):
        model._linear_bounds_config = {'enabled': True, 'input_bounds': None}
    
    # Monkey-patch RangeLinear
    original_linear_forward = RangeLinear.forward
    
    def linear_forward_hybrid(self, x):
        if isinstance(x, LinearRangeTensor):
            return LinearBoundPropagation.propagate_linear(
                x, self.weight, self.bias
            )
        return original_linear_forward(self, x)
    
    RangeLinear.forward = linear_forward_hybrid
    
    # Monkey-patch RangeReLU to auto-compute input bounds
    original_relu_forward = RangeReLU.forward
    
    def relu_forward_hybrid(self, x):
        if isinstance(x, LinearRangeTensor):
            # Auto-compute input bounds from the LinearRangeTensor
            # Get the input bounds from model config or compute from x
            if hasattr(x, '_input_bounds'):
                input_bounds = x._input_bounds
            else:
                # Use the LinearRangeTensor's own bounds as estimate
                # This assumes x was created from an epsilon ball
                if x.concrete_lower is not None and x.concrete_upper is not None:
                    input_bounds = (x.concrete_lower, x.concrete_upper)
                else:
                    # Concretize first
                    # We need the original input bounds - store them during creation
                    raise ValueError(
                        "LinearRangeTensor missing input_bounds. "
                        "Create with from_epsilon_ball and ensure bounds are stored."
                    )
            return LinearBoundPropagation.propagate_relu(x, input_bounds)
        return original_relu_forward(self, x)
    
    RangeReLU.forward = relu_forward_hybrid
    
    print("✓ Linear bounds enabled for model")