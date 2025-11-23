"""
RangeFlow Loss Functions
========================
Robust loss functions for training with uncertainty.
"""

from .backend import get_backend
import numpy as np

xp = get_backend()

# Detect backend type
BACKEND = 'torch' if hasattr(xp, 'nn') else 'numpy'


def robust_cross_entropy(out_range, targets, mode='worst_case'):
    """
    Robust cross-entropy loss for classification.
    
    Considers worst-case scenario within the predicted range,
    making the model learn to be correct even under uncertainty.
    
    Args:
        out_range: RangeTensor of logits (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        mode: 'worst_case' (minimax) or 'average' (expected)
    
    Returns:
        Scalar loss (differentiable)
    
    Example:
        >>> x_range = RangeTensor.from_epsilon_ball(x, 0.1)
        >>> logits_range = model(x_range)
        >>> loss = robust_cross_entropy(logits_range, labels)
        >>> loss.backward()
    """
    l, h = out_range.decay()
    
    if mode == 'worst_case':
        worst = h.clone() if hasattr(h, 'clone') else h.copy()
        
        # Handle device placement for indices
        if hasattr(worst, 'device'):
            rows = xp.arange(len(targets), device=worst.device)
        else:
            rows = xp.arange(len(targets))
            
        worst[rows, targets] = l[rows, targets]
        
        # Handle dimension args (dim vs axis) - FIXED
        if BACKEND == 'torch':
            shift = xp.max(worst, dim=1, keepdim=True)
            if isinstance(shift, tuple): shift = shift[0]  # PyTorch max returns (values, indices)
            
            z = worst - shift
            sum_exp = xp.sum(xp.exp(z), dim=1, keepdim=True)
            log_probs = z - xp.log(sum_exp)
        else:
            shift = xp.max(worst, axis=1, keepdims=True)
            z = worst - shift
            sum_exp = xp.sum(xp.exp(z), axis=1, keepdims=True)
            log_probs = z - xp.log(sum_exp)
        
        return -xp.mean(log_probs[rows, targets])
    
    elif mode == 'average':
        # Average case: use center of range
        center = (l + h) / 2
        
        if BACKEND == 'torch':
            shift = xp.max(center, dim=1, keepdim=True)
            if isinstance(shift, tuple): shift = shift[0]
            z = center - shift
            log_probs = z - xp.log(xp.sum(xp.exp(z), dim=1, keepdim=True))
        else:
            shift = xp.max(center, axis=1, keepdims=True)
            z = center - shift
            log_probs = z - xp.log(xp.sum(xp.exp(z), axis=1, keepdims=True))
        
        rows = xp.arange(len(targets)) if not hasattr(center, 'device') else xp.arange(len(targets), device=center.device)
        return -xp.mean(log_probs[rows, targets])
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def robust_mse(y_range, y_target, mode='worst_case'):
    """
    Robust Mean Squared Error for regression.
    
    Args:
        y_range: RangeTensor of predictions (batch_size, output_dim)
        y_target: Ground truth values (batch_size, output_dim)
        mode: 'worst_case' or 'average'
    
    Returns:
        Scalar MSE loss
    
    Example:
        >>> x_range = RangeTensor.from_epsilon_ball(x, 0.05)
        >>> y_range = model(x_range)
        >>> loss = robust_mse(y_range, y_true)
    """
    min_pred, max_pred = y_range.decay()
    
    if mode == 'worst_case':
        # Worst case error
        loss_min = (min_pred - y_target) ** 2
        loss_max = (max_pred - y_target) ** 2
        
        if BACKEND == 'torch':
            loss = xp.max(xp.stack([loss_min, loss_max]), dim=0)[0]
        else:
            loss = xp.maximum(loss_min, loss_max)
        
        return xp.mean(loss)
    
    elif mode == 'average':
        # Average case (center)
        center = (min_pred + max_pred) / 2
        return xp.mean((center - y_target) ** 2)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def robust_mae(y_range, y_target, mode='worst_case'):
    """
    Robust Mean Absolute Error for regression.
    
    Args:
        y_range: RangeTensor of predictions
        y_target: Ground truth values
        mode: 'worst_case' or 'average'
    
    Returns:
        Scalar MAE loss
    """
    min_pred, max_pred = y_range.decay()
    
    if mode == 'worst_case':
        loss_min = xp.abs(min_pred - y_target)
        loss_max = xp.abs(max_pred - y_target)
        
        if BACKEND == 'torch':
            loss = xp.max(xp.stack([loss_min, loss_max]), dim=0)[0]
        else:
            loss = xp.maximum(loss_min, loss_max)
        
        return xp.mean(loss)
    
    elif mode == 'average':
        center = (min_pred + max_pred) / 2
        return xp.mean(xp.abs(center - y_target))
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def robust_bce(y_range, y_target, mode='worst_case'):
    """
    Robust Binary Cross Entropy for binary classification.
    
    Args:
        y_range: RangeTensor of predictions (0-1 probabilities)
        y_target: Ground truth binary labels (0 or 1)
        mode: 'worst_case' or 'average'
    
    Returns:
        Scalar BCE loss
    
    Example:
        >>> logits_range = model(x_range)
        >>> probs_range = logits_range.sigmoid()
        >>> loss = robust_bce(probs_range, labels)
    """
    min_pred, max_pred = y_range.decay()
    
    # Clip to avoid log(0)
    eps = 1e-7
    if BACKEND == 'torch':
        min_pred = xp.clamp(min_pred, eps, 1 - eps)
        max_pred = xp.clamp(max_pred, eps, 1 - eps)
    else:
        min_pred = xp.clip(min_pred, eps, 1 - eps)
        max_pred = xp.clip(max_pred, eps, 1 - eps)
    
    if mode == 'worst_case':
        # For y=1: worst is min_pred (underestimates probability)
        # For y=0: worst is max_pred (overestimates probability)
        loss_at_min = -y_target * xp.log(min_pred) - (1 - y_target) * xp.log(1 - min_pred)
        loss_at_max = -y_target * xp.log(max_pred) - (1 - y_target) * xp.log(1 - max_pred)
        
        loss = xp.where(y_target == 1, loss_at_min, loss_at_max)
        return xp.mean(loss)
    
    elif mode == 'average':
        center = (min_pred + max_pred) / 2
        return -xp.mean(y_target * xp.log(center) + (1 - y_target) * xp.log(1 - center))
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def robust_hinge(y_range, y_target, margin=1.0):
    """
    Robust Hinge Loss for SVM-style training.
    
    Used for max-margin robust classification.
    
    Args:
        y_range: RangeTensor of predictions (batch_size, num_classes)
        y_target: Ground truth labels (batch_size,)
        margin: Margin parameter (default: 1.0)
    
    Returns:
        Scalar hinge loss
    
    Example:
        >>> logits_range = model(x_range)
        >>> loss = robust_hinge(logits_range, labels, margin=1.0)
    """
    min_logits, max_logits = y_range.decay()
    
    batch_size = len(y_target)
    if hasattr(min_logits, 'device'):
        rows = xp.arange(batch_size, device=min_logits.device)
    else:
        rows = xp.arange(batch_size)
    
    # Get correct class logits (worst case: minimum)
    correct_min = min_logits[rows, y_target]
    
    # Get wrong class logits (worst case: maximum)
    # Create mask for wrong classes
    mask = xp.ones_like(max_logits, dtype=bool)
    mask[rows, y_target] = False
    
    # Reshape to get per-sample max of wrong classes
    if BACKEND == 'torch':
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), dim=1)
        if isinstance(wrong_max, tuple): wrong_max = wrong_max[0]
        loss = xp.clamp(margin + wrong_max - correct_min, min=0)
    else:
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), axis=1)
        loss = xp.maximum(0, margin + wrong_max - correct_min)
    
    return xp.mean(loss)


def width_regularization(range_tensor, lambda_width=0.1):
    """
    Regularization term to penalize excessive uncertainty growth.
    
    Encourages the model to produce tight bounds (low uncertainty).
    Add this to your main loss to control range explosion.
    
    Args:
        range_tensor: RangeTensor output from model
        lambda_width: Regularization strength (default: 0.1)
    
    Returns:
        Scalar regularization loss
    
    Example:
        >>> output_range = model(x_range)
        >>> main_loss = robust_cross_entropy(output_range, labels)
        >>> reg_loss = width_regularization(output_range, lambda_width=0.1)
        >>> total_loss = main_loss + reg_loss
    """
    width = range_tensor.width()
    return lambda_width * xp.mean(width)


def tightness_regularization(range_tensor, lambda_tight=0.01):
    """
    Alternative to width_regularization using relative width.
    
    Penalizes ranges that are wide relative to their center value.
    
    Args:
        range_tensor: RangeTensor output
        lambda_tight: Regularization strength
    
    Returns:
        Scalar regularization loss
    """
    relative_width = range_tensor.relative_width()
    return lambda_tight * xp.mean(relative_width)


def ibp_loss(out_range, targets, kappa=1.0):
    """
    Interval Bound Propagation (IBP) loss.
    
    Standard loss from Gowal et al. 2018 paper on IBP training.
    Combines robust cross-entropy with a margin term.
    
    Args:
        out_range: RangeTensor of logits
        targets: Ground truth labels
        kappa: Margin scaling factor
    
    Returns:
        Scalar IBP loss
    
    Reference:
        Gowal et al. "On the Effectiveness of Interval Bound Propagation 
        for Training Verifiably Robust Models" (2018)
    """
    min_logits, max_logits = out_range.decay()
    
    batch_size = len(targets)
    if hasattr(min_logits, 'device'):
        rows = xp.arange(batch_size, device=min_logits.device)
    else:
        rows = xp.arange(batch_size)
    
    # Standard cross-entropy on center
    center = (min_logits + max_logits) / 2
    
    if BACKEND == 'torch':
        shift = xp.max(center, dim=1, keepdim=True)
        if isinstance(shift, tuple): shift = shift[0]
        z = center - shift
        log_probs = z - xp.log(xp.sum(xp.exp(z), dim=1, keepdim=True))
    else:
        shift = xp.max(center, axis=1, keepdims=True)
        z = center - shift
        log_probs = z - xp.log(xp.sum(xp.exp(z), axis=1, keepdims=True))
    
    ce_loss = -xp.mean(log_probs[rows, targets])
    
    # Robust margin term
    correct_min = min_logits[rows, targets]
    mask = xp.ones_like(max_logits, dtype=bool)
    mask[rows, targets] = False
    
    if BACKEND == 'torch':
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), dim=1)
        if isinstance(wrong_max, tuple): wrong_max = wrong_max[0]
        margin_loss = xp.mean(xp.clamp(kappa - (correct_min - wrong_max), min=0))
    else:
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), axis=1)
        margin_loss = xp.mean(xp.maximum(0, kappa - (correct_min - wrong_max)))
    
    return ce_loss + margin_loss


def crown_loss(out_range, targets, mode='IBP+CROWN'):
    """
    CROWN-style loss combining IBP with tighter bounds.
    
    Args:
        out_range: RangeTensor of logits
        targets: Ground truth labels
        mode: 'IBP+CROWN' or 'CROWN-only'
    
    Returns:
        Scalar loss
    
    Reference:
        Zhang et al. "Towards Stable and Efficient Training of 
        Verifiably Robust Neural Networks" (2019)
    """
    # For now, use IBP loss as approximation
    # Full CROWN would require linear bound propagation
    return ibp_loss(out_range, targets)


def certified_accuracy_loss(out_range, targets, epsilon):
    """
    Direct optimization of certified accuracy.
    
    Experimental loss that directly maximizes the margin
    needed for certification.
    
    Args:
        out_range: RangeTensor of logits
        targets: Ground truth labels
        epsilon: Target perturbation size
    
    Returns:
        Scalar loss (lower = more certified samples)
    """
    min_logits, max_logits = out_range.decay()
    
    batch_size = len(targets)
    if hasattr(min_logits, 'device'):
        rows = xp.arange(batch_size, device=min_logits.device)
    else:
        rows = xp.arange(batch_size)
    
    # Certification requires: min(correct) > max(others)
    correct_min = min_logits[rows, targets]
    
    mask = xp.ones_like(max_logits, dtype=bool)
    mask[rows, targets] = False
    
    if BACKEND == 'torch':
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), dim=1)
        if isinstance(wrong_max, tuple): wrong_max = wrong_max[0]
    else:
        wrong_max = xp.max(xp.where(mask, max_logits, -xp.inf), axis=1)
    
    # Loss = how much we violate certification
    # Negative margin = not certified
    margin = correct_min - wrong_max
    
    # Penalize negative margins heavily
    if BACKEND == 'torch':
        loss = xp.mean(xp.clamp(-margin, min=0) ** 2)
    else:
        loss = xp.mean(xp.maximum(0, -margin) ** 2)
    
    return loss


# Alias for backward compatibility
def robust_loss(y_range, y_target, mode='worst_case'):
    """
    General robust loss (defaults to MSE).
    
    Deprecated: Use robust_mse or robust_cross_entropy directly.
    """
    return robust_mse(y_range, y_target, mode=mode)

def trades_loss(model, x_clean, target, epsilon, beta=6.0):
    """
    TRADES Loss (Zhang et al., 2019) adapted for Certified Robustness.
    
    Optimizes the trade-off between standard accuracy and robustness
    regularization. Essential for high-epsilon training.
    
    Args:
        model: RangeFlow model
        x_clean: Standard input tensor
        target: Labels
        epsilon: Current perturbation radius
        beta: Regularization strength (higher = more robust, less accurate)
    """
    import torch
    
    # 1. Standard Loss (Clean Accuracy)
    logits_clean = model(x_clean)
    if hasattr(logits_clean, 'decay'): logits_clean = logits_clean.avg()
    loss_clean = torch.nn.functional.cross_entropy(logits_clean, target)
    
    # 2. Robust Regularization (The "Anchor")
    # We want the worst-case output to stay close to the clean output
    from .core import RangeTensor
    x_range = RangeTensor.from_epsilon_ball(x_clean, epsilon)
    y_range = model(x_range)
    
    # Get worst-case divergence
    # KL-Divergence between Softmax(Clean) and Softmax(Worst-Case)
    # We approximate worst-case by taking the bound that maximizes distance
    min_logits, max_logits = y_range.decay()
    
    # We want to maximize distance, so we pick bounds that contradict the clean prediction
    probs_clean = torch.softmax(logits_clean, dim=1)
    
    # Construct "Adversarial" Logits from bounds
    # If clean prob is high, use min bound (drag it down)
    # If clean prob is low, use max bound (push it up)
    # This creates the maximum possible divergence within the certified box
    adv_logits = torch.where(logits_clean > 0, min_logits, max_logits)
    
    log_probs_adv = torch.log_softmax(adv_logits, dim=1)
    
    loss_robust = torch.nn.functional.kl_div(
        log_probs_adv, 
        probs_clean, 
        reduction='batchmean', 
        log_target=False
    )
    
    return loss_clean + beta * loss_robust