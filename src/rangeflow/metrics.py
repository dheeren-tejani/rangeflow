"""
RangeFlow Evaluation Metrics
=============================
Metrics for evaluating robustness and uncertainty quantification.
"""

from .core import RangeTensor
from .backend import get_backend
import numpy as np

xp = get_backend()


def certified_accuracy(model, data_loader, epsilon, device='cpu'):
    """
    Compute certified accuracy: % of samples provably robust.
    
    For each sample, verifies that the prediction cannot change
    within an epsilon-ball around the input.
    
    Args:
        model: Neural network model (RangeFlow-compatible)
        data_loader: PyTorch DataLoader
        epsilon: Perturbation radius (L-infinity norm)
        device: 'cpu' or 'cuda'
    
    Returns:
        float: Certified accuracy (0 to 1)
    
    Example:
        >>> model = load_robust_model()
        >>> acc = certified_accuracy(model, test_loader, epsilon=0.3)
        >>> print(f"Certified robust at ε=0.3: {acc:.1%}")
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required for certified_accuracy")
    
    model.eval()
    certified_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Create epsilon balls
            images_range = RangeTensor.from_epsilon_ball(images, epsilon)
            
            # Forward pass
            logits_range = model(images_range)
            min_logits, max_logits = logits_range.decay()
            
            # Convert back to torch for indexing
            if hasattr(min_logits, 'get'):  # CuPy
                min_logits = torch.from_numpy(min_logits.get())
                max_logits = torch.from_numpy(max_logits.get())
            else:  # NumPy
                min_logits = torch.from_numpy(min_logits)
                max_logits = torch.from_numpy(max_logits)
            
            # Check certification for each sample
            for i in range(len(labels)):
                target = labels[i].item()
                
                # Min logit of correct class
                correct_min = min_logits[i, target].item()
                
                # Max logit of all other classes
                mask = torch.ones(min_logits.shape[1], dtype=torch.bool)
                mask[target] = False
                others_max = max_logits[i, mask].max().item()
                
                # Certified if correct_min > others_max
                if correct_min > others_max:
                    certified_correct += 1
                
                total += 1
    
    return certified_correct / total if total > 0 else 0.0


def average_certified_radius(model, data_loader, max_epsilon=1.0, steps=20, device='cpu'):
    """
    Compute Average Certified Radius (ACR).
    
    For each sample, finds the maximum epsilon for which it's certified,
    then averages across all samples.
    
    Args:
        model: Neural network model
        data_loader: PyTorch DataLoader
        max_epsilon: Maximum epsilon to search
        steps: Number of epsilon values to try
        device: 'cpu' or 'cuda'
    
    Returns:
        float: Average certified radius
    
    Example:
        >>> acr = average_certified_radius(model, test_loader, max_epsilon=1.0)
        >>> print(f"Average Certified Radius: {acr:.3f}")
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required")
    
    model.eval()
    epsilon_values = np.linspace(0, max_epsilon, steps)
    total_radius = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            for i in range(len(images)):
                img = images[i:i+1]
                label = labels[i:i+1]
                
                # Binary search for maximum epsilon
                certified_radius = 0.0
                
                for eps in epsilon_values:
                    img_range = RangeTensor.from_epsilon_ball(img, eps)
                    logits_range = model(img_range)
                    min_logits, max_logits = logits_range.decay()
                    
                    # Check certification
                    target = label.item()
                    if hasattr(min_logits, 'get'):
                        min_logits = torch.from_numpy(min_logits.get())
                        max_logits = torch.from_numpy(max_logits.get())
                    else:
                        min_logits = torch.from_numpy(min_logits)
                        max_logits = torch.from_numpy(max_logits)
                    
                    correct_min = min_logits[0, target].item()
                    mask = torch.ones(min_logits.shape[1], dtype=torch.bool)
                    mask[target] = False
                    others_max = max_logits[0, mask].max().item()
                    
                    if correct_min > others_max:
                        certified_radius = eps
                    else:
                        break  # No point checking larger epsilon
                
                total_radius += certified_radius
                total_samples += 1
    
    return total_radius / total_samples if total_samples > 0 else 0.0


def robustness_curve(model, data_loader, epsilon_values, device='cpu'):
    """
    Compute certified accuracy for multiple epsilon values.
    
    Args:
        model: Neural network model
        data_loader: PyTorch DataLoader
        epsilon_values: List of epsilon values to test
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: {epsilon: certified_accuracy}
    
    Example:
        >>> epsilons = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        >>> curve = robustness_curve(model, test_loader, epsilons)
        >>> for eps, acc in curve.items():
        ...     print(f"ε={eps:.1f}: {acc:.1%}")
    """
    curve = {}
    for eps in epsilon_values:
        acc = certified_accuracy(model, data_loader, eps, device)
        curve[eps] = acc
    return curve


def standard_accuracy(model, data_loader, device='cpu'):
    """
    Compute standard (non-robust) accuracy.
    
    Args:
        model: Neural network model
        data_loader: PyTorch DataLoader
        device: 'cpu' or 'cuda'
    
    Returns:
        float: Standard accuracy (0 to 1)
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Standard forward pass (no ranges)
            outputs = model(images)
            
            # If output is RangeTensor, get center
            if isinstance(outputs, RangeTensor):
                outputs = outputs.avg()
            
            # Convert to torch if needed
            if hasattr(outputs, 'get'):
                outputs = torch.from_numpy(outputs.get())
            elif not isinstance(outputs, torch.Tensor):
                outputs = torch.from_numpy(outputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return correct / total if total > 0 else 0.0


def uncertainty_calibration(model, data_loader, num_bins=10, device='cpu'):
    """
    Check if predicted uncertainty matches actual error.
    
    A well-calibrated model should have:
    - Wide ranges when predictions are uncertain/wrong
    - Tight ranges when predictions are confident/correct
    
    Args:
        model: Neural network model
        data_loader: PyTorch DataLoader
        num_bins: Number of bins for calibration plot
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Calibration statistics
    
    Example:
        >>> calib = uncertainty_calibration(model, val_loader)
        >>> print(f"Calibration error: {calib['ece']:.3f}")
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required")
    
    model.eval()
    
    uncertainties = []
    errors = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward with ranges
            img_range = RangeTensor.from_epsilon_ball(images, 0.0)  # Epistemic uncertainty
            logits_range = model(img_range)
            
            # Get uncertainty (width) and prediction
            width = logits_range.width()
            center = logits_range.avg()
            
            if hasattr(width, 'get'):
                width = torch.from_numpy(width.get())
                center = torch.from_numpy(center.get())
            else:
                width = torch.from_numpy(width)
                center = torch.from_numpy(center)
            
            # Prediction correctness
            _, predicted = center.max(1)
            correct = predicted.eq(labels).float()
            
            # Store per-sample uncertainty and error
            avg_width = width.mean(dim=1)  # Average width across classes
            uncertainties.extend(avg_width.cpu().numpy())
            errors.extend((1 - correct).cpu().numpy())
    
    uncertainties = np.array(uncertainties)
    errors = np.array(errors)
    
    # Bin by uncertainty
    bins = np.linspace(uncertainties.min(), uncertainties.max(), num_bins + 1)
    bin_errors = []
    bin_uncertainties = []
    
    for i in range(num_bins):
        mask = (uncertainties >= bins[i]) & (uncertainties < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(errors[mask].mean())
            bin_uncertainties.append(uncertainties[mask].mean())
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(np.array(bin_uncertainties) - np.array(bin_errors)))
    
    return {
        'ece': ece,
        'bin_uncertainties': bin_uncertainties,
        'bin_errors': bin_errors,
        'bins': bins
    }


def worst_case_error(y_range, y_true):
    """
    Compute worst-case error within the predicted range.
    
    Args:
        y_range: RangeTensor of predictions
        y_true: Ground truth values
    
    Returns:
        float: Maximum possible error
    
    Example:
        >>> y_range = model(x_range)
        >>> error = worst_case_error(y_range, y_true)
        >>> print(f"Worst-case MSE: {error:.3f}")
    """
    min_pred, max_pred = y_range.decay()
    
    # Error at both extremes
    error_at_min = xp.abs(min_pred - y_true)
    error_at_max = xp.abs(max_pred - y_true)
    
    # Worst case
    worst = xp.maximum(error_at_min, error_at_max)
    
    return float(xp.mean(worst))


def certified_robustness_score(model, data_loader, epsilon, device='cpu'):
    """
    Comprehensive robustness score (0-100).
    
    Combines:
    - Certified accuracy (main metric)
    - Standard accuracy (baseline)
    - Average certified radius
    
    Args:
        model: Neural network model
        data_loader: PyTorch DataLoader
        epsilon: Target perturbation size
        device: 'cpu' or 'cuda'
    
    Returns:
        dict: Comprehensive metrics
    
    Example:
        >>> scores = certified_robustness_score(model, test_loader, 0.3)
        >>> print(f"Overall Score: {scores['overall']:.1f}/100")
    """
    cert_acc = certified_accuracy(model, data_loader, epsilon, device)
    std_acc = standard_accuracy(model, data_loader, device)
    acr = average_certified_radius(model, data_loader, max_epsilon=epsilon*2, device=device)
    
    # Overall score (weighted combination)
    overall = (cert_acc * 50) + (std_acc * 30) + (min(acr / epsilon, 1.0) * 20)
    
    return {
        'certified_accuracy': cert_acc,
        'standard_accuracy': std_acc,
        'average_certified_radius': acr,
        'overall': overall * 100  # 0-100 scale
    }


def range_statistics(range_tensor):
    """
    Compute comprehensive statistics of a RangeTensor.
    
    Args:
        range_tensor: RangeTensor to analyze
    
    Returns:
        dict: Statistics
    
    Example:
        >>> output_range = model(x_range)
        >>> stats = range_statistics(output_range)
        >>> print(f"Mean width: {stats['mean_width']:.3f}")
    """
    min_val, max_val = range_tensor.decay()
    width = range_tensor.width()
    center = range_tensor.center()
    relative_width = range_tensor.relative_width()
    
    return {
        'mean_width': float(xp.mean(width)),
        'max_width': float(xp.max(width)),
        'min_width': float(xp.min(width)),
        'mean_relative_width': float(xp.mean(relative_width)),
        'mean_center': float(xp.mean(center)),
        'std_center': float(xp.std(center)),
    }