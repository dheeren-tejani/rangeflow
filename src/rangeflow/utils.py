"""Helper Utilities"""

from .core import RangeTensor
from .backend import get_backend
import numpy as np
xp = get_backend()

def range_to_tensor(range_tensor):
    """Convert RangeTensor to PyTorch tensor (center)"""
    try:
        import torch
        center = range_tensor.avg()
        if hasattr(center, 'get'):
            center = center.get()
        return torch.from_numpy(center)
    except ImportError:
        return range_tensor.avg()

def tensor_to_range(tensor, epsilon=0.0):
    """Convert tensor to RangeTensor"""
    if hasattr(tensor, 'cpu'):
        tensor = tensor.cpu().numpy()
    return RangeTensor.from_epsilon_ball(tensor, epsilon)

def save_model(model, path):
    """Save RangeFlow model"""
    try:
        import torch
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")
    except ImportError:
        print("PyTorch not available")

def load_model(model, path):
    """Load RangeFlow model"""
    try:
        import torch
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    except ImportError:
        print("PyTorch not available")

def print_range_stats(range_tensor, name="Range"):
    """Pretty print range statistics"""
    min_val, max_val = range_tensor.decay()
    print(f"\n{name} Statistics:")
    print(f"  Shape: {range_tensor.shape}")
    print(f"  Min: {xp.min(min_val):.4f}")
    print(f"  Max: {xp.max(max_val):.4f}")
    print(f"  Center (mean): {xp.mean(range_tensor.avg()):.4f}")
    print(f"  Width (mean): {xp.mean(range_tensor.width()):.4f}")
    print(f"  Relative Width: {xp.mean(range_tensor.relative_width()):.2%}")

class RangeLogger:
    """Log range evolution during training"""
    def __init__(self):
        self.logs = {'widths': [], 'centers': [], 'rel_widths': []}
    
    def log(self, range_tensor, step):
        self.logs['widths'].append((step, range_tensor.width().mean()))
        self.logs['centers'].append((step, range_tensor.avg().mean()))
        self.logs['rel_widths'].append((step, range_tensor.relative_width().mean()))
    
    def plot(self):
        """Plot logged values"""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for i, (key, ax) in enumerate(zip(['widths', 'centers', 'rel_widths'], axes)):
            steps, values = zip(*self.logs[key])
            ax.plot(steps, values)
            ax.set_title(key.capitalize())
            ax.set_xlabel('Step')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig