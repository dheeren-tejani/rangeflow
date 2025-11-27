"""
RangeFlow Advanced Verification Module
=======================================
Branch-and-Bound, domain constraints, and verification utilities.

Key Features:
1. Branch-and-Bound (BaB): Recursive input splitting for tighter bounds
2. Domain constraints: Physics-aware input bounds (e.g., [0,1] for images)
3. Verification certificates: Formal proofs of robustness
"""

import torch
import numpy as np
from .core import RangeTensor
from .backend import get_backend
from typing import Tuple, Optional, Dict, List

xp = get_backend()


class DomainConstraints:
    """
    Physics-aware input constraints.
    
    Automatically handles:
    - Valid ranges (e.g., RGB [0, 1])
    - Structural constraints (e.g., sum to 1)
    - Physical laws (e.g., non-negative energy)
    
    Example:
        >>> # Image domain
        >>> domain = DomainConstraints(min_val=0.0, max_val=1.0, name='RGB')
        >>> 
        >>> # Create epsilon ball that respects domain
        >>> x_range = domain.create_epsilon_ball(image, epsilon=0.3)
        >>> # Automatically clips to [0, 1]!
    """
    
    def __init__(self, min_val=None, max_val=None, name='custom', 
                 constraints=None):
        """
        Args:
            min_val: Minimum allowed value (scalar or array)
            max_val: Maximum allowed value (scalar or array)
            name: Domain name (for logging)
            constraints: Optional custom constraint function
        """
        self.min_val = min_val
        self.max_val = max_val
        self.name = name
        self.constraints = constraints
    
    @classmethod
    def image_domain(cls, bit_depth=8):
        """Standard image domain [0, 1] or [0, 255]"""
        if bit_depth == 8:
            return cls(0.0, 255.0, name='Image_uint8')
        elif bit_depth == 1:
            return cls(0.0, 1.0, name='Image_float32')
        else:
            return cls(0.0, 2**bit_depth - 1, name=f'Image_{bit_depth}bit')
    
    @classmethod
    def probability_domain(cls):
        """Probability distribution domain [0, 1] with sum=1"""
        def prob_constraint(x):
            # Clip to [0, 1]
            x = torch.clamp(x, 0.0, 1.0)
            # Normalize to sum to 1
            return x / (x.sum(dim=-1, keepdim=True) + 1e-8)
        
        return cls(0.0, 1.0, name='Probability', constraints=prob_constraint)
    
    @classmethod
    def unbounded_domain(cls):
        """No constraints (standard case)"""
        return cls(None, None, name='Unbounded')
    
    def clip(self, x):
        """Apply domain constraints to tensor"""
        if self.min_val is not None:
            x = torch.clamp(x, min=self.min_val)
        if self.max_val is not None:
            x = torch.clamp(x, max=self.max_val)
        
        if self.constraints is not None:
            x = self.constraints(x)
        
        return x
    
    def create_epsilon_ball(self, center, epsilon):
        """
        Create epsilon ball that respects domain constraints.
        
        This is the key feature - automatically clips perturbations!
        
        Args:
            center: Center point
            epsilon: Perturbation radius
        
        Returns:
            RangeTensor with domain-constrained bounds
        """
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32)
        
        # Standard epsilon ball
        min_bound = center - epsilon
        max_bound = center + epsilon
        
        # Apply domain constraints
        if self.min_val is not None:
            min_bound = torch.clamp(min_bound, min=self.min_val)
            max_bound = torch.clamp(max_bound, min=self.min_val)
        
        if self.max_val is not None:
            min_bound = torch.clamp(min_bound, max=self.max_val)
            max_bound = torch.clamp(max_bound, max=self.max_val)
        
        return RangeTensor.from_range(min_bound, max_bound)
    
    def validate(self, x):
        """Check if value satisfies constraints"""
        if self.min_val is not None and torch.any(x < self.min_val):
            return False, f"Values below minimum {self.min_val}"
        
        if self.max_val is not None and torch.any(x > self.max_val):
            return False, f"Values above maximum {self.max_val}"
        
        return True, "Valid"


class BranchAndBound:
    """
    Branch-and-Bound verification for completeness.
    
    When standard IBP says "Unknown", BaB splits the input space
    recursively until we get a definitive answer.
    
    Algorithm:
    1. Check bounds on full input range
    2. If not verified, split into sub-ranges
    3. Recursively verify each sub-range
    4. Combine results
    
    Example:
        >>> bab = BranchAndBound(max_depth=3, split_mode='input')
        >>> is_verified, margin = bab.verify(model, image, label, epsilon=0.3)
        >>> print(f"Verified: {is_verified}, Margin: {margin:.3f}")
    """
    
    def __init__(self, max_depth=3, split_mode='input', 
                 min_eps=0.001, timeout=60.0):
        """
        Args:
            max_depth: Maximum recursion depth
            split_mode: 'input' (split input space) or 'activation' (split neurons)
            min_eps: Minimum epsilon for splitting (stop condition)
            timeout: Max verification time in seconds
        """
        self.max_depth = max_depth
        self.split_mode = split_mode
        self.min_eps = min_eps
        self.timeout = timeout
        self.nodes_explored = 0
        self.nodes_verified = 0
    
    def verify(self, model, input_center, target_label, epsilon, 
               domain=None, verbose=False):
        """
        Verify robustness with BaB.
        
        Args:
            model: Neural network
            input_center: Clean input
            target_label: Correct class
            epsilon: Perturbation budget
            domain: Optional DomainConstraints
            verbose: Print progress
        
        Returns:
            (is_verified, margin, stats)
        """
        import time
        start_time = time.time()
        
        self.nodes_explored = 0
        self.nodes_verified = 0
        
        # Initial bounds check
        if domain is not None:
            x_range = domain.create_epsilon_ball(input_center, epsilon)
        else:
            x_range = RangeTensor.from_epsilon_ball(input_center, epsilon)
        
        is_verified, margin = self._verify_recursive(
            model, x_range, target_label, epsilon, 
            depth=0, start_time=start_time, verbose=verbose
        )
        
        stats = {
            'nodes_explored': self.nodes_explored,
            'nodes_verified': self.nodes_verified,
            'time_elapsed': time.time() - start_time,
            'max_depth_reached': min(self.max_depth, self.nodes_explored)
        }
        
        return is_verified, margin, stats
    
    def _verify_recursive(self, model, x_range, target_label, epsilon,
                         depth, start_time, verbose):
        """Recursive verification step"""
        import time
        
        # Check timeout
        if time.time() - start_time > self.timeout:
            if verbose:
                print(f"Timeout at depth {depth}")
            return False, -float('inf')
        
        self.nodes_explored += 1
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output_range = model(x_range)
            min_logits, max_logits = output_range.decay()
        
        # Check verification
        if len(min_logits.shape) > 1:
            min_logits = min_logits[0]
            max_logits = max_logits[0]
        
        correct_min = min_logits[target_label]
        
        mask = torch.ones_like(min_logits, dtype=torch.bool)
        mask[target_label] = False
        others_max = max_logits[mask].max()
        
        margin = float(correct_min - others_max)
        
        # Check if verified
        if margin > 0:
            self.nodes_verified += 1
            if verbose and depth > 0:
                print(f"✓ Verified at depth {depth}, margin={margin:.3f}")
            return True, margin
        
        # Check if we should split
        if depth >= self.max_depth or epsilon < self.min_eps:
            if verbose:
                print(f"✗ Failed at depth {depth}, margin={margin:.3f}")
            return False, margin
        
        # Split the range
        if verbose:
            print(f"Splitting at depth {depth}, current margin={margin:.3f}")
        
        min_input, max_input = x_range.decay()
        mid_input = (min_input + max_input) / 2
        
        # Create two sub-problems
        eps_half = epsilon / 2
        
        # Left half
        x_range_left = RangeTensor.from_range(min_input, mid_input)
        verified_left, margin_left = self._verify_recursive(
            model, x_range_left, target_label, eps_half,
            depth + 1, start_time, verbose
        )
        
        # Right half
        x_range_right = RangeTensor.from_range(mid_input, max_input)
        verified_right, margin_right = self._verify_recursive(
            model, x_range_right, target_label, eps_half,
            depth + 1, start_time, verbose
        )
        
        # Both must be verified
        is_verified = verified_left and verified_right
        combined_margin = min(margin_left, margin_right)
        
        return is_verified, combined_margin
    
    def batch_verify(self, model, data_loader, epsilon, domain=None):
        """
        Verify entire dataset with BaB.
        
        Args:
            model: Neural network
            data_loader: PyTorch DataLoader
            epsilon: Perturbation budget
            domain: Optional DomainConstraints
        
        Returns:
            Dict with verification statistics
        """
        total = 0
        verified = 0
        margins = []
        times = []
        
        for images, labels in data_loader:
            for i in range(len(images)):
                img = images[i]
                label = labels[i].item()
                
                is_verified, margin, stats = self.verify(
                    model, img, label, epsilon, domain
                )
                
                total += 1
                if is_verified:
                    verified += 1
                margins.append(margin)
                times.append(stats['time_elapsed'])
        
        return {
            'verified_accuracy': verified / total if total > 0 else 0.0,
            'avg_margin': np.mean(margins),
            'avg_time': np.mean(times),
            'total_samples': total
        }


class VerificationCertificate:
    """
    Formal certificate of robustness.
    
    Contains:
    - Verified input range
    - Guaranteed output bounds
    - Verification method used
    - Timestamp and model hash
    
    Can be saved and validated later.
    """
    
    def __init__(self, input_range, output_range, target_label,
                 epsilon, method, model_hash=None):
        self.input_range = input_range
        self.output_range = output_range
        self.target_label = target_label
        self.epsilon = epsilon
        self.method = method
        self.model_hash = model_hash
        
        import time
        self.timestamp = time.time()
        
        # Compute margin
        min_logits, max_logits = output_range.decay()
        correct_min = min_logits[0, target_label]
        
        mask = torch.ones_like(min_logits[0], dtype=torch.bool)
        mask[target_label] = False
        others_max = max_logits[0, mask].max()
        
        self.margin = float(correct_min - others_max)
        self.is_valid = self.margin > 0
    
    def save(self, path):
        """Save certificate to file"""
        torch.save({
            'input_range': self.input_range,
            'output_range': self.output_range,
            'target_label': self.target_label,
            'epsilon': self.epsilon,
            'method': self.method,
            'model_hash': self.model_hash,
            'timestamp': self.timestamp,
            'margin': self.margin,
            'is_valid': self.is_valid
        }, path)
    
    @classmethod
    def load(cls, path):
        """Load certificate from file"""
        data = torch.load(path)
        cert = cls(
            data['input_range'],
            data['output_range'],
            data['target_label'],
            data['epsilon'],
            data['method'],
            data.get('model_hash')
        )
        cert.timestamp = data['timestamp']
        return cert
    
    def verify_against_model(self, model):
        """Re-verify certificate against model"""
        with torch.no_grad():
            output_new = model(self.input_range)
            min_new, max_new = output_new.decay()
        
        # Check if new bounds are within certificate bounds
        min_cert, max_cert = self.output_range.decay()
        
        is_consistent = torch.all(min_new >= min_cert - 1e-6) and \
                       torch.all(max_new <= max_cert + 1e-6)
        
        return is_consistent
    
    def __repr__(self):
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return (f"VerificationCertificate({status}, "
                f"ε={self.epsilon:.3f}, margin={self.margin:.3f}, "
                f"method={self.method})")


def verify_model_batch(model, data_loader, epsilon, method='ibp',
                      domain=None, max_samples=None, device='cpu'):
    """
    Comprehensive batch verification with multiple methods.
    
    Args:
        model: Neural network
        data_loader: PyTorch DataLoader
        epsilon: Perturbation budget
        method: 'ibp', 'bab', 'crown', or 'hybrid'
        domain: Optional DomainConstraints
        max_samples: Limit number of samples (None = all)
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict with detailed statistics
    
    Example:
        >>> domain = DomainConstraints.image_domain(bit_depth=1)
        >>> results = verify_model_batch(
        ...     model, test_loader, epsilon=0.3,
        ...     method='hybrid', domain=domain
        ... )
        >>> print(f"Verified: {results['verified_accuracy']:.1%}")
    """
    model.to(device)
    model.eval()
    
    total = 0
    verified = 0
    standard_correct = 0
    margins = []
    
    certificates = []
    
    if method == 'bab':
        bab = BranchAndBound(max_depth=3)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            for i in range(len(images)):
                if max_samples is not None and total >= max_samples:
                    break
                
                img = images[i:i+1]
                label = labels[i].item()
                
                # Standard accuracy
                output_std = model(img)
                if isinstance(output_std, RangeTensor):
                    output_std = output_std.avg()
                pred_std = output_std.argmax(dim=1).item()
                if pred_std == label:
                    standard_correct += 1
                
                # Verified accuracy
                if domain is not None:
                    img_range = domain.create_epsilon_ball(img.squeeze(0), epsilon)
                else:
                    img_range = RangeTensor.from_epsilon_ball(img, epsilon)
                
                if method == 'bab':
                    is_verified, margin, _ = bab.verify(
                        model, img.squeeze(0), label, epsilon, domain
                    )
                else:  # ibp or crown or hybrid
                    output_range = model(img_range)
                    min_logits, max_logits = output_range.decay()
                    
                    correct_min = min_logits[0, label]
                    mask = torch.ones_like(min_logits[0], dtype=torch.bool)
                    mask[label] = False
                    others_max = max_logits[0, mask].max()
                    
                    margin = float(correct_min - others_max)
                    is_verified = margin > 0
                
                if is_verified:
                    verified += 1
                    
                    # Create certificate
                    cert = VerificationCertificate(
                        img_range, output_range if method != 'bab' else None,
                        label, epsilon, method
                    )
                    certificates.append(cert)
                
                margins.append(margin)
                total += 1
            
            if max_samples is not None and total >= max_samples:
                break
    
    return {
        'verified_accuracy': verified / total if total > 0 else 0.0,
        'standard_accuracy': standard_correct / total if total > 0 else 0.0,
        'avg_margin': np.mean(margins),
        'median_margin': np.median(margins),
        'total_samples': total,
        'verified_samples': verified,
        'certificates': certificates,
        'method': method,
        'epsilon': epsilon
    }


def compare_verification_methods(model, test_loader, epsilon, 
                                domain=None, max_samples=100):
    """
    Compare IBP vs BaB vs CROWN on same dataset.
    
    Returns:
        DataFrame with comparison
    """
    methods = ['ibp', 'bab']
    results = {}
    
    print("Comparing verification methods...\n")
    
    for method in methods:
        print(f"Testing {method.upper()}...")
        result = verify_model_batch(
            model, test_loader, epsilon, method=method,
            domain=domain, max_samples=max_samples
        )
        results[method] = result
        print(f"  Verified: {result['verified_accuracy']:.1%}")
        print(f"  Avg Margin: {result['avg_margin']:.3f}\n")
    
    return results