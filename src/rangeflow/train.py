"""
RangeFlow Training Utilities
=============================
High-level training utilities for robust models.
"""

from .core import RangeTensor
from .metrics import certified_accuracy, standard_accuracy
from .backend import get_backend
import numpy as np

xp = get_backend()


class RobustTrainer:
    """
    High-level trainer for robust models.
    
    Handles:
    - Robust training with epsilon scheduling
    - Validation with certification
    - Logging and checkpointing
    - Mixed training (clean + robust)
    
    Args:
        model: Neural network model
        optimizer: PyTorch optimizer
        loss_fn: Robust loss function
        device: 'cpu' or 'cuda'
        log_interval: Steps between logging
    
    Example:
        >>> from rangeflow.train import RobustTrainer
        >>> from rangeflow.loss import robust_cross_entropy
        >>> 
        >>> trainer = RobustTrainer(model, optimizer, robust_cross_entropy)
        >>> trainer.fit(train_loader, val_loader, epochs=10, epsilon_schedule='linear')
    """
    def __init__(self, model, optimizer, loss_fn, device='cpu', log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.log_interval = log_interval
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'standard_acc': [],
            'certified_acc': [],
            'epoch_epsilon': []
        }
        
        self.best_certified_acc = 0.0
        self.current_epoch = 0
    
    def train_epoch(self, train_loader, epsilon, mode='worst_case', 
                   mix_ratio=0.0, width_reg=0.0):
        """
        Train one epoch with robust loss.
        
        Args:
            train_loader: PyTorch DataLoader
            epsilon: Perturbation radius for this epoch
            mode: 'worst_case' or 'average'
            mix_ratio: Fraction of clean (non-robust) training (0-1)
            width_reg: Width regularization coefficient
        
        Returns:
            Average loss for epoch
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required for training")
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed training: some clean, some robust
            if mix_ratio > 0 and np.random.rand() < mix_ratio:
                # Clean training
                output = self.model(data)
                if isinstance(output, RangeTensor):
                    output = output.avg()  # Use center
                
                # Standard loss (convert to torch if needed)
                if hasattr(output, 'get'):
                    output = torch.from_numpy(output.get())
                elif not isinstance(output, torch.Tensor):
                    output = torch.from_numpy(output)
                
                loss = torch.nn.functional.cross_entropy(output, target)
            else:
                # Robust training
                data_range = RangeTensor.from_epsilon_ball(data, epsilon)
                output_range = self.model(data_range)
                loss = self.loss_fn(output_range, target, mode=mode)
                
                # Add width regularization if requested
                if width_reg > 0:
                    from .loss import width_regularization
                    loss = loss + width_regularization(output_range, width_reg)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, ε: {epsilon:.3f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader, epsilon, compute_certified=True):
        """
        Validate model performance.
        
        Args:
            val_loader: Validation DataLoader
            epsilon: Perturbation radius for certification
            compute_certified: Whether to compute certified accuracy (slow!)
        
        Returns:
            dict: Validation metrics
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required")
        
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Robust forward
                data_range = RangeTensor.from_epsilon_ball(data, epsilon)
                output_range = self.model(data_range)
                loss = self.loss_fn(output_range, target, mode='worst_case')
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        
        # Compute accuracies
        std_acc = standard_accuracy(self.model, val_loader, self.device)
        
        if compute_certified:
            cert_acc = certified_accuracy(self.model, val_loader, epsilon, self.device)
        else:
            cert_acc = 0.0
        
        return {
            'val_loss': avg_val_loss,
            'standard_acc': std_acc,
            'certified_acc': cert_acc
        }
    
    def fit(self, train_loader, val_loader, epochs, epsilon_schedule='constant',
            start_epsilon=0.1, end_epsilon=0.3, mix_ratio=0.0, 
            width_reg=0.0, checkpoint_path=None):
        """
        Complete training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            epsilon_schedule: 'constant', 'linear', 'exponential', or custom function
            start_epsilon: Initial epsilon
            end_epsilon: Final epsilon
            mix_ratio: Fraction of clean training
            width_reg: Width regularization
            checkpoint_path: Path to save best model
        
        Returns:
            Training history
        
        Example:
            >>> history = trainer.fit(
            ...     train_loader, val_loader, 
            ...     epochs=50, 
            ...     epsilon_schedule='linear',
            ...     start_epsilon=0.0,
            ...     end_epsilon=0.3
            ... )
        """
        scheduler = EpsilonScheduler(
            schedule_type=epsilon_schedule,
            start_epsilon=start_epsilon,
            end_epsilon=end_epsilon,
            total_epochs=epochs
        )
        
        print(f"Starting robust training for {epochs} epochs")
        print(f"Epsilon schedule: {epsilon_schedule} [{start_epsilon} → {end_epsilon}]")
        print(f"Mix ratio: {mix_ratio}, Width reg: {width_reg}")
        print("-" * 70)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epsilon = scheduler.get_epsilon(epoch)
            
            print(f"\nEpoch {epoch+1}/{epochs} (ε={epsilon:.3f})")
            
            # Train
            train_loss = self.train_epoch(
                train_loader, epsilon, 
                mix_ratio=mix_ratio,
                width_reg=width_reg
            )
            
            # Validate (compute certified accuracy every 5 epochs to save time)
            compute_cert = (epoch % 5 == 0) or (epoch == epochs - 1)
            val_metrics = self.validate(val_loader, epsilon, compute_cert)
            
            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['standard_acc'].append(val_metrics['standard_acc'])
            self.history['certified_acc'].append(val_metrics['certified_acc'])
            self.history['epoch_epsilon'].append(epsilon)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Standard Acc: {val_metrics['standard_acc']:.2%}")
            if compute_cert:
                print(f"  Certified Acc: {val_metrics['certified_acc']:.2%}")
            
            # Checkpoint best model
            if compute_cert and val_metrics['certified_acc'] > self.best_certified_acc:
                self.best_certified_acc = val_metrics['certified_acc']
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                    print(f"  ✓ Saved checkpoint (best certified acc: {self.best_certified_acc:.2%})")
        
        print("\n" + "=" * 70)
        print(f"Training complete!")
        print(f"Best certified accuracy: {self.best_certified_acc:.2%}")
        
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        try:
            import torch
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'history': self.history,
                'epoch': self.current_epoch,
                'best_certified_acc': self.best_certified_acc
            }, path)
        except ImportError:
            print("Warning: PyTorch not available, cannot save checkpoint")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        try:
            import torch
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.history = checkpoint['history']
            self.current_epoch = checkpoint['epoch']
            self.best_certified_acc = checkpoint['best_certified_acc']
            print(f"Loaded checkpoint from epoch {self.current_epoch}")
        except ImportError:
            print("Warning: PyTorch not available, cannot load checkpoint")


class EpsilonScheduler:
    """
    Epsilon scheduling for curriculum learning.
    
    Gradually increases perturbation size during training,
    starting from clean data and progressing to robust training.
    
    Args:
        schedule_type: 'constant', 'linear', 'exponential', 'step', or callable
        start_epsilon: Initial epsilon
        end_epsilon: Final epsilon
        total_epochs: Total training epochs
        warmup_epochs: Epochs of warmup (default: 0)
    
    Example:
        >>> scheduler = EpsilonScheduler('linear', 0.0, 0.3, 100)
        >>> for epoch in range(100):
        ...     eps = scheduler.get_epsilon(epoch)
        ...     # Train with eps
    """
    def __init__(self, schedule_type='linear', start_epsilon=0.0, 
                 end_epsilon=0.3, total_epochs=100, warmup_epochs=0):
        self.schedule_type = schedule_type
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
    
    def get_epsilon(self, epoch):
        """Get epsilon for current epoch"""
        # Warmup phase
        if epoch < self.warmup_epochs:
            return self.start_epsilon
        
        # Adjust epoch for warmup
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        if callable(self.schedule_type):
            return self.schedule_type(adjusted_epoch, adjusted_total)
        
        elif self.schedule_type == 'constant':
            return self.end_epsilon
        
        elif self.schedule_type == 'linear':
            progress = adjusted_epoch / adjusted_total
            return self.start_epsilon + progress * (self.end_epsilon - self.start_epsilon)
        
        elif self.schedule_type == 'exponential':
            progress = adjusted_epoch / adjusted_total
            return self.start_epsilon * ((self.end_epsilon / self.start_epsilon) ** progress)
        
        elif self.schedule_type == 'step':
            # Step every 25% of training
            steps = 4
            step_size = adjusted_total // steps
            step_num = min(adjusted_epoch // step_size, steps - 1)
            step_epsilon = self.start_epsilon + (step_num / (steps - 1)) * (self.end_epsilon - self.start_epsilon)
            return step_epsilon
        
        elif self.schedule_type == 'cosine':
            progress = adjusted_epoch / adjusted_total
            return self.start_epsilon + 0.5 * (self.end_epsilon - self.start_epsilon) * (1 + np.cos(np.pi * (1 - progress)))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def robust_data_augmentation(images, augmentation_ranges):
    """
    Apply data augmentation as ranges.
    
    Args:
        images: Batch of images
        augmentation_ranges: Dict of augmentation parameters
            e.g., {'brightness': 0.2, 'contrast': 0.1}
    
    Returns:
        RangeTensor representing all augmented versions
    
    Example:
        >>> images = load_batch()
        >>> aug_params = {'brightness': 0.2, 'rotation': 5}
        >>> images_range = robust_data_augmentation(images, aug_params)
    """
    # Start with base images
    result = RangeTensor.from_array(images)
    
    # Apply each augmentation as uncertainty
    for aug_type, magnitude in augmentation_ranges.items():
        if aug_type == 'brightness':
            result = RangeTensor.from_range(
                result.decay()[0] - magnitude,
                result.decay()[1] + magnitude
            )
        elif aug_type == 'noise':
            result = RangeTensor.from_epsilon_ball(result.avg(), magnitude)
        # Add more augmentation types as needed
    
    return result


def adversarial_training_step(model, optimizer, data, target, epsilon, alpha=0.01, num_steps=7):
    """
    Single step of PGD adversarial training.
    
    Combines with RangeFlow for certified + empirical robustness.
    
    Args:
        model: Neural network
        optimizer: Optimizer
        data: Clean data
        target: Labels
        epsilon: Attack budget
        alpha: Step size
        num_steps: Number of PGD steps
    
    Returns:
        Loss value
    """
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch required")
    
    # PGD attack to find adversarial example
    data_adv = data.clone().detach()
    data_adv.requires_grad = True
    
    for _ in range(num_steps):
        output = model(data_adv)
        if isinstance(output, RangeTensor):
            output = output.avg()
        
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Update adversarial example
        data_adv = data_adv + alpha * data_adv.grad.sign()
        data_adv = torch.clamp(data_adv, data - epsilon, data + epsilon)
        data_adv = torch.clamp(data_adv, 0, 1)
        data_adv = data_adv.detach()
        data_adv.requires_grad = True
    
    # Train on adversarial example with ranges
    optimizer.zero_grad()
    data_range = RangeTensor.from_epsilon_ball(data_adv, epsilon / 2)
    output_range = model(data_range)
    
    from .loss import robust_cross_entropy
    loss = robust_cross_entropy(output_range, target)
    loss.backward()
    optimizer.step()
    
    return loss.item()