"""
RangeFlow Advanced Training Utilities
======================================
State-aware schedulers, monitoring hooks, and advanced training loops.

New Features:
1. Resumable epsilon scheduling
2. Automatic range monitoring (LiveProbe system)
3. TRADES loss for robust-accurate trade-off
4. Mixed precision training support
"""

import torch
import torch.nn as nn
import numpy as np
from .train import EpsilonScheduler, RobustTrainer
from .core import RangeTensor
from typing import Dict, List, Optional, Callable
import time


class StatefulEpsilonScheduler(EpsilonScheduler):
    """
    Epsilon scheduler with state persistence.
    
    Critical for resumable training - remembers where it left off!
    
    Example:
        >>> scheduler = StatefulEpsilonScheduler('linear', 0.0, 0.3, 100)
        >>> 
        >>> # Train for 20 epochs
        >>> for epoch in range(20):
        >>>     eps = scheduler.step()
        >>>     train_epoch(model, eps)
        >>> 
        >>> # Save checkpoint
        >>> torch.save({
        >>>     'model': model.state_dict(),
        >>>     'scheduler': scheduler.state_dict()
        >>> }, 'checkpoint.pt')
        >>> 
        >>> # Resume later
        >>> checkpoint = torch.load('checkpoint.pt')
        >>> scheduler.load_state_dict(checkpoint['scheduler'])
        >>> # Continue from epoch 20!
    """
    
    def __init__(self, schedule_type='linear', start_epsilon=0.0,
                 end_epsilon=0.3, total_epochs=100, warmup_epochs=0,
                 current_epoch=0):
        super().__init__(schedule_type, start_epsilon, end_epsilon,
                        total_epochs, warmup_epochs)
        self.current_epoch = current_epoch
        self.history = []  # Track epsilon values
    
    def step(self):
        """
        Step scheduler and return current epsilon.
        
        Automatically increments epoch counter.
        
        Returns:
            Current epsilon value
        """
        eps = self.get_epsilon(self.current_epoch)
        self.history.append(eps)
        self.current_epoch += 1
        return eps
    
    def state_dict(self):
        """Save scheduler state"""
        return {
            'schedule_type': self.schedule_type,
            'start_epsilon': self.start_epsilon,
            'end_epsilon': self.end_epsilon,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'current_epoch': self.current_epoch,
            'history': self.history
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.schedule_type = state_dict['schedule_type']
        self.start_epsilon = state_dict['start_epsilon']
        self.end_epsilon = state_dict['end_epsilon']
        self.total_epochs = state_dict['total_epochs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.current_epoch = state_dict['current_epoch']
        self.history = state_dict.get('history', [])
    
    def reset(self):
        """Reset to beginning"""
        self.current_epoch = 0
        self.history = []
    
    def get_current_epsilon(self):
        """Get current epsilon without stepping"""
        return self.get_epsilon(self.current_epoch)
    
    def plot_schedule(self):
        """Visualize epsilon schedule"""
        import matplotlib.pyplot as plt
        
        future_eps = [self.get_epsilon(e) for e in range(self.total_epochs)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if self.history:
            ax.plot(range(len(self.history)), self.history, 
                   'b-', linewidth=2, label='Completed')
        
        ax.plot(range(self.current_epoch, self.total_epochs), 
               future_eps[self.current_epoch:],
               'b--', linewidth=2, alpha=0.5, label='Remaining')
        
        ax.axvline(self.current_epoch, color='r', linestyle='--', 
                  label=f'Current Epoch ({self.current_epoch})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Epsilon')
        ax.set_title(f'Epsilon Schedule ({self.schedule_type})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig


class RangeMonitorHook:
    """
    Automatic range width monitoring via PyTorch hooks.
    
    No need to modify your model! Just register this hook.
    
    Features:
    - Track width through all layers
    - Detect explosions automatically
    - Log statistics
    - Alert when threshold exceeded
    
    Example:
        >>> from rangeflow.advanced_train import monitor_ranges
        >>> 
        >>> model = MyRobustModel()
        >>> hook = monitor_ranges(model, explosion_threshold=50.0)
        >>> 
        >>> # Train normally
        >>> for data, target in train_loader:
        >>>     output = model(data)
        >>>     # Hook automatically tracks ranges!
    """
    
    def __init__(self, module, name, explosion_threshold=50.0, 
                 log_interval=100):
        self.module = module
        self.name = name
        self.explosion_threshold = explosion_threshold
        self.log_interval = log_interval
        
        self.step = 0
        self.width_history = []
        self.center_history = []
        self.explosion_count = 0
        
        # Register forward hook
        self.handle = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        """Hook function called on forward pass"""
        if isinstance(output, RangeTensor):
            width = output.width().mean().item()
            center = output.avg().mean().item()
            
            self.width_history.append(width)
            self.center_history.append(center)
            
            # Check for explosion
            if width > self.explosion_threshold:
                self.explosion_count += 1
                if self.step % self.log_interval == 0:
                    print(f"⚠️ RANGE EXPLOSION in {self.name}: "
                          f"width={width:.2f} > {self.explosion_threshold}")
            
            self.step += 1
    
    def remove(self):
        """Remove hook"""
        self.handle.remove()
    
    def get_stats(self):
        """Get summary statistics"""
        if not self.width_history:
            return None
        
        return {
            'name': self.name,
            'avg_width': np.mean(self.width_history),
            'max_width': np.max(self.width_history),
            'min_width': np.min(self.width_history),
            'avg_center': np.mean(self.center_history),
            'explosion_count': self.explosion_count,
            'steps': self.step
        }
    
    def plot(self):
        """Plot width evolution"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(self.width_history, alpha=0.7)
        ax1.axhline(self.explosion_threshold, color='r', 
                   linestyle='--', label='Explosion Threshold')
        ax1.set_ylabel('Width')
        ax1.set_title(f'Range Width Evolution - {self.name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.center_history, alpha=0.7, color='orange')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Center')
        ax2.set_title(f'Center Value Evolution - {self.name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def monitor_ranges(model, explosion_threshold=50.0, log_interval=100):
    """
    Register monitoring hooks on all RangeModule layers.
    
    Args:
        model: Neural network model
        explosion_threshold: Width above which to alert
        log_interval: Steps between logging
    
    Returns:
        List of hook objects
    
    Example:
        >>> model = RobustCNN()
        >>> hooks = monitor_ranges(model)
        >>> 
        >>> # Train...
        >>> 
        >>> # Check statistics
        >>> for hook in hooks:
        >>>     print(hook.get_stats())
        >>> 
        >>> # Remove hooks
        >>> for hook in hooks:
        >>>     hook.remove()
    """
    from .layers import RangeModule
    
    hooks = []
    
    for name, module in model.named_modules():
        if isinstance(module, RangeModule):
            hook = RangeMonitorHook(
                module, name, explosion_threshold, log_interval
            )
            hooks.append(hook)
            print(f"✓ Monitoring: {name}")
    
    print(f"\nRegistered {len(hooks)} monitoring hooks")
    return hooks


class TRADESTrainer(RobustTrainer):
    """
    Trainer with TRADES loss (Trade-off between Robust and Accurate).
    
    TRADES is critical for achieving both:
    - High standard accuracy (clean data)
    - High certified accuracy (robust data)
    
    Original Paper: Zhang et al. 2019
    
    Key Idea: Minimize distance between clean and adversarial predictions
    instead of maximizing worst-case loss.
    
    Example:
        >>> trainer = TRADESTrainer(model, optimizer, beta=6.0)
        >>> trainer.fit(train_loader, val_loader, epochs=50)
    """
    
    def __init__(self, model, optimizer, beta=6.0, device='cpu', 
                 log_interval=100):
        """
        Args:
            model: Neural network
            optimizer: PyTorch optimizer
            beta: Trade-off parameter (higher = more robust, less accurate)
            device: 'cpu' or 'cuda'
            log_interval: Logging frequency
        """
        # Use dummy loss_fn (will be overridden)
        super().__init__(model, optimizer, None, device, log_interval)
        self.beta = beta
    
    def trades_loss(self, x_clean, y_true, epsilon):
        """
        Compute TRADES loss.
        
        Loss = CE(f(x_clean), y) + β * KL(f(x_clean) || f(x_adv))
        
        Args:
            x_clean: Clean input
            y_true: True labels
            epsilon: Perturbation budget
        
        Returns:
            (total_loss, clean_loss, robust_loss)
        """
        # 1. Standard Loss (Clean Accuracy)
        logits_clean = self.model(x_clean)
        if isinstance(logits_clean, RangeTensor):
            logits_clean = logits_clean.avg()
        
        loss_clean = torch.nn.functional.cross_entropy(logits_clean, y_true)
        
        # 2. Robust Regularization (The "Anchor")
        # We want worst-case output to stay close to clean output
        x_range = RangeTensor.from_epsilon_ball(x_clean, epsilon)
        y_range = self.model(x_range)
        
        # Get worst-case divergence
        min_logits, max_logits = y_range.decay()
        
        # Construct "Adversarial" Logits from bounds
        # If clean prob is high, use min bound (drag it down)
        # If clean prob is low, use max bound (push it up)
        probs_clean = torch.softmax(logits_clean, dim=1)
        adv_logits = torch.where(logits_clean > 0, min_logits, max_logits)
        
        log_probs_adv = torch.log_softmax(adv_logits, dim=1)
        
        # KL divergence
        loss_robust = torch.nn.functional.kl_div(
            log_probs_adv,
            probs_clean,
            reduction='batchmean',
            log_target=False
        )
        
        total_loss = loss_clean + self.beta * loss_robust
        
        return total_loss, loss_clean, loss_robust
    
    def train_epoch(self, train_loader, epsilon, **kwargs):
        """Train one epoch with TRADES loss"""
        self.model.train()
        total_loss = 0.0
        total_clean = 0.0
        total_robust = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # TRADES loss
            loss, loss_clean, loss_robust = self.trades_loss(
                data, target, epsilon
            )
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_clean += loss_clean.item()
            total_robust += loss_robust.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.log_interval == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f} '
                      f'(Clean: {loss_clean.item():.4f}, '
                      f'Robust: {loss_robust.item():.4f})')
        
        return total_loss / num_batches


class CheckpointManager:
    """
    Manages checkpoints with epsilon scheduler state.
    
    Ensures resumable training with correct curriculum.
    
    Example:
        >>> manager = CheckpointManager(checkpoint_dir='./checkpoints')
        >>> 
        >>> # Save
        >>> manager.save(model, optimizer, scheduler, epoch=42, 
        >>>              metrics={'acc': 0.95})
        >>> 
        >>> # Load
        >>> state = manager.load_best()
        >>> model.load_state_dict(state['model'])
        >>> scheduler.load_state_dict(state['scheduler'])
    """
    
    def __init__(self, checkpoint_dir='./checkpoints', keep_best=3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Number of best checkpoints to keep
        """
        import os
        self.checkpoint_dir = checkpoint_dir
        self.keep_best = keep_best
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_metrics = []
    
    def save(self, model, optimizer, scheduler, epoch, metrics=None,
             name='checkpoint'):
        """
        Save checkpoint with all training state.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: EpsilonScheduler state
            epoch: Current epoch
            metrics: Dict of metrics (e.g., {'acc': 0.95})
            name: Checkpoint name prefix
        """
        import os
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {}
        }
        
        # Save scheduler state if available
        if hasattr(scheduler, 'state_dict'):
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Filename with metrics
        metric_str = ''
        if metrics:
            metric_str = '_'.join([f"{k}={v:.4f}" for k, v in metrics.items()])
        
        filename = f"{name}_epoch{epoch}_{metric_str}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        
        torch.save(checkpoint, path)
        print(f"✓ Saved checkpoint: {path}")
        
        # Track best checkpoints
        if metrics:
            self.best_metrics.append((metrics.get('certified_acc', 0), path))
            self.best_metrics.sort(reverse=True)
            
            # Remove old checkpoints
            if len(self.best_metrics) > self.keep_best:
                _, old_path = self.best_metrics.pop()
                if os.path.exists(old_path):
                    os.remove(old_path)
                    print(f"Removed old checkpoint: {old_path}")
    
    def load(self, path):
        """Load checkpoint from path"""
        checkpoint = torch.load(path)
        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def load_best(self):
        """Load best checkpoint by certified accuracy"""
        if not self.best_metrics:
            raise ValueError("No checkpoints saved yet")
        
        _, best_path = self.best_metrics[0]
        return self.load(best_path)
    
    def load_latest(self):
        """Load most recent checkpoint"""
        import os
        import glob
        
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, '*.pt'))
        if not checkpoints:
            raise ValueError("No checkpoints found")
        
        latest = max(checkpoints, key=os.path.getctime)
        return self.load(latest)


def train_with_curriculum(model, train_loader, val_loader, 
                         epochs=100, start_eps=0.0, end_eps=0.3,
                         method='trades', beta=6.0, checkpoint_dir='./checkpoints'):
    """
    Complete curriculum training with all bells and whistles.
    
    Includes:
    - Stateful epsilon scheduling
    - Range monitoring
    - TRADES loss
    - Checkpoint management
    - Automatic resumption
    
    Args:
        model: Neural network
        train_loader: Training data
        val_loader: Validation data
        epochs: Total epochs
        start_eps: Starting epsilon
        end_eps: Final epsilon
        method: 'trades' or 'standard'
        beta: TRADES beta parameter
        checkpoint_dir: Where to save checkpoints
    
    Returns:
        Trained model and training history
    
    Example:
        >>> model = RobustCNN()
        >>> model, history = train_with_curriculum(
        ...     model, train_loader, val_loader,
        ...     epochs=100, start_eps=0.0, end_eps=0.3,
        ...     method='trades', beta=6.0
        ... )
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StatefulEpsilonScheduler('linear', start_eps, end_eps, epochs)
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    # Try to resume
    try:
        checkpoint = checkpoint_mgr.load_latest()
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    except:
        print("Starting fresh training")
    
    # Monitor ranges
    hooks = monitor_ranges(model)
    
    # Choose trainer
    if method == 'trades':
        trainer = TRADESTrainer(model, optimizer, beta, device)
    else:
        from .loss import robust_cross_entropy
        trainer = RobustTrainer(model, optimizer, robust_cross_entropy, device)
    
    # Training loop
    print("=" * 70)
    print("CURRICULUM TRAINING")
    print("=" * 70)
    
    history = {'train_loss': [], 'val_acc': [], 'cert_acc': [], 'epsilon': []}
    
    for epoch in range(scheduler.current_epoch, epochs):
        eps = scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{epochs} (ε={eps:.3f})")
        
        # Train
        if method == 'trades':
            train_loss = trainer.train_epoch(train_loader, eps)
        else:
            train_loss = trainer.train_epoch(train_loader, eps)
        
        # Validate
        val_metrics = trainer.validate(val_loader, eps, 
                                      compute_certified=(epoch % 5 == 0))
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_metrics['standard_acc'])
        history['cert_acc'].append(val_metrics['certified_acc'])
        history['epsilon'].append(eps)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Acc: {val_metrics['standard_acc']:.2%}")
        if val_metrics['certified_acc'] > 0:
            print(f"Cert Acc: {val_metrics['certified_acc']:.2%}")
        
        # Save checkpoint
        checkpoint_mgr.save(
            model, optimizer, scheduler, epoch,
            metrics={
                'val_acc': val_metrics['standard_acc'],
                'certified_acc': val_metrics['certified_acc']
            }
        )
    
    # Cleanup hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, history