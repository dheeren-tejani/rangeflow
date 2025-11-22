"""Visualization Tools"""

import matplotlib.pyplot as plt
import numpy as np
from .core import RangeTensor

def plot_range_evolution(model, input_range, layer_names=None):
    """Plot width/center through layers"""
    widths = []
    centers = []
    names = []
    
    x = input_range
    for i, (name, layer) in enumerate(model.named_modules()):
        if hasattr(layer, 'forward'):
            try:
                x = layer(x)
                if isinstance(x, RangeTensor):
                    widths.append(float(x.width().mean()))
                    centers.append(float(x.avg().mean()))
                    names.append(name if layer_names is None else layer_names[i])
            except:
                continue
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(widths, 'o-')
    ax1.set_title('Uncertainty Width Through Layers')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Average Width')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(centers, 'o-')
    ax2.set_title('Center Value Through Layers')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Average Center')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_uncertainty_map(image, model, epsilon):
    """Visualize spatial uncertainty"""
    img_range = RangeTensor.from_epsilon_ball(image, epsilon)
    output_range = model(img_range)
    
    # Get uncertainty width
    width = output_range.width()
    
    # Reshape if needed
    if len(width.shape) > 2:
        width = width.mean(axis=0)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image[0] if len(image.shape) == 4 else image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(width, cmap='hot')
    plt.title('Uncertainty Map')
    plt.colorbar()
    plt.axis('off')
    
    return plt.gcf()

def plot_certified_accuracy_curve(model, data_loader, epsilon_values):
    """Plot certified accuracy vs epsilon"""
    from .metrics import certified_accuracy
    
    accuracies = []
    for eps in epsilon_values:
        acc = certified_accuracy(model, data_loader, eps)
        accuracies.append(acc)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epsilon_values, accuracies, 'o-', linewidth=2)
    plt.xlabel('Perturbation Size (ε)', fontsize=12)
    plt.ylabel('Certified Accuracy', fontsize=12)
    plt.title('Robustness Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    return plt.gcf()

def visualize_decision_boundary(model, X_range, y, resolution=100):
    """Plot 2D decision boundary with uncertainty"""
    # Only works for 2D input
    if X_range.shape[1] != 2:
        raise ValueError("Can only visualize 2D data")
    
    min_x, max_x = X_range.decay()
    x1_min, x1_max = min_x[:, 0].min(), max_x[:, 0].max()
    x2_min, x2_max = min_x[:, 1].min(), max_x[:, 1].max()
    
    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution)
    )
    
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    grid_range = RangeTensor.from_array(grid)
    
    pred_range = model(grid_range)
    centers = pred_range.avg().reshape(xx1.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, centers, alpha=0.4, cmap='RdYlBu')
    plt.scatter(min_x[:, 0], min_x[:, 1], c=y, cmap='RdYlBu', edgecolor='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary with Uncertainty')
    
    return plt.gcf()