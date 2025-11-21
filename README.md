RangeFlow 🌊Certified Robustness & Uncertainty Quantification for AIRangeFlow is a Python library for Interval Bound Propagation (IBP) and Certified Robustness. It allows you to propagate intervals (ranges of values) through neural networks instead of just single points. This enables you to mathematically prove safety properties, quantify uncertainty rigorously, and train models that are immune to adversarial attacks and noise.🌟 Why RangeFlow?Standard AI models are fragile. A tiny, invisible perturbation to an input image can cause a confident misclassification. Standard uncertainty metrics (like Softmax probability) are often overconfident and wrong.RangeFlow solves this by treating every value as a range: [min, max].For AI Safety Researchers: Verify that your model cannot be tricked by adversarial attacks within a specific bound ($\epsilon$).For Engineers: Quantify the "Worst Case" behavior of your system under sensor noise or quantization.For Scientists: Automatically propagate error bars through complex computations.🚀 InstallationInstall the core library (lightweight, NumPy-based):pip install rangeflow
Hardware AccelerationRangeFlow automatically detects and uses CuPy for GPU acceleration if available. To enable GPU support, install the appropriate CuPy version for your CUDA toolkit (e.g., for CUDA 12.x):pip install cupy-cuda12x
⚡ Quick Start1. The "Hello World" of UncertaintyLet's verify if a simple mathematical operation is stable.import rangeflow as rf
import numpy as np

# Define an uncertain input: 10.0 +/- 0.1
center = np.array([10.0])
uncertainty = 0.1
x = rf.RangeTensor.from_range(center - uncertainty, center + uncertainty)

# Define a computation graph
# f(x) = (x * 2) + 5
y = (x * 2) + 5

# Calculate the rigorous bounds
min_val, max_val = y.decay()

print(f"Input: [{x.decay()[0].item()}, {x.decay()[1].item()}]")
print(f"Output: [{min_val.item()}, {max_val.item()}]")
# Output should be [24.8, 25.2]
2. Robust Neural Network LayerRangeFlow provides drop-in replacements for PyTorch layers that can handle range inputs.import torch
import rangeflow as rf
from rangeflow.layers import RangeLinear

# Create a Robust Linear Layer (3 inputs, 1 output)
layer = RangeLinear(3, 1)

# Create a batch of 5 uncertain inputs (Batch=5, Features=3)
data = torch.randn(5, 3).numpy()
# Wrap data in a range with epsilon=0.1 uncertainty
x_range = rf.RangeTensor.from_array(data)
x_noise = rf.RangeTensor.from_array(np.ones_like(data) * 0.1)
x_robust = x_range + x_noise # Implicit broadcasting for bounds [data, data+0.1] -> effectively [data, data] + [0, 0.1]... 
# Better:
x_robust = rf.RangeTensor.from_range(data - 0.1, data + 0.1)


# Forward Pass (Builds the Lazy Graph)
y_range = layer(x_robust)

# Execute (Compute Bounds)
y_min, y_max = y_range.decay()

print("Output Center:", (y_min + y_max) / 2)
print("Output Uncertainty Width:", y_max - y_min)
🧠 Core Concepts1. Lazy Graph ExecutionRangeFlow uses a Symbolic Computation Graph. When you perform operations like x + y, no math is done immediately. Instead, a node is added to a graph. The actual calculation (which can be expensive for high-dimensional ranges) only happens when you call .decay(). This allows for graph optimizations and efficient memory usage.2. Flowing Conservative Decay (FCD)The core algorithm of RangeFlow. Standard Interval Arithmetic often explodes in error (the "Dependency Problem"). FCD mitigates this by tracking the symbolic history of operations and applying monotonicity rules (e.g., knowing that ReLU is monotonic) to find tighter, more accurate bounds.3. RangeNorm (The Stabilizer)In deep networks, uncertainty intervals can expand exponentially ("The Balloon Effect"), causing training collapse. RangeLayerNorm and RangeBatchNorm counteract this by normalizing not just the signal, but the width of the uncertainty, keeping the model stable during training.🛠️ FeaturesFramework Agnostic: Core engine runs on NumPy or CuPy. No heavy dependencies required for basic usage.PyTorch Integration: rangeflow.patch allows you to "hijack" existing PyTorch models (like Hugging Face Transformers) and make them range-aware instantly.GPU Accelerated: Auto-detects GPUs via CuPy for massive parallelization.Comprehensive Layers:RangeLinear, RangeConv2dRangeRNN, RangeLSTMRangeAttention (for Transformers)RangeLayerNorm, RangeDropout🛡️ Advanced Usage: Hugging Face IntegrationYou can make a pre-trained BERT or GPT model robust without retraining it from scratch.from transformers import GPT2Model
from rangeflow.patch import convert_model_to_rangeflow
import torch

# 1. Load Standard Model
model = GPT2Model.from_pretrained('gpt2')

# 2. Convert to RangeFlow (In-Place)
convert_model_to_rangeflow(model)

# 3. The model now accepts RangeTensors!
# (Note: You must manually handle the embedding layer conversion for full pipeline)
🤝 ContributingRangeFlow is an open-source research project. We welcome contributions!Found a bug? Open an Issue.Want a new layer? Submit a Pull Request.Have a research idea? Start a Discussion.See CONTRIBUTING.md for more details.