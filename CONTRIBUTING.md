# Contributing to RangeFlow 🤝

Thank you for your interest in contributing to RangeFlow! This document will guide you through the process.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [How to Contribute](#how-to-contribute)
5. [Code Style](#code-style)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Adding New Features](#adding-new-features)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of interval arithmetic (see README.md)
- Familiarity with NumPy and (optionally) PyTorch

### Quick Start

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/dheeren-tejani/rangeflow.git
cd rangeflow

# Add upstream remote
git remote add upstream https://github.com/dheeren-tejani/rangeflow.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/
```

---

## 🛠️ Development Setup

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker
- `sphinx` - Documentation generator

### Optional: GPU Support

```bash
pip install cupy-cuda12x  # For CUDA 12.x
# or
pip install cupy-cuda11x  # For CUDA 11.x
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install
```

Now code will be automatically formatted before each commit!

---

## 📁 Project Structure

```
rangeflow/
├── src/rangeflow/          # Main package
│   ├── __init__.py         # Package exports
│   ├── core.py             # RangeTensor class
│   ├── ops.py              # Interval operations
│   ├── layers.py           # Neural network layers
│   ├── loss.py             # Robust loss functions
│   ├── backend.py          # NumPy/CuPy backend
│   ├── vision.py           # Computer vision utilities
│   ├── nlp.py              # NLP utilities
│   ├── rl.py               # Reinforcement learning
│   ├── analysis.py         # Analysis tools
│   ├── metrics.py          # Evaluation metrics
│   ├── train.py            # Training utilities
│   ├── timeseries.py       # Time series support
│   ├── transforms.py       # Data augmentation
│   ├── utils.py            # Helper functions
│   ├── visualize.py        # Visualization tools
│   └── patch.py            # PyTorch model conversion
├── tests/                  # Test suite
│   ├── test_core.py        # Core functionality tests
│   ├── test_ops.py         # Operations tests
│   ├── test_layers.py      # Layer tests
│   └── ...
├── examples/               # Example scripts
│   ├── mnist_robust.py     # MNIST example
│   ├── linear_regression.py
│   └── ...
├── docs/                   # Documentation
│   ├── THEORY.md           # Mathematical foundations
│   ├── QUICKSTART.md       # Quick start guide
│   └── API.md              # API reference
├── README.md               # Main documentation
├── CONTRIBUTING.md         # This file
├── LICENSE                 # MIT License
└── pyproject.toml          # Project configuration
```

### Key Files to Know

| File | Purpose |
|------|---------|
| `core.py` | Defines `RangeTensor` and `Symbol` classes |
| `ops.py` | Implements interval arithmetic operations |
| `layers.py` | Neural network layer implementations |
| `backend.py` | NumPy/CuPy abstraction layer |

---

## 🎯 How to Contribute

### Types of Contributions

1. **Bug Reports** 🐛
   - Found a bug? Open an issue!
   - Include: Python version, OS, minimal reproduction code

2. **Feature Requests** 💡
   - Have an idea? Open an issue with the `enhancement` label
   - Explain: What problem does it solve? How should it work?

3. **Code Contributions** 💻
   - New layers, operations, or utilities
   - Bug fixes
   - Performance improvements
   - Documentation improvements

4. **Documentation** 📖
   - Tutorials, examples, or API docs
   - Fixing typos or clarifying explanations

5. **Testing** 🧪
   - Adding test cases
   - Improving test coverage

---

## ✨ Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good ✅
class RangeLinear(RangeModule):
    """
    Fully connected layer with range propagation.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias term
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = initialize_weights(in_features, out_features)
        self.bias = initialize_bias(out_features) if bias else None
    
    def forward(self, x: RangeTensor) -> RangeTensor:
        """Forward pass with range propagation."""
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias
        return out

# Bad ❌
class rangelinear:  # Use PascalCase for classes
    def __init__(self,i,o,b=True):  # Add spaces, type hints
        self.w=init(i,o)  # Use descriptive names
        self.b=init(o) if b else None
    def forward(self,x):return x@self.w.T+self.b  # No single-line methods
```

### Key Conventions

1. **Naming:**
   - Classes: `PascalCase` (e.g., `RangeTensor`)
   - Functions/methods: `snake_case` (e.g., `evaluate_bounds`)
   - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_ITERATIONS`)
   - Private: prefix with `_` (e.g., `_internal_func`)

2. **Type Hints:**
   ```python
   def process_range(x: RangeTensor, epsilon: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
       """Always include type hints for public APIs."""
       pass
   ```

3. **Docstrings:**
   ```python
   def my_function(arg1: int, arg2: str = "default") -> bool:
       """
       One-line summary of what the function does.
       
       Longer description if needed. Explain the purpose,
       behavior, and any important details.
       
       Args:
           arg1: Description of first argument
           arg2: Description of second argument
       
       Returns:
           Description of return value
       
       Raises:
           ValueError: When and why this is raised
       
       Examples:
           >>> my_function(42, "test")
           True
       """
       pass
   ```

4. **Formatting:**
   - Line length: 100 characters max
   - Use `black` for auto-formatting: `black src/rangeflow/`
   - Use `flake8` for linting: `flake8 src/rangeflow/`

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rangeflow --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run specific test
pytest tests/test_core.py::test_range_addition

# Run with verbose output
pytest -v
```

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_my_feature.py
import pytest
import numpy as np
from rangeflow import RangeTensor

class TestMyFeature:
    """Test suite for my new feature."""
    
    def test_basic_functionality(self):
        """Test basic case."""
        x = RangeTensor.from_range(1.0, 2.0)
        result = my_feature(x)
        
        min_val, max_val = result.decay()
        assert min_val == pytest.approx(expected_min, abs=1e-5)
        assert max_val == pytest.approx(expected_max, abs=1e-5)
    
    def test_edge_case_zero(self):
        """Test edge case with zero."""
        x = RangeTensor.from_range(-1.0, 1.0)
        result = my_feature(x)
        # Assertions...
    
    def test_invalid_input(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="invalid input"):
            my_feature(invalid_input)
    
    @pytest.mark.parametrize("min_val,max_val,expected", [
        (1.0, 2.0, 3.0),
        (5.0, 10.0, 15.0),
        (-2.0, 2.0, 0.0),
    ])
    def test_parametrized(self, min_val, max_val, expected):
        """Test multiple cases."""
        x = RangeTensor.from_range(min_val, max_val)
        assert my_feature(x).avg() == pytest.approx(expected)
```

### Test Coverage Goals

- **Core modules** (`core.py`, `ops.py`): >95% coverage
- **Layers** (`layers.py`): >90% coverage
- **Applications** (`vision.py`, `nlp.py`, etc.): >80% coverage

---

## 📝 Documentation

### Docstring Standards

Every public function, class, and method needs a docstring:

```python
def robust_loss(y_range: RangeTensor, y_target: np.ndarray, 
                mode: str = 'worst_case') -> float:
    """
    Compute robust loss over uncertainty interval.
    
    This loss function considers the worst-case error within
    the predicted range, making the model robust to input uncertainty.
    
    Args:
        y_range: Predicted output as RangeTensor with uncertainty
        y_target: Ground truth labels (numpy array)
        mode: Loss computation mode. Options:
            - 'worst_case': Minimize maximum error (minimax)
            - 'average': Minimize expected error
    
    Returns:
        Scalar loss value (differentiable via PyTorch autograd)
    
    Raises:
        ValueError: If mode is not recognized
    
    Examples:
        >>> x_range = RangeTensor.from_epsilon_ball(x, 0.1)
        >>> y_range = model(x_range)
        >>> loss = robust_loss(y_range, y_true, mode='worst_case')
        >>> loss.backward()
    
    Note:
        This function is differentiable and works with PyTorch's
        autograd system.
    
    See Also:
        - robust_cross_entropy: For classification tasks
        - robust_mse: For regression tasks
    """
    pass
```

### Building Documentation

```bash
cd docs/
sphinx-build -b html source/ build/
# Open build/index.html in browser
```

---

## 🔄 Pull Request Process

### Before Submitting

1. **Create an issue first** (for significant changes)
2. **Fork the repository**
3. **Create a feature branch:**
   ```bash
   git checkout -b feature/my-awesome-feature
   ```
4. **Make your changes**
5. **Add tests** for new functionality
6. **Update documentation** if needed
7. **Run tests and linting:**
   ```bash
   pytest
   black src/rangeflow/
   flake8 src/rangeflow/
   mypy src/rangeflow/
   ```
8. **Commit with clear messages:**
   ```bash
   git commit -m "Add robust LSTM layer with uncertainty propagation"
   ```

### Submitting the PR

1. **Push to your fork:**
   ```bash
   git push origin feature/my-awesome-feature
   ```

2. **Open a Pull Request** on GitHub

3. **Fill out the PR template:**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Motivation
   Why is this change needed?
   
   ## Changes Made
   - Added X
   - Fixed Y
   - Updated Z
   
   ## Testing
   - [ ] Added tests
   - [ ] All tests pass
   - [ ] Coverage maintained/improved
   
   ## Checklist
   - [ ] Code follows style guide
   - [ ] Documentation updated
   - [ ] No breaking changes (or documented)
   ```

4. **Respond to review feedback**

5. **Celebrate! 🎉** Your contribution will be merged!

---

## 🆕 Adding New Features

### Adding a New Layer

1. **Implement in `layers.py`:**

```python
class RangeMyLayer(RangeModule):
    """Your layer description."""
    
    def __init__(self, param1: int, param2: float = 0.5):
        super().__init__()
        # Initialize parameters
        self.param1 = param1
        self.param2 = param2
    
    def forward(self, x: RangeTensor) -> RangeTensor:
        """Forward pass."""
        # Implement using existing ops
        return _op("my_operation", x, param=self.param2)
```

2. **Implement the operation in `ops.py`:**

```python
def infer_shape(op, shapes, **kwargs):
    # Add shape inference logic
    if op == "my_operation":
        return shapes[0]  # Or compute new shape
    # ...

def evaluate_bounds(node):
    # Add evaluation logic
    elif node.op_name == "my_operation":
        (min_x, max_x) = parents[0]
        param = node.kwargs['param']
        
        # Interval arithmetic logic
        min_result = compute_min(min_x, param)
        max_result = compute_max(max_x, param)
        
        return min_result, max_result
    # ...
```

3. **Add tests:**

```python
# tests/test_layers.py
def test_my_layer():
    layer = RangeMyLayer(param1=10)
    x = RangeTensor.from_range(1.0, 2.0)
    
    result = layer(x)
    min_val, max_val = result.decay()
    
    assert min_val < max_val
    # More assertions...
```

4. **Update `__init__.py`:**

```python
from .layers import RangeMyLayer

__all__ = [..., 'RangeMyLayer']
```

5. **Add documentation and examples**

### Adding a New Operation

Similar process but focus on `ops.py`:

```python
# In infer_shape()
if op == "my_new_op":
    return compute_output_shape(shapes, **kwargs)

# In evaluate_bounds()
elif node.op_name == "my_new_op":
    (al, ah) = parents[0]
    # Interval arithmetic for your operation
    rl = ...
    rh = ...
    return rl, rh
```

---

## 🎨 Best Practices

### DO ✅

- Write clear, self-documenting code
- Add comprehensive tests
- Update documentation
- Use type hints
- Handle edge cases
- Consider performance
- Follow existing patterns

### DON'T ❌

- Break backward compatibility without discussion
- Add dependencies without justification
- Copy-paste code (refactor instead)
- Skip tests ("it's just a small change")
- Ignore linter warnings
- Leave commented-out code
- Use magic numbers (use constants)

---

## 🐛 Reporting Bugs

Good bug reports include:

1. **Title:** Clear, specific description
2. **Environment:**
   - OS (Windows/Linux/Mac)
   - Python version
   - RangeFlow version
   - GPU/CPU
3. **Steps to reproduce:**
   ```python
   import rangeflow as rf
   x = rf.RangeTensor.from_range(1, 2)
   # ... minimal code to reproduce bug
   ```
4. **Expected behavior:** What should happen
5. **Actual behavior:** What actually happens
6. **Error messages:** Full traceback if applicable

---

## 💡 Feature Requests

Good feature requests include:

1. **Use case:** What problem does it solve?
2. **Proposed solution:** How should it work?
3. **Alternatives considered:** Other approaches?
4. **Willing to implement:** Can you contribute code?

---

## 📞 Getting Help

- **Questions:** Use [GitHub Discussions](https://github.com/dheeren-tejani/rangeflow/discussions)
- **Bugs:** Use [GitHub Issues](https://github.com/dheeren-tejani/rangeflow/issues)
- **Chat:** Join our Discord server (link in README)
- **Email:** contact@rangeflow.ai

---

## 🎓 Learning Resources

### Interval Arithmetic
- Moore, R. E. (1966). *Interval Analysis*
- [Wikipedia: Interval Arithmetic](https://en.wikipedia.org/wiki/Interval_arithmetic)

### Robust AI
- Gowal et al. (2018). *On the Effectiveness of Interval Bound Propagation*
- Wong & Kolter (2018). *Provable Defenses against Adversarial Examples*

### RangeFlow Specific
- [Theory Document](docs/THEORY.md)
- [Quick Start Guide](docs/QUICKSTART.md)
- [API Reference](docs/API.md)

---

## 🏆 Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation
- Conference/paper acknowledgments (if applicable)

---

## 📜 Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

Be kind, respectful, and constructive. We're all here to make AI safer!

---

## 🙏 Thank You!

Your contributions make RangeFlow better for everyone. Whether it's a bug report, documentation fix, or major feature - **every contribution matters**!

Welcome to the RangeFlow community! 🌊

---

*Last updated: 2024*