"""
RangeFlow Test Suite
====================
Complete tests for all RangeFlow functionality.

Run with: pytest tests/
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rangeflow import RangeTensor
from src.rangeflow.layers import (RangeLinear, RangeConv2d, RangeLayerNorm, 
                                   RangeBatchNorm1d, RangeReLU, RangeSequential)
from src.rangeflow.loss import robust_cross_entropy, robust_mse
from src.rangeflow.backend import get_backend

xp = get_backend()


class TestRangeTensorBasics:
    """Test basic RangeTensor functionality"""
    
    def test_creation_from_range(self):
        """Test creating RangeTensor from min/max"""
        r = RangeTensor.from_range(1.0, 2.0)
        min_val, max_val = r.decay()
        
        assert float(min_val) == 1.0
        assert float(max_val) == 2.0
    
    def test_creation_from_epsilon_ball(self):
        """Test creating epsilon ball"""
        center = np.array([5.0])
        r = RangeTensor.from_epsilon_ball(center, epsilon=0.5)
        min_val, max_val = r.decay()
        
        assert float(min_val) == pytest.approx(4.5, abs=1e-5)
        assert float(max_val) == pytest.approx(5.5, abs=1e-5)
    
    def test_creation_from_array(self):
        """Test creating from array (degenerate range)"""
        data = np.array([1.0, 2.0, 3.0])
        r = RangeTensor.from_array(data)
        min_val, max_val = r.decay()
        
        np.testing.assert_array_almost_equal(min_val, max_val)
    
    def test_len_function(self):
        """Test range width calculation"""
        r = RangeTensor.from_range(1.0, 4.0)
        width = r.len()
        
        assert float(width) == pytest.approx(3.0, abs=1e-5)
    
    def test_avg_function(self):
        """Test range center calculation"""
        r = RangeTensor.from_range(2.0, 8.0)
        center = r.avg()
        
        assert float(center) == pytest.approx(5.0, abs=1e-5)
    
    def test_relative_width(self):
        """Test relative width calculation"""
        r = RangeTensor.from_range(10.0, 12.0)
        rel_width = r.relative_width()
        
        # Width = 2, Center = 11, Relative = 2/11 ≈ 0.182
        assert float(rel_width) == pytest.approx(2.0/11.0, abs=1e-5)


class TestRangeTensorArithmetic:
    """Test arithmetic operations"""
    
    def test_addition(self):
        """Test range addition"""
        r1 = RangeTensor.from_range(1.0, 2.0)
        r2 = RangeTensor.from_range(3.0, 4.0)
        
        result = r1 + r2
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(4.0, abs=1e-5)
        assert float(max_val) == pytest.approx(6.0, abs=1e-5)
    
    def test_subtraction(self):
        """Test range subtraction"""
        r1 = RangeTensor.from_range(5.0, 7.0)
        r2 = RangeTensor.from_range(1.0, 2.0)
        
        result = r1 - r2
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(3.0, abs=1e-5)
        assert float(max_val) == pytest.approx(6.0, abs=1e-5)
    
    def test_multiplication(self):
        """Test range multiplication"""
        r1 = RangeTensor.from_range(2.0, 3.0)
        r2 = RangeTensor.from_range(4.0, 5.0)
        
        result = r1 * r2
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(8.0, abs=1e-5)
        assert float(max_val) == pytest.approx(15.0, abs=1e-5)
    
    def test_multiplication_with_negatives(self):
        """Test multiplication handles negative values correctly"""
        r1 = RangeTensor.from_range(-2.0, 1.0)
        r2 = RangeTensor.from_range(-1.0, 2.0)
        
        result = r1 * r2
        min_val, max_val = result.decay()
        
        # Min should be -4 (from -2 * 2), Max should be 2 (from -2 * -1)
        assert float(min_val) == pytest.approx(-4.0, abs=1e-5)
        assert float(max_val) == pytest.approx(2.0, abs=1e-5)
    
    def test_scalar_addition(self):
        """Test adding scalar to range"""
        r = RangeTensor.from_range(1.0, 2.0)
        result = r + 5.0
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(6.0, abs=1e-5)
        assert float(max_val) == pytest.approx(7.0, abs=1e-5)


class TestRangeTensorActivations:
    """Test activation functions"""
    
    def test_relu(self):
        """Test ReLU activation"""
        r = RangeTensor.from_range(-2.0, 3.0)
        result = r.relu()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(0.0, abs=1e-5)
        assert float(max_val) == pytest.approx(3.0, abs=1e-5)
    
    def test_sigmoid(self):
        """Test sigmoid activation"""
        r = RangeTensor.from_range(-1.0, 1.0)
        result = r.sigmoid()
        
        min_val, max_val = result.decay()
        # sigmoid(-1) ≈ 0.268, sigmoid(1) ≈ 0.731
        assert float(min_val) == pytest.approx(0.268, abs=0.01)
        assert float(max_val) == pytest.approx(0.731, abs=0.01)
    
    def test_tanh(self):
        """Test tanh activation"""
        r = RangeTensor.from_range(-1.0, 1.0)
        result = r.tanh()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(np.tanh(-1.0), abs=1e-5)
        assert float(max_val) == pytest.approx(np.tanh(1.0), abs=1e-5)
    
    def test_exp(self):
        """Test exponential"""
        r = RangeTensor.from_range(0.0, 1.0)
        result = r.exp()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(1.0, abs=1e-5)
        assert float(max_val) == pytest.approx(np.e, abs=1e-5)


class TestNonMonotonicFunctions:
    """Test non-monotonic functions with critical points"""
    
    def test_square_positive_range(self):
        """Test squaring positive range"""
        r = RangeTensor.from_range(2.0, 3.0)
        result = r.square()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(4.0, abs=1e-5)
        assert float(max_val) == pytest.approx(9.0, abs=1e-5)
    
    def test_square_negative_range(self):
        """Test squaring negative range"""
        r = RangeTensor.from_range(-3.0, -2.0)
        result = r.square()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(4.0, abs=1e-5)
        assert float(max_val) == pytest.approx(9.0, abs=1e-5)
    
    def test_square_crossing_zero(self):
        """Test squaring range that crosses zero (critical point!)"""
        r = RangeTensor.from_range(-3.0, 2.0)
        result = r.square()
        
        min_val, max_val = result.decay()
        # Min should be 0 (at critical point), Max should be 9 (from -3)
        assert float(min_val) == pytest.approx(0.0, abs=1e-5)
        assert float(max_val) == pytest.approx(9.0, abs=1e-5)
    
    def test_abs(self):
        """Test absolute value"""
        r = RangeTensor.from_range(-5.0, 3.0)
        result = r.abs()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(0.0, abs=1e-5)
        assert float(max_val) == pytest.approx(5.0, abs=1e-5)
    
    def test_sqrt(self):
        """Test square root"""
        r = RangeTensor.from_range(4.0, 9.0)
        result = r.sqrt()
        
        min_val, max_val = result.decay()
        assert float(min_val) == pytest.approx(2.0, abs=1e-5)
        assert float(max_val) == pytest.approx(3.0, abs=1e-5)


class TestLayers:
    """Test neural network layers"""
    
    def test_range_linear(self):
        """Test RangeLinear layer"""
        layer = RangeLinear(3, 2, bias=True)
        x = RangeTensor.from_range(
            np.array([[1.0, 2.0, 3.0]]),
            np.array([[1.1, 2.1, 3.1]])
        )
        
        result = layer(x)
        min_val, max_val = result.decay()
        
        assert min_val.shape == (1, 2)
        assert max_val.shape == (1, 2)
        assert np.all(min_val <= max_val)
    
    def test_range_relu_layer(self):
        """Test RangeReLU as layer"""
        relu = RangeReLU()
        x = RangeTensor.from_range(-2.0, 3.0)
        
        result = relu(x)
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(0.0, abs=1e-5)
        assert float(max_val) == pytest.approx(3.0, abs=1e-5)
    
    def test_range_sequential(self):
        """Test sequential container"""
        model = RangeSequential(
            RangeLinear(3, 5),
            RangeReLU(),
            RangeLinear(5, 2)
        )
        
        x = RangeTensor.from_epsilon_ball(np.random.randn(1, 3), 0.1)
        result = model(x)
        
        min_val, max_val = result.decay()
        assert min_val.shape == (1, 2)
        assert np.all(min_val <= max_val)
    
    def test_range_layernorm(self):
        """Test layer normalization"""
        norm = RangeLayerNorm(5)
        x = RangeTensor.from_epsilon_ball(np.random.randn(2, 5), 0.1)
        
        result = norm(x)
        min_val, max_val = result.decay()
        
        assert min_val.shape == (2, 5)
        assert np.all(min_val <= max_val)


class TestLossFunctions:
    """Test robust loss functions"""
    
    def test_robust_mse(self):
        """Test robust MSE loss"""
        pred_range = RangeTensor.from_range(
            np.array([[1.0, 2.0]]),
            np.array([[1.5, 2.5]])
        )
        target = np.array([[1.2, 2.2]])
        
        loss = robust_mse(pred_range, target, mode='worst_case')
        assert isinstance(loss, (float, np.ndarray))
        assert loss >= 0
    
    def test_robust_cross_entropy(self):
        """Test robust cross-entropy"""
        # Create logits range
        logits_range = RangeTensor.from_range(
            np.array([[1.0, 0.5, 0.2]]),
            np.array([[1.5, 1.0, 0.8]])
        )
        labels = np.array([0])  # Correct class is 0
        
        loss = robust_cross_entropy(logits_range, labels, mode='worst_case')
        assert isinstance(loss, (float, np.ndarray))
        assert loss >= 0


class TestRangeOperations:
    """Test range-specific operations"""
    
    def test_union(self):
        """Test range union"""
        r1 = RangeTensor.from_range(1.0, 3.0)
        r2 = RangeTensor.from_range(2.0, 4.0)
        
        result = r1.union(r2)
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(1.0, abs=1e-5)
        assert float(max_val) == pytest.approx(4.0, abs=1e-5)
    
    def test_intersection(self):
        """Test range intersection"""
        r1 = RangeTensor.from_range(1.0, 5.0)
        r2 = RangeTensor.from_range(3.0, 7.0)
        
        result = r1.intersection(r2)
        min_val, max_val = result.decay()
        
        assert float(min_val) == pytest.approx(3.0, abs=1e-5)
        assert float(max_val) == pytest.approx(5.0, abs=1e-5)
    
    def test_transpose_range(self):
        """Test range transpose"""
        r = RangeTensor.from_range(2.0, 5.0)
        result = r.transpose_range()
        
        min_val, max_val = result.decay()
        # Should flip: [2, 5] -> [5, 2]
        assert float(min_val) == pytest.approx(5.0, abs=1e-5)
        assert float(max_val) == pytest.approx(2.0, abs=1e-5)
    
    def test_contains(self):
        """Test containment check"""
        r1 = RangeTensor.from_range(1.0, 10.0)
        r2 = RangeTensor.from_range(3.0, 7.0)
        
        assert r1.contains(r2) == True
        assert r2.contains(r1) == False
    
    def test_overlaps(self):
        """Test overlap check"""
        r1 = RangeTensor.from_range(1.0, 5.0)
        r2 = RangeTensor.from_range(4.0, 8.0)
        r3 = RangeTensor.from_range(10.0, 15.0)
        
        assert r1.overlaps(r2) == True
        assert r1.overlaps(r3) == False


class TestComparison:
    """Test comparison operations"""
    
    def test_less_than(self):
        """Test less than (by center)"""
        r1 = RangeTensor.from_range(1.0, 2.0)  # center = 1.5
        r2 = RangeTensor.from_range(3.0, 4.0)  # center = 3.5
        
        assert r1 < r2
        assert not (r2 < r1)
    
    def test_is_wider_than(self):
        """Test width comparison"""
        r1 = RangeTensor.from_range(1.0, 5.0)  # width = 4
        r2 = RangeTensor.from_range(2.0, 3.0)  # width = 1
        
        assert r1.is_wider_than(r2)
        assert not r2.is_wider_than(r1)


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_zero_width_range(self):
        """Test degenerate range (point)"""
        r = RangeTensor.from_range(5.0, 5.0)
        width = r.len()
        
        assert float(width) == pytest.approx(0.0, abs=1e-5)
        assert r.is_degenerate(tol=1e-6) == True
    
    def test_large_values(self):
        """Test with large values"""
        r = RangeTensor.from_range(1e6, 1e6 + 100)
        width = r.len()
        
        assert float(width) == pytest.approx(100.0, abs=1e-3)
    
    def test_negative_ranges(self):
        """Test with negative ranges"""
        r = RangeTensor.from_range(-100.0, -50.0)
        center = r.avg()
        
        assert float(center) == pytest.approx(-75.0, abs=1e-5)


class TestIntegration:
    """Integration tests with full workflow"""
    
    def test_simple_network_forward(self):
        """Test forward pass through simple network"""
        model = RangeSequential(
            RangeLinear(10, 20),
            RangeReLU(),
            RangeLinear(20, 5)
        )
        
        # Create input with uncertainty
        x = RangeTensor.from_epsilon_ball(np.random.randn(3, 10), epsilon=0.1)
        
        # Forward pass
        output = model(x)
        min_out, max_out = output.decay()
        
        # Verify output properties
        assert min_out.shape == (3, 5)
        assert max_out.shape == (3, 5)
        assert np.all(min_out <= max_out)
        
        # Verify uncertainty propagated
        output_width = output.width().mean()
        assert output_width > 0  # Should have some uncertainty
    
    def test_robustness_preservation(self):
        """Test that robustness is preserved through layers"""
        layer = RangeLinear(5, 5)
        
        # Small perturbation
        x_clean = np.random.randn(1, 5)
        epsilon = 0.01
        x_range = RangeTensor.from_epsilon_ball(x_clean, epsilon)
        
        # Forward
        y_range = layer(x_range)
        
        # Output should still be a valid range
        min_y, max_y = y_range.decay()
        assert np.all(min_y <= max_y)
        
        # Width should be bounded
        width = y_range.width().mean()
        assert width < 100  # Shouldn't explode


# ==========================================
# PYTEST CONFIGURATION
# ==========================================

def pytest_configure(config):
    """Configure pytest"""
    print("\n" + "="*70)
    print("RangeFlow Test Suite")
    print("="*70 + "\n")


def pytest_collection_finish(session):
    """Print test summary"""
    print(f"\nCollected {len(session.items)} tests\n")


# ==========================================
# RUN TESTS
# ==========================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])