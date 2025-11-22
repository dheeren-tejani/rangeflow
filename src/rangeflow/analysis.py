from .core import RangeTensor
import numpy as np

def check_quantization_robustness(model, inputs, bits=8):
    """
    Simulates quantization noise to check if model breaks.
    int8 quantization introduces error approx range / 2^8.
    """
    # Calculate quantization noise magnitude
    dynamic_range = inputs.max() - inputs.min()
    quant_noise = dynamic_range / (2**bits)
    
    # Create Range Input representing Quantization Error
    # Center = input, Width = 2 * quant_noise
    # This covers any value the quantized input could snap to.
    r_input = RangeTensor.from_range(inputs - quant_noise, inputs + quant_noise)
    
    # Propagate
    output = model(r_input)
    min_out, max_out = output.decay()
    
    # Check consistency: Does the argmax change across the range?
    pred_min = np.argmax(min_out, axis=1)
    pred_max = np.argmax(max_out, axis=1)
    
    # If min_prediction == max_prediction, the model is robust to int8 conversion
    stable_count = np.sum(pred_min == pred_max)
    score = stable_count / len(inputs)
    
    return score

def sensitivity_analysis(model, input_data, epsilon_values):
    """Analyze how output changes with input perturbation"""
    results = []
    for eps in epsilon_values:
        input_range = RangeTensor.from_epsilon_ball(input_data, eps)
        output_range = model(input_range)
        width = output_range.width().mean()
        results.append({'epsilon': eps, 'output_width': width})
    return results

def layer_wise_uncertainty(model, input_range):
    """Track uncertainty through each layer"""
    widths = []
    centers = []
    
    x = input_range
    for name, layer in model.named_modules():
        if isinstance(layer, RangeModule):
            x = layer(x)
            widths.append((name, x.width().mean()))
            centers.append((name, x.avg().mean()))
    
    return {'widths': widths, 'centers': centers}

def check_training_stability(model, data_loader, epsilon):
    """Check if model training is stable"""
    stable_count = 0
    total = 0
    
    for data, _ in data_loader:
        data_range = RangeTensor.from_epsilon_ball(data, epsilon)
        output_range = model(data_range)
        
        # Check if output width is reasonable
        width = output_range.width().mean()
        if width < 10.0:  # Arbitrary threshold
            stable_count += 1
        total += 1
    
    return stable_count / total