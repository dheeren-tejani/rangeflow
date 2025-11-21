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