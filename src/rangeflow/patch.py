import torch.nn as nn
from .nn import RangeLinear, RangeLayerNorm

def convert_model_to_rangeflow(model: nn.Module):
    """
    Recursively replaces standard PyTorch layers with RangeFlow layers.
    Copies the weights.
    """
    import torch
    
    # Helper to copy weights
    def copy_weights(src, dst):
        # dst.weight is a RangeTensor, src.weight is a torch.Tensor
        # We need to access the underlying value of the RangeTensor leaf
        # Assuming RangeTensor.from_array creates a LEAF with value
        # In a real library, we'd need a cleaner setter.
        # For this prototype, we assume we can re-assign the attribute.
        from .core import RangeTensor
        
        # Convert torch tensor to numpy/cupy
        w_data = src.weight.detach().cpu().numpy()
        dst.weight = RangeTensor.from_array(w_data)
        
        if src.bias is not None:
            b_data = src.bias.detach().cpu().numpy()
            dst.bias = RangeTensor.from_array(b_data)
    
    # Iterate children
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace Linear
            print(f"Patching Linear: {name}")
            new_layer = RangeLinear(module.in_features, module.out_features, bias=module.bias is not None)
            copy_weights(module, new_layer)
            setattr(model, name, new_layer)
            
        elif isinstance(module, nn.LayerNorm):
            # Replace LayerNorm
            print(f"Patching LayerNorm: {name}")
            new_layer = RangeLayerNorm(module.normalized_shape, eps=module.eps)
            copy_weights(module, new_layer)
            setattr(model, name, new_layer)
            
        elif isinstance(module, nn.Conv2d):
            # Need to implement RangeConv2d fully in nn.py first
            # For now, we skip or warn
            print(f"Warning: Conv2d patching not fully automated yet for {name}")
            
        else:
            # Recurse
            convert_model_to_rangeflow(module)
            
    return model