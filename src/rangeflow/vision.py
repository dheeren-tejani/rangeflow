from .core import RangeTensor
from .backend import get_backend
import numpy as np

xp = get_backend()

class RangeTransform:
    """Base class for rigorous image perturbations."""
    pass

class RangeBrightness(RangeTransform):
    """
    Represents an image with uncertain lighting conditions.
    Input: Image tensor [B, C, H, W]
    Output: RangeTensor representing [Image - factor, Image + factor]
    """
    def __init__(self, brightness_limit):
        self.limit = brightness_limit

    def __call__(self, img):
        # img is a standard tensor (point)
        # We wrap it into a range representing ALL possible brightnesses
        r_img = RangeTensor.from_array(img)
        noise = RangeTensor.from_array(xp.ones_like(img) * self.limit)
        
        # The output range covers [pixel - limit, pixel + limit]
        # Clamped to valid image range [0, 1]
        low = xp.clip(img - self.limit, 0.0, 1.0)
        high = xp.clip(img + self.limit, 0.0, 1.0)
        return RangeTensor.from_range(low, high)

class RangeNoise(RangeTransform):
    """
    Represents sensor noise (Gaussian/Uniform) as a bounded interval.
    """
    def __init__(self, epsilon):
        self.eps = epsilon

    def __call__(self, img):
        # Represents the set of all images within distance epsilon (L-infinity ball)
        low = img - self.eps
        high = img + self.eps
        return RangeTensor.from_range(low, high)

def verify_invariance(model, img, transform):
    """
    Checks if the model prediction is constant for ALL perturbations.
    """
    # 1. Create the abstract "Range Image"
    range_img = transform(img)
    
    # 2. Propagate
    output_range = model(range_img)
    
    # 3. Check robustness
    # If the lower bound of the top class is higher than the upper bound 
    # of all other classes, the prediction CANNOT change.
    min_logits, max_logits = output_range.decay()
    
    # Get the prediction of the center
    center_pred = (min_logits + max_logits) / 2
    target_class = center_pred.argmax()
    
    # Logic: Min(Target) > Max(Others)
    target_min = min_logits[target_class]
    
    # Mask out target to find max of others
    mask = xp.ones_like(max_logits, dtype=bool)
    mask[target_class] = False
    others_max = xp.max(max_logits[mask])
    
    is_robust = target_min > others_max
    margin = target_min - others_max
    
    return is_robust, margin