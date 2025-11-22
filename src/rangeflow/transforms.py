"""Data Augmentation as Ranges"""

from .core import RangeTensor

class RangeCompose:
    """Compose multiple transformations"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class RangeRandomCrop:
    """Crop with position uncertainty"""
    def __init__(self, crop_size, padding=4):
        self.crop_size = crop_size
        self.padding = padding
    
    def __call__(self, img):
        # Add padding uncertainty
        return RangeTensor.from_epsilon_ball(img, self.padding / img.shape[-1])

class RangeColorJitter:
    """Color variations"""
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
    
    def __call__(self, img):
        # Combine brightness and contrast uncertainty
        total_uncertainty = self.brightness + self.contrast
        return RangeTensor.from_epsilon_ball(img, total_uncertainty)

class RangeGaussianNoise:
    """Additive Gaussian noise"""
    def __init__(self, std):
        self.std = std
    
    def __call__(self, x):
        return RangeTensor.from_epsilon_ball(x, self.std * 3)

class RangeMixUp:
    """MixUp augmentation"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x1, x2, lambda_val=None):
        if lambda_val is None:
            lambda_val = np.random.beta(self.alpha, self.alpha)
        
        # MixUp creates range between two samples
        mixed = lambda_val * x1 + (1 - lambda_val) * x2
        return RangeTensor.from_epsilon_ball(mixed, abs(lambda_val - 0.5) * 0.1)