from .core import RangeTensor
from .layers import RangeLinear, RangeConv2d, RangeLayerNorm, RangeDropout, RangeAttention, RangeRNN
from .loss import robust_cross_entropy
from .backend import get_device

# New Modules
from . import vision
from . import rl
from . import analysis

__version__ = "0.2.0" # Bump version!