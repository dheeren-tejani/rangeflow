from .core import RangeTensor
from .layers import (RangeLinear, RangeConv2d, RangeLayerNorm, RangeBatchNorm1d, 
                     RangeBatchNorm2d, RangeDropout, RangeAttention, RangeRNN, 
                     RangeLSTM, RangeGRU, RangeMaxPool2d, RangeAvgPool2d, 
                     RangeReLU, RangeSigmoid, RangeTanh, RangeGELU, RangeSequential)
from .loss import robust_cross_entropy, robust_mse, robust_bce
from .backend import get_device, get_backend

# New Modules
from . import vision
from . import rl
from . import analysis
from . import nlp
from . import metrics
from . import train
from . import timeseries
from . import transforms
from . import utils
from . import visualize

__version__ = "0.3.2"