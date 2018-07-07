"""SENet in PyTorch.

See the paper "Squeeze-and-Excitation Networks" for more details.
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

