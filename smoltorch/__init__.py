"""
smoltorch: A tiny autograd engine and neural network library
Built from first principles for educational purposes
"""

__version__ = "0.1.0"

from smoltorch.optim import SGD
from smoltorch.tensor import Tensor
from smoltorch.nn import Linear, MLP, binary_cross_entropy

__all__ = [
  "Tensor",
  "Linear",
  "MLP",
  "binary_cross_entropy",
  "SGD"
]
