import numpy as np
from nanotorch.tensor import Tensor

class Linear:
  def __init__(self, in_features, out_features):
    """
    A linear layer: y = x @ W + b

    Args:
      in_features: input dims
      out_features: output dims
    """
    self.W = Tensor(np.random.randn(in_features, out_features) * 0.1)
    self.b = Tensor(np.zeros(out_features))
  
  def __call__(self, x):
    """
    Forward pass: y = x @ W + b
    
    Args:
      x: input tensor, shape = (batch_size, in_features)
    
    Returns:
      output tensor, shape = (batch_size, out_features)
    """
    return x @ self.W + self.b
  
  def parameters(self):
    """
    Returns:
      List of trainable parameters
    """
    return [self.W, self.b]

class MLP:
  """
  An MLP is just stacked linear layers with activations: Input → Linear → ReLU → Linear → ReLU → Linear → Output
  """
  def __init__(self, layer_sizes):
    """
    MLP with ReLU activation

    Args:
      layer_sizes: list of layer dims [input, hidden1, hidden2, ..., output]
        e.g. [2, 16, 16, 1] means:
          - input: 2 features
          - 2 hidden layers with 16 neurons each
          - output: 1 value
    """
    self.layers = []
    for i in range(len(layer_sizes) - 1):
      self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
  
  def __call__(self, x):
    """
    Forward pass with ReLU activation between layers.
    No activation on the final layer (common for regression/raw logits).
    """
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if i < len(self.layers) - 1:
        x = x.relu()
    
    return x
  
  def parameters(self):
    params = []
    for layer in self.layers:
      params.extend(layer.parameters())
    return params

class SGD:
  def __init__(self, parameters, lr=0.01):
    """
    Args
      parameters: list of Tensor objects to minimize
      lr: learning rate
    """
    self.parameters = parameters
    self.lr = lr
  
  def step(self):
    for param in self.parameters:
      param.data -= self.lr * param.grad
  
  def zero_grad(self):
    for param in self.parameters:
      param.grad = np.zeros_like(param.data, dtype=np.float64)
