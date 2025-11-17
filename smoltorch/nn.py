import numpy as np
from smoltorch.tensor import Tensor

# helper functions
def binary_cross_entropy(y_pred, y_true):
  """
  Binary cross entropy loss with numerical stability

  Args:
    y_pred: predicted probabilities, shape (batch_size, 1)
    y_true: true labels (0 or 1), shape (batch_size, 1)
  
  Returns:
    scalar loss
  """
  # clip preds to avoid 'log(0)'
  epsilon = 1e-7

  # bce: -[y*log(p) + (1-y)*log(1-p)]
  term1 = y_true * y_pred.log()
  term2 = (Tensor(1.0) - y_true) * (Tensor(1.0) - y_pred + epsilon).log()
  return -(term1 + term2).mean()

class Linear:
  def __init__(self, in_features, out_features):
    """
    A linear layer: y = x @ W + b

    Args:
      in_features: input dims
      out_features: output dims
    """
    # xavier/glorot initialization
    limit = np.sqrt(6 / (in_features + out_features))
    self.W = Tensor(np.random.uniform(-limit, limit, (in_features, out_features)))
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
