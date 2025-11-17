import numpy as np

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