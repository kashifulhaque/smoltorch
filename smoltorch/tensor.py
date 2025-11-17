import numpy as np

class Tensor:
  def __init__(self, data, _parents=(), _op=''):
    self.data = np.array(data) if not isinstance(data, np.ndarray) else data
    self._parents = _parents
    self._op = _op

    # gradient: same shape as data, init to zeros
    self.grad = np.zeros_like(self.data, dtype=np.float64)
    self._backward = lambda: None
  
  def __repr__(self) -> str:
    return f"Tensor(data={self.data}, grad={self.grad})"
  
  def __neg__(self) -> 'Tensor':
    return self * -1
  
  def __sub__(self, other: 'Tensor') -> 'Tensor':
    return self + (-other)
  
  def __rsub__(self, other: 'Tensor') -> 'Tensor':
    return Tensor(other) - self
  
  def __rmul__(self, other: 'Tensor') -> 'Tensor':
    return self * other

  def __radd__(self, other: 'Tensor') -> 'Tensor':
    return self + other
  
  def _unbroadcast(self, grad, original_shape):
    # sum over leading dimensions that were added
    while len(grad.shape) > len(original_shape):
      grad = grad.sum(axis=0)
    
    # sum over dims that were size 1 but got broadcasted
    for i, (grad_dim, orig_dim) in enumerate(zip(grad.shape, original_shape)):
      if orig_dim == 1 and grad_dim > 1:
        grad = grad.sum(axis=i, keepdims=True)
    
    return grad

  def __add__(self, other: 'Tensor') -> 'Tensor':
    # handle scalar addition
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, (self, other), '+')

    def _backward():
      # undo broadcast by summing over broadcast dims
      grad_self = self._unbroadcast(out.grad, self.data.shape)
      grad_other = other._unbroadcast(out.grad, other.data.shape)

      # accumulate grads
      self.grad += grad_self
      other.grad += grad_other
    
    out._backward = _backward
    return out
  
  def __mul__(self, other: 'Tensor') -> 'Tensor':
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data * other.data, (self, other), '*')

    def _backward():
      # local gradients with broadcasting
      grad_self = out.grad * other.data
      grad_other = out.grad * self.data
      grad_self = self._unbroadcast(grad_self, self.data.shape)
      grad_other = other._unbroadcast(grad_other, other.data.shape)

      # accumulate grads
      self.grad += grad_self
      other.grad += grad_other
    
    out._backward = _backward
    return out
  
  def __matmul__(self, other: 'Tensor') -> 'Tensor':
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data @ other.data, (self, other), '@')

    def _backward():
      self_data = self.data
      other_data = other.data

      if other_data.ndim == 1:
        grad_self = out.grad.reshape(-1, 1) @ other_data.reshape(1, -1)
      else:
        other_data_T = other_data.swapaxes(-2, -1)
        grad_self = out.grad @ other_data_T
      grad_self = self._unbroadcast(grad_self, self_data.shape)

      if self_data.ndim == 1:
        self_data_T = self_data.reshape(-1, 1)
        grad_other = self_data_T @ out.grad.reshape(1, -1)
      else:
        self_data_T = self_data.swapaxes(-2, -1)
        grad_other = self_data_T @ out.grad
      grad_other = self._unbroadcast(grad_other, other_data.shape)

      # accumulate grads
      self.grad += grad_self
      other.grad += grad_other
    
    out._backward = _backward
    return out
  
  def __pow__(self, power) -> 'Tensor':
    assert isinstance(power, (int, float)), "only support int/float powers"

    out = Tensor(self.data ** power, (self, ), f'**{power}')

    def _backward():
      self.grad += power * (self.data ** (power - 1)) * out.grad
    
    out._backward = _backward
    return out
  
  def sum(self, axis=None, keepdims=False) -> 'Tensor':
    out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self, ), 'sum')

    def _backward():
      grad = out.grad
      if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis=axis)
      
      self.grad += np.broadcast_to(grad, self.data.shape)
    
    out._backward = _backward
    return out
  
  def mean(self, axis=None, keepdims=False):
    if axis is None:
      count = self.data.size
    else:
      count = self.data.shape[axis]
    
    out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self, ), 'mean')

    def _backward():
      grad = out.grad / count
      if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis=axis)
      
      self.grad += np.broadcast_to(grad, self.data.shape)
    
    out._backward = _backward
    return out
  
  def log(self) -> 'Tensor':
    out = Tensor(np.log(self.data), (self, ), 'log')

    def _backward():
      self.grad += (1 / self.data) * out.grad
    
    out._backward = _backward
    return out
  
  def backward(self):
    # build topological order
    topo = []
    visited = set()

    def build_topo(tensor):
      if tensor not in visited:
        visited.add(tensor)
        for parent in tensor._parents:
          build_topo(parent)
        topo.append(tensor)
    
    build_topo(self)

    # init gradient of output to 1
    self.grad = np.ones_like(self.data, dtype=np.float64)

    # backprop
    for node in reversed(topo):
      node._backward()
  
  # activation functions
  def relu(self) -> 'Tensor':
    out = Tensor(np.maximum(0, self.data), (self, ), 'ReLU')

    def _backward():
      self.grad += (self.data > 0) * out.grad
    
    out._backward = _backward
    return out
  
  def tanh(self) -> 'Tensor':
    t = np.tanh(self.data)
    out = Tensor(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    
    out._backward = _backward
    return out
  
  def sigmoid(self) -> 'Tensor':
    sig = 1 / (1 + np.exp(-self.data))
    out = Tensor(sig, (self, ), 'sigmoid')

    def _backward():
      self.grad += sig * (1 - sig) * out.grad
    
    out._backward = _backward
    return out
  