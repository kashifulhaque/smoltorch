import math

class Value:
  def __init__(self, data, _parents=(), _op=''):
    self.data = data
    self._parents = _parents
    self._op = _op

    # gradient
    self.grad = 0.0 # at init, the value does not affect the output
    self._backward = lambda: None
  
  def __repr__(self):
    return f"Value(data={self.data})"
  
  def __add__(self, other: 'Value') -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * out.grad
      other.grad += 1.0 * out.grad
    out._backward = _backward

    return out
  
  def __radd__(self, other: 'Value') -> 'Value':
    return self + other
  
  def __mul__(self, other: 'Value') -> 'Value':
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * out.grad
      other.grad += self.data * out.grad
    out._backward = _backward

    return out
  
  def __neg__(self) -> 'Value':
    return -1 * self
  
  def __sub__(self, other: 'Value') -> 'Value':
    return self + (-other)
  
  def __rsub__(self, other: 'Value') -> 'Value':
    return Value(other) - self
  
  def __rmul__(self, other: 'Value') -> 'Value':
    return self * other
  
  def __pow__(self, other: 'Value') -> 'Value':
    assert isinstance(other, (int, float)), "only support int/float powers for now"
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      self.grad += (other * self.data**(other - 1)) * out.grad
    out._backward = _backward

    return out
  
  def __truediv__(self, other: 'Value') -> 'Value':
    return self * other**-1
  
  def tanh(self) -> 'Value':
    x = self.data
    _tanh = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
    out = Value(_tanh, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - _tanh ** 2) * out.grad
    out._backward = _backward

    return out
  
  def exp(self) -> 'Value':
    x = self.data
    out = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += out.data * out.grad
    out._backward = _backward

    return out
  
  def backward(self):
    topo = []
    visited = set()

    def build_topo(v: 'Value'):
      if v not in visited:
        visited.add(v)
        
        for child in v._parents:
          build_topo(child)
        
        topo.append(v)
    
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()
