import random
from micrograd.engine import Value

class Neuron:
  def __init__(self, n_inputs: int):
    self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
    self.b = Value(random.uniform(-1, 1))
  
  def __call__(self, x: list) -> Value:
    activations = sum((w_i * x_i for w_i, x_i in zip(self.w, x)), self.b)
    out = activations.tanh()
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  def __init__(self, n_inputs: int, n_outputs: int):
    self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]
  
  def __call__(self, x: list) -> list[Value]:
    outs = [n(x) for n in self.neurons]
    return outs
  
  def parameters(self):
    return [p for n in self.neurons for p in n.parameters()]

class MLP:
  def __init__(self, n_inputs: int, n_outputs: int):
    sz = [n_inputs] + n_outputs
    self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(n_outputs))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
