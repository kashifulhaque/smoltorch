# 🔥 smoltorch

<div align="center">

**A tiny autograd engine and neural network library built from first principles**

[![PyPI version](https://badge.fury.io/py/smoltorch.svg)](https://badge.fury.io/py/smoltorch)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Inspired by Andrej Karpathy's micrograd, built for learning*

</div>

---

## 🎯 What is smoltorch?

smoltorch is a minimalist deep learning library that implements automatic differentiation (autograd) and neural networks from scratch using only NumPy. It's designed to be:

- **Educational**: Understand how modern deep learning frameworks work under the hood
- **Transparent**: Every operation is visible and understandable
- **Functional**: Train real models on real datasets with competitive performance
- **Minimal**: ~500 lines of readable, well-documented Python code

### Why "smoltorch"?

"Smol" + PyTorch. It's a tiny implementation that captures the essence of modern deep learning frameworks.

---

## ✨ Features

### Core Engine
- ✅ **Automatic differentiation** with dynamic computational graphs
- ✅ **NumPy-backed tensors** for efficient numerical computing
- ✅ **Broadcasting support** with proper gradient handling
- ✅ **Topological sorting** for correct backpropagation

### Operations
- **Arithmetic**: `+`, `-`, `*`, `/`, `**`
- **Matrix operations**: `@` (matmul)
- **Activations**: ReLU, tanh, sigmoid
- **Reductions**: sum, mean
- **Element-wise**: log

### Neural Networks
- **Layers**: Linear (fully connected)
- **Models**: Multi-layer perceptron (MLP)
- **Loss functions**: MSE, Binary Cross-Entropy
- **Optimizers**: SGD (Stochastic Gradient Descent)

---

## 📦 Installation

### From PyPI (recommended)
```bash
uv add smoltorch
```

### From source
```bash
git clone https://github.com/kashifulhaque/smoltorch.git
cd smoltorch
uv pip install -e .
```

### Development installation
```bash
uv pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### Basic Tensor Operations
```python
from smoltorch import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0])
y = Tensor([4.0, 5.0, 6.0])

# Operations
z = x + y           # Element-wise addition
w = x * y           # Element-wise multiplication
a = x @ y.T         # Matrix multiplication

# Backward pass
a.backward()
print(x.grad)       # Gradients computed automatically!
```

### Training a Neural Network (Regression)
```python
from smoltorch import Tensor, MLP, SGD
from sklearn.datasets import make_regression
import numpy as np

# Generate data
X, y = make_regression(n_samples=100, n_features=5, noise=10)
y = y.reshape(-1, 1)

# Create model
model = MLP([5, 16, 16, 1])  # 5 inputs -> 16 -> 16 -> 1 output
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)
    y_pred = model(X_tensor)
    
    # Compute loss (MSE)
    loss = ((y_pred - y_tensor) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.data:.4f}")
```

### Binary Classification
```python
from smoltorch import Tensor, MLP, SGD, binary_cross_entropy
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
data = load_breast_cancer()
X, y = data.data, data.target.reshape(-1, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create classifier with sigmoid output
class BinaryClassifier(MLP):
    def __call__(self, x):
        x = super().__call__(x)
        return x.sigmoid()  # Output probabilities

model = BinaryClassifier([30, 16, 8, 1])
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    X_tensor = Tensor(X)
    y_tensor = Tensor(y)
    
    y_pred = model(X_tensor)
    loss = binary_cross_entropy(y_pred, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        accuracy = ((y_pred.data > 0.5) == y).mean()
        print(f"Epoch {epoch + 1}, Loss: {loss.data:.4f}, Acc: {accuracy:.4f}")

# Result: ~96% test accuracy on breast cancer dataset! 🎉
```

---

## 📊 Real-World Performance

smoltorch achieves competitive results on standard benchmarks:

| Dataset | Task | Test Accuracy | Epochs |
|---------|------|---------------|--------|
| Breast Cancer | Binary Classification | 96.5% | 200 |
| Synthetic Regression | Regression | MSE: 95.7 | 100 |

---

## 🏗️ Architecture

### Computational Graph

smoltorch builds a dynamic computational graph during the forward pass:
```python
x = Tensor([2.0])
y = Tensor([3.0])
z = (x * y) + (x ** 2)  # Graph: z -> [+] -> [*, **] -> [x, y]

z.backward()  # Backpropagate through graph
print(x.grad)  # dz/dx = y + 2x = 3 + 4 = 7.0
```

### How Autograd Works

1. **Forward pass**: Build computational graph with operations as nodes
2. **Topological sort**: Order nodes for correct gradient flow
3. **Backward pass**: Apply chain rule in reverse topological order
4. **Gradient accumulation**: Sum gradients from multiple paths

Example with broadcasting:
```python
x = Tensor([[1, 2, 3]])    # shape (1, 3)
y = Tensor([[1], [2]])      # shape (2, 1)
z = x + y                   # shape (2, 3) - broadcasting!

z.backward()
# x.grad sums over broadcast dimensions: shape (1, 3)
# y.grad sums over broadcast dimensions: shape (2, 1)
```

---

## 🧠 Supported Operations

### Element-wise Operations
```python
z = x + y      # Addition with broadcasting
z = x - y      # Subtraction
z = x * y      # Multiplication
z = x / y      # Division
z = x ** 2     # Power
```

### Matrix Operations
```python
z = x @ y      # Matrix multiplication (with batch support)
```

### Activation Functions
```python
z = x.relu()     # ReLU: max(0, x)
z = x.tanh()     # Tanh: (e^2x - 1) / (e^2x + 1)
z = x.sigmoid()  # Sigmoid: 1 / (1 + e^-x)
```

### Reductions
```python
z = x.sum()              # Sum all elements
z = x.sum(axis=0)        # Sum along axis
z = x.mean()             # Mean of all elements
z = x.mean(axis=1)       # Mean along axis
```

### Other
```python
z = x.log()    # Natural logarithm
```

---

## 📚 Examples

Check out the `examples/` directory:

- [`train_regression.py`](examples/train_regression.py) - Train on synthetic regression data
- [`train_classification.py`](examples/train_classification.py) - Binary classification on breast cancer dataset

Run them:
```bash
uv run examples/train_regression.py
uv run examples/train_classification.py
```

---

## 🧪 Testing

Run the test suite:
```bash
uv run pytest
```

Tests cover:
- ✅ Addition with broadcasting
- ✅ Multiplication with broadcasting
- ✅ Matrix multiplication
- ✅ Activation functions (ReLU, tanh, sigmoid)
- ✅ Reductions (sum, mean)
- ✅ Linear layers
- ✅ Multi-layer perceptrons
- ✅ End-to-end training

---

## 🗺️ Roadmap

### Coming Soon
- [ ] **More optimizers**: Adam, RMSprop with momentum
- [ ] **More activations**: Leaky ReLU, ELU, Softmax
- [ ] **Regularization**: Dropout, L2 weight decay
- [ ] **Mini-batch training**: Efficient batch processing
- [ ] **Multi-class classification**: Softmax + Cross-Entropy loss

### Future
- [ ] **Convolutional layers**: CNN support for images
- [ ] **Model serialization**: Save/load weights in safetensors format
- [ ] **GPU acceleration**: Explore Metal Performance Shaders for Apple Silicon
- [ ] **Better initialization**: He initialization for ReLU networks
- [ ] **Learning rate scheduling**: Decay strategies

---

## 🎓 Learning Resources

If you're learning from smoltorch, these resources complement it well:

- [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) - The original inspiration
- [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) - Video series by Andrej Karpathy
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) - Paper on backpropagation math

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Andrej Karpathy** for [micrograd](https://github.com/karpathy/micrograd) and the brilliant educational content
- **PyTorch team** for API design inspiration
- The deep learning community for making knowledge accessible

---

## 📬 Contact

Created by Kashif - feel free to reach out!

- GitHub: [@kashifulhaque](https://github.com/kashifulhaque)
- Twitter: [@notifkash](https://twitter.com/notifkash)

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Built with ❤️ for learners and tinkerers

</div>