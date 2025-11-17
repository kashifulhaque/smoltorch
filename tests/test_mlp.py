import numpy as np
from smoltorch.tensor import Tensor
from smoltorch.nn import MLP

# Test 1: MLP forward pass
print("Test 1 - MLP forward pass:")
model = MLP([2, 4, 3, 1])  # 2 inputs -> 4 hidden -> 3 hidden -> 1 output

x = Tensor([[1.0, 2.0]])   # Single sample, shape (1, 2)
y = model(x)                # shape (1, 1)

print(f"Input shape: {x.data.shape}")
print(f"Output shape: {y.data.shape}")
print(f"Output value: {y.data}")

# Test 2: Batch processing
print("\nTest 2 - Batch processing:")
x_batch = Tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])  # 3 samples, shape (3, 2)
y_batch = model(x_batch)        # shape (3, 1)

print(f"Batch input shape: {x_batch.data.shape}")
print(f"Batch output shape: {y_batch.data.shape}")

# Test 3: Backward pass through entire network
print("\nTest 3 - Full backward pass:")
x = Tensor([[1.0, 2.0]])
y_pred = model(x)
y_true = Tensor([[5.0]])

# MSE loss
loss = ((y_pred - y_true) ** 2).mean()

loss.backward()

print(f"Loss: {loss.data}")
print(f"Number of parameters: {len(model.parameters())}")
print(f"First layer W.grad shape: {model.layers[0].W.grad.shape}")
print(f"Last layer W.grad shape: {model.layers[-1].W.grad.shape}")

# Verify gradients exist for all parameters
all_have_grads = all(np.any(p.grad != 0) or p.grad.shape == () 
                     for p in model.parameters())
print(f"All parameters have gradients: {all_have_grads}")