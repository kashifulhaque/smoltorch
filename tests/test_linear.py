from smoltorch.nn import Linear
from smoltorch.tensor import Tensor

# Test 1: Single sample forward pass
print("Test 1 - Single forward pass:")
layer = Linear(3, 2)  # 3 inputs -> 2 outputs

x = Tensor([1.0, 2.0, 3.0])  # shape (3,)
y = layer(x)                  # shape (2,)

print(f"x.shape: {x.data.shape}")
print(f"y.shape: {y.data.shape}")
print(f"y.data: {y.data}")

# Test 2: Batch forward pass
print("\nTest 2 - Batch forward pass:")
x_batch = Tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])  # shape (2, 3) - batch of 2
y_batch = layer(x_batch)             # shape (2, 2)

print(f"x_batch.shape: {x_batch.data.shape}")
print(f"y_batch.shape: {y_batch.data.shape}")

# Test 3: Backward pass
print("\nTest 3 - Backward pass:")
x = Tensor([[1.0, 2.0]])  # shape (1, 2)
layer = Linear(2, 3)       # 2 -> 3

y = layer(x)
loss = y.sum()  # Simple loss for testing

loss.backward()

print(f"W.grad shape: {layer.W.grad.shape}")  # Should be (2, 3)
print(f"b.grad shape: {layer.b.grad.shape}")  # Should be (3,)
print(f"x.grad shape: {x.grad.shape}")        # Should be (1, 2)
print(f"W.grad:\n{layer.W.grad}")
print(f"b.grad: {layer.b.grad}")