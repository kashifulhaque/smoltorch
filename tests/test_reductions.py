from smoltorch.tensor import Tensor

# Test 1: Sum all elements
print("Test 1 - Sum (all elements):")
x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])  # shape (2, 3)
y = x.sum()                     # 21.0, shape ()

y.backward()

print(f"y.data: {y.data}")
print(f"x.grad:\n{x.grad}")  # Should be all 1s

# Test 2: Sum along axis
print("\nTest 2 - Sum (axis=1):")
x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])  # shape (2, 3)
y = x.sum(axis=1)               # [6, 15], shape (2,)

y.backward()

print(f"y.data: {y.data}")
print(f"x.grad:\n{x.grad}")  # Should be all 1s

# Test 3: Mean all elements
print("\nTest 3 - Mean (all elements):")
x = Tensor([[2.0, 4.0],
            [6.0, 8.0]])  # shape (2, 2)
y = x.mean()               # 5.0, shape ()

y.backward()

print(f"y.data: {y.data}")
print(f"x.grad:\n{x.grad}")  # Should be all 0.25 (1/4)

# Test 4: Mean along axis
print("\nTest 4 - Mean (axis=0):")
x = Tensor([[1.0, 2.0],
            [3.0, 4.0]])  # shape (2, 2)
y = x.mean(axis=0)         # [2, 3], shape (2,)

y.backward()

print(f"y.data: {y.data}")
print(f"x.grad:\n{x.grad}")  # Should be all 0.5 (1/2)

# Test 5: Chain rule with operations
print("\nTest 5 - MSE Loss simulation:")
pred = Tensor([1.0, 2.0, 3.0])
target = Tensor([1.5, 2.5, 2.0])
diff = pred - target
squared = diff * diff
loss = squared.mean()

loss.backward()

print(f"loss.data: {loss.data}")
print(f"pred.grad: {pred.grad}")  # Should show gradient for each prediction