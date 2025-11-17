from smoltorch.tensor import Tensor

# Test 1: Simple 2D matmul
print("Test 1 - Simple 2D matmul:")
x = Tensor([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]])      # (2, 3)
y = Tensor([[7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]])          # (3, 2)
z = x @ y                           # (2, 2)

z.backward()

print(f"z.data:\n{z.data}")
print(f"x.grad:\n{x.grad}")  # Should be z.grad @ y.T
print(f"y.grad:\n{y.grad}")  # Should be x.T @ z.grad

# Test 2: Vector-matrix multiplication
print("\nTest 2 - Vector @ Matrix:")
x = Tensor([1.0, 2.0, 3.0])        # (3,)
y = Tensor([[4.0, 5.0],
            [6.0, 7.0],
            [8.0, 9.0]])            # (3, 2)
z = x @ y                           # (2,)

z.backward()

print(f"z.data: {z.data}")
print(f"x.grad: {x.grad}")
print(f"y.grad:\n{y.grad}")