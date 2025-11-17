from smoltorch.tensor import Tensor

# Test 1: ReLU
print("Test 1 - ReLU:")
x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = x.relu()

y.backward()

print(f"x.data: {x.data}")
print(f"y.data: {y.data}")      # Should be [0, 0, 0, 1, 2]
print(f"x.grad: {x.grad}")      # Should be [0, 0, 0, 1, 1]

# Test 2: Tanh
print("\nTest 2 - Tanh:")
x = Tensor([0.0, 1.0, 2.0])
y = x.tanh()

y.backward()

print(f"x.data: {x.data}")
print(f"y.data: {y.data}")      # Should be [0, 0.76, 0.96] approx
print(f"x.grad: {x.grad}")      # Should be [1, 0.42, 0.07] approx (1 - tanh²)

# Test 3: ReLU in a computation graph
print("\nTest 3 - ReLU in computation:")
x = Tensor([[-1.0, 2.0],
            [3.0, -4.0]])
w = Tensor([[0.5, 0.5],
            [0.5, 0.5]])
z = (x @ w).relu()  # Linear layer + ReLU

z.backward()

print(f"z.data:\n{z.data}")
print(f"x.grad:\n{x.grad}")
print(f"w.grad:\n{w.grad}")