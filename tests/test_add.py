from smoltorch.tensor import Tensor

# Test 1: Simple addition (no broadcasting)
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])
c = a + b

c.backward()

print("Test 1 - No broadcasting:")
print(f"a.grad: {a.grad}")  # Should be [1, 1, 1]
print(f"b.grad: {b.grad}")  # Should be [1, 1, 1]

# Test 2: Broadcasting
a = Tensor([[1.0, 2.0]])       # shape (1, 2)
b = Tensor([[3.0], [4.0]])     # shape (2, 1)
c = a + b                      # shape (2, 2)

c.backward() 

print("\nTest 2 - Broadcasting:")
print(f"a.grad shape: {a.grad.shape}, values: {a.grad}")  # Should be (1,2) with [[2, 2]]
print(f"b.grad shape: {b.grad.shape}, values: {b.grad}")  # Should be (2,1) with [[2], [2]]
