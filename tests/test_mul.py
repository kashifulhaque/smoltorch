from smoltorch.tensor import Tensor

# Test 1: Simple multiplication (no broadcasting)
print("Test 1 - No broadcasting:")
a = Tensor([2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0])
c = a * b  # [10, 18, 28]

c.backward()

print(f"c.data: {c.data}")
print(f"a.grad: {a.grad}")  # Should be [5, 6, 7] (b's values)
print(f"b.grad: {b.grad}")  # Should be [2, 3, 4] (a's values)

# Test 2: Broadcasting case
print("\nTest 2 - Broadcasting:")
a = Tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
b = Tensor([[2.0], [3.0]])      # shape (2, 1)
c = a * b  # shape (2, 3)

c.backward()

print(f"c.data:\n{c.data}")
print(f"a.grad shape: {a.grad.shape}, values: {a.grad}")  # Should be (1,3) with [[5, 5, 5]]
print(f"b.grad shape: {b.grad.shape}, values: {b.grad}")  # Should be (2,1) with [[6], [6]]

# Test 3: Chain rule test (addition + multiplication)
print("\nTest 3 - Chain rule:")
x = Tensor([2.0, 3.0])
y = Tensor([4.0, 5.0])
z = x * y       # [8, 15]
w = z + z       # [16, 30]

w.backward()

print(f"w.data: {w.data}")
print(f"x.grad: {x.grad}")  # Should be [8, 10] (2 * y, because w = 2*x*y)
print(f"y.grad: {y.grad}")  # Should be [4, 6] (2 * x)