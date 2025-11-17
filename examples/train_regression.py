from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from nanotorch.tensor import Tensor
from nanotorch.nn import MLP, SGD

# Generate synthetic regression data
print("Generating data...")
X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
y = y.reshape(-1, 1)  # Make y 2D: (200, 1)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Create model
model = MLP([5, 16, 16, 1])  # 5 inputs -> 16 -> 16 -> 1 output
optimizer = SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 100
print("\nTraining...")

for epoch in range(epochs):
    # Convert to tensors
    X_tensor = Tensor(X_train)
    y_tensor = Tensor(y_train)
    
    # Forward pass
    y_pred = model(X_tensor)
    
    # MSE loss
    loss = ((y_pred - y_tensor) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
X_test_tensor = Tensor(X_test)
y_test_tensor = Tensor(y_test)

y_pred_test = model(X_test_tensor)
test_loss = ((y_pred_test - y_test_tensor) ** 2).mean()

print(f"Test Loss (MSE): {test_loss.data:.4f}")

# Show some predictions
print("\nSample predictions:")
for i in range(5):
    print(f"True: {y_test[i, 0]:.2f}, Predicted: {y_pred_test.data[i, 0]:.2f}")