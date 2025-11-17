import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nanotorch.tensor import Tensor
from nanotorch.nn import MLP, SGD, binary_cross_entropy

# Load breast cancer dataset (binary classification)
print("Loading breast cancer dataset...")
data = load_breast_cancer()
X, y = data.data, data.target.reshape(-1, 1)

# Normalize features (important for neural networks!)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# Create model (note: output goes through sigmoid)
class BinaryClassifier(MLP):
    def __call__(self, x):
        # Forward pass through MLP
        x = super().__call__(x)
        # Apply sigmoid for probabilities
        return x.sigmoid()

model = BinaryClassifier([30, 16, 8, 1])  # 30 features -> 1 output probability
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop
epochs = 200
print("\nTraining...")

for epoch in range(epochs):
    # Convert to tensors
    X_tensor = Tensor(X_train)
    y_tensor = Tensor(y_train)
    
    # Forward pass
    y_pred = model(X_tensor)
    
    # Binary cross-entropy loss
    loss = binary_cross_entropy(y_pred, y_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # After loss.backward(), before optimizer.step()
    if (epoch + 1) % 20 == 0:
        # Check gradient magnitudes
        grad_norm = sum(np.sum(p.grad**2) for p in model.parameters())
        print(f"Epoch {epoch + 1}, Loss: {loss.data:.4f}, Grad norm: {grad_norm:.6f}")
    
    # Update weights
    optimizer.step()
    
    # Calculate accuracy
    if (epoch + 1) % 20 == 0:
        predictions = (y_pred.data > 0.5).astype(float)
        accuracy = (predictions == y_train).mean()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data:.4f}, Accuracy: {accuracy:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
X_test_tensor = Tensor(X_test)
y_test_tensor = Tensor(y_test)

y_pred_test = model(X_test_tensor)
test_loss = binary_cross_entropy(y_pred_test, y_test_tensor)

predictions = (y_pred_test.data > 0.5).astype(float)
test_accuracy = (predictions == y_test).mean()

print(f"Test Loss: {test_loss.data:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nSample predictions:")
for i in range(5):
    prob = y_pred_test.data[i, 0]
    pred = "Malignant" if prob > 0.5 else "Benign"
    true = "Malignant" if y_test[i, 0] == 1 else "Benign"
    print(f"True: {true}, Predicted: {pred} (prob: {prob:.3f})")