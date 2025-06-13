import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = X.shape[1]
hidden_dim = 10
output_dim = 1

w1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
w2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(y, y_hat):
    return -np.mean(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

losses = []
val_losses = []

for epoch in range(1, 51):
    z1 = np.dot(x_train, w1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    train_loss = loss(y_train, a2)

    dz2 = a2 - y_train
    dw2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (1 - np.power(a1, 2)) * np.dot(dz2, w2.T)
    dw1 = np.dot(x_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    w1 -= 0.01 * dw1
    b1 -= 0.01 * db1
    w2 -= 0.01 * dw2
    b2 -= 0.01 * db2

    z1_test = np.dot(x_test, w1) + b1
    a1_test = np.tanh(z1_test)
    z2_test = np.dot(a1_test, w2) + b2
    a2_test = sigmoid(z2_test)

    val_loss = loss(y_test, a2_test)

    losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} - loss: {train_loss:.4f}, val-loss: {val_loss:.4f}")

y_pred = (a2_test > 0.5).astype(int)
accuracy = np.mean(y_pred == y_test) * 100
print(f"\nFinal Test Accuracy: {accuracy:.4f}")


plt.plot(range(1, 51), losses, label='Train Loss', linestyle='--')
plt.plot(range(1, 51), val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.show()
