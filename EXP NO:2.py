"""
Logistic Regression (from scratch, with NumPy)
Dataset   : Breast-cancer (sklearn)
Author    : <you>
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train = (X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-8)
X_test  = (X_test  - X_test.mean(axis=0))  / (X_test.std(axis=0)  + 1e-8)


class LogisticRegressionModel:
    """Binary logistic regression implemented with NumPy."""

    def __init__(self, learning_rate: float = 0.01, num_iterations: int = 1000):
        self.learning_rate   = learning_rate
        self.num_iterations  = num_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for iteration in range(self.num_iterations):
            linear_model   = np.dot(X, self.weights) + self.bias
            y_predicted    = self._sigmoid(linear_model)

            cost = -np.mean(
                y * np.log(y_predicted + 1e-8)
                + (1 - y) * np.log(1 - y_predicted + 1e-8)
            )

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            if (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1:>5}/{self.num_iterations} — "
                    f"Cost: {cost:.4f}"
                )

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self._predict_proba(X)
        return (proba > 0.5).astype(int)

if __name__ == "__main__":
    lr_model = LogisticRegressionModel(learning_rate=0.001, num_iterations=5000)
    lr_model.fit(X_train, y_train)

    y_pred   = lr_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel accuracy on test set:", accuracy)
