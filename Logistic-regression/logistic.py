import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

X = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m = len(y)
    z = w * X + b
    y_pred = sigmoid(z)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)

    cost = -(1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost

def train_logistic_regression(X, y, learning_rate=0.1, epochs=5000):
    w = 0.0
    b = 0.0
    m = len(y)
    cost_history = []

    for epoch in range(epochs):
        z = w * X + b
        y_pred = sigmoid(z)

        dw = (1 / m) * np.sum((y_pred - y) * X)
        db = (1 / m) * np.sum(y_pred - y)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Cost: {cost:.6f}")

    return w, b, cost_history

def predict_probability(X, w, b):
    z = w * X + b
    return sigmoid(z)

def predict_class(X, w, b, threshold=0.5):
    probs = predict_probability(X, w, b)
    return (probs >= threshold).astype(int)

w, b, cost_history = train_logistic_regression(X, y)

print("\nFinal weight:", w)
print("Final bias:", b)

probs = predict_probability(X, w, b)
preds = predict_class(X, w, b)

print("\nPredicted probabilities:")
print(probs)

print("\nPredicted classes:")
print(preds)

accuracy = np.mean(preds == y) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

new_x = 4.5
new_prob = predict_probability(np.array([new_x]), w, b)[0]
new_class = predict_class(np.array([new_x]), w, b)[0]

print(f"\nFor X = {new_x}")
print(f"Probability = {new_prob:.4f}")
print(f"Predicted class = {new_class}")

x_range = np.linspace(min(X) - 1, max(X) + 1, 200)
y_curve = predict_probability(x_range, w, b)

plt.scatter(X, y, label="Actual Data")
plt.plot(x_range, y_curve, label="Logistic Regression Curve")
plt.xlabel("X")
plt.ylabel("Probability / Class")
plt.title("Logistic Regression (One X and One Y)")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Cost Reduction Over Time")
plt.grid(True)
plt.show()
