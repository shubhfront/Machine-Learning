import numpy as np

np.random.seed(42)

X = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [6.2, 3.4, 5.4, 2.3],
    [5.9, 3.0, 5.1, 1.8],
    [6.0, 2.2, 4.0, 1.0],
    [5.6, 2.9, 3.6, 1.3],
    [5.5, 2.3, 4.0, 1.3],
    [6.7, 3.1, 4.7, 1.5],
    [4.7, 3.2, 1.3, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9]
])

y = np.array([0, 0, 2, 2, 1, 1, 1, 1, 0, 0, 2, 2])

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

num_samples = X.shape[0]
num_features = X.shape[1]
num_classes = len(np.unique(y))

Y = np.zeros((num_samples, num_classes))
Y[np.arange(num_samples), y] = 1

W = np.random.randn(num_features, num_classes) * 0.01
b = np.zeros((1, num_classes))

learning_rate = 0.1
epochs = 1000

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

for epoch in range(epochs):
    logits = np.dot(X, W) + b
    probabilities = softmax(logits)

    loss = cross_entropy(Y, probabilities)

    dW = np.dot(X.T, probabilities - Y) / num_samples
    db = np.sum(probabilities - Y, axis=0, keepdims=True) / num_samples

    W = W - learning_rate * dW
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss)

logits = np.dot(X, W) + b
probabilities = softmax(logits)
predictions = np.argmax(probabilities, axis=1)

accuracy = np.mean(predictions == y)

print("Predictions:", predictions)
print("Actual:", y)
print("Accuracy:", accuracy)
