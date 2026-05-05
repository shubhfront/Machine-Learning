import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

X = np.array([[1], [2], [3], [5], [6], [7]])
y = np.array([0, 0, 0, 1, 1, 1])

weight = 0
bias = 0

alpha = 0.01
pred = []

epochs = 1000

for epoch in range(epochs):
    for i in range(len(X)):
        pred = (X[i][0] * weight) + bias

        if pred >= 0:
            z = 1
        else:
            z = 0

        error = y[i] - z

        weight = X[i][0] * alpha * error
        bias = alpha * error

        if error != 0:
            error += 1

    print("Epoch", epoch , "error", error)

prediction = []
for x in X:
    z = np.dot(x, weight) + bias
    if z>=0:
        predn = 1
    else :
        predn = 0

    prediction.append(predn)

print(prediction)

