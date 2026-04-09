from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

with open("logistic/loan.csv") as f:
    read = csv.reader(f)
    data = list(read)

x_set, y_set = [], []

for i in range(1, len(data)):
    x_num = int(data[i][0])
    x_set.append(x_num)
    y_set.append(int(data[i][1]))

X = np.array(x_set, dtype=float).reshape(-1, 1)
y = np.array(y_set, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("accuracy: ", accuracy)

new_data = [[430]]
prediction = model.predict(new_data)
print("Prediction:", prediction)

plt.scatter(X, y, color="blue", label="Data points")
x_curve = np.linspace(min(X), max(X), 100).reshape(-1, 1)
y_prob = model.predict_proba(x_curve)[:, 1]

plt.plot(x_curve, y_prob, color="blue", label="sigmoid curve")

plt.xlabel("Credit Score")
plt.ylabel("Loan Approval")
plt.legend()
plt.title("Logistic Regression Visualisation")
plt.show()
