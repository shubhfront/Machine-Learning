import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

X = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

class logisticRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = 0
        self.b = 0

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))                      # sigmoid function 

    def ypred(self, x):
        return self.m*x + self.b                     # evaluating linear equation
     
    def zpred(self, x):
        return self.sigmoid(self.ypred(x))           # y=mx+b to sigmoid 

    def gradient(self, x, y):
        z = self.zpred(x)
        dz = z - y                                   
        dm = np.dot(dz, x) / len(x)                  
        db = np.sum(dz) / len(x)                     # calculating the gradients
        return dm, db
    
    def updateParameters(self, dm, db, al=0.01):
        self.m -= al * dm                               # update m
        self.b -= al * db                               # update b

    def fit(self, epochs=1000, al=0.01):
        for i in range(epochs):
            dm, db = self.gradient(self.x, self.y)      
            self.updateParameters(dm, db, al)           

    def plot(self):
        plt.scatter(self.x, self.y, color='red', label='Data points')
        x_range = np.linspace(min(self.x), max(self.x), 100)
        y_range = self.zpred(x_range)
        plt.plot(x_range, y_range, color='blue', label='Logistic Regression Curve')          
        plt.xlabel('X')
        plt.ylabel('Probability')
        plt.title('Logistic Regression Fit')
        plt.legend()
        plt.show()                                  #plotting the graph
