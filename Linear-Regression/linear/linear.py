import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qt5Agg")

class linearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = 0
        self.b = 0               # initializing the parametres
    
    def predict(self, x):
        return self.m*x + self.b   # y=mx+c
    
    def error(self, x, y):
        yPred = self.predict(x)
        return np.mean((y-yPred)**2)    # mean squared error

    def gradient_descent(self, al=0.01, epoch=1000):
        n = len(self.x)
        for i in range(epoch):
            yPred = self.predict(self.x)
            grad_m = (-2/n)*np.sum(self.x*(self.y-yPred))
            grad_b = (-2/n)*np.sum(self.y-yPred)
            self.m = self.m - al*grad_m                         # update m
            self.b = self.b - al*grad_b                         # update b

    def fit(self, al=0.01, epoch=1000):
        self.gradient_descent(al, epoch)
        print(f"m: {self.m}, b: {self.b}, error: {self.error(self.x, self.y)}")     # providing the model with the data set to train

    def plot(self):
        plt.scatter(self.x, self.y, color='blue', label='Data points')                  # plot the graph for the model 
        plt.plot(self.x, self.predict(self.x), color='red', label='Regression line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()





