# Machine Learning Practice Library

This library contains a collection of fundamental machine learning algorithms built completely from scratch using only mathematics. The goal of this repository is to demonstrate how these models function under the hood without relying on external or black box machine learning frameworks.

## Table of Contents

1. [Supervised Learning](#supervised-learning)
2. [Unsupervised Learning](#unsupervised-learning)
3. [Projects Overview](#projects-overview)

## Supervised Learning

### Linear Regression
Predicts a continuous output based on one or more input features by fitting a straight line to the data.
* **File**: `linear-regression/linear_regression.py`
* **Inner Project**: **House Predictor**
  Predicts house prices based on input data. Built with a full frontend and backend architecture.

### Locally Weighted Linear Regression
A non parametric variation of linear regression. It fits a model to a target point by giving more weight to the training data that is closest to it locally.
* **File**: `locally-weighted-linear-regression/LWLR.py`
* **Inner Project**: **Weather Predictor**
  Predicts weather trends using local data features from 2013 to 2024.

### Logistic Regression
Used for binary classification. It outputs probabilities for a certain class using a mathematical sigmoid function.
* **File**: `logistic-regression/Logistic_Regression.py`
* **Inner Project**: **SMS Spam Detector**
  Detects whether an SMS message is spam or not. Uses a custom TF IDF Vectorizer.

### Softmax Regression
A generalization of logistic regression used for multi class classification problems.
* **File**: `softmax-regression/softmax_regression.py`
* **Inner Project**: **Digit Reader**
  A web application that reads and classifies handwritten digits.

### Gaussian Discriminative Analysis
A generative classification model that assumes the features follow a multivariate normal distribution.
* **File**: `Gaussian-Discriminative-Analysis/GDA.py`
* **Inner Project**: **Breast Cancer Predictor**
  Classifies breast cancer data into varying severity levels based on clinical features.

### Naive Bayes
A simple but effective probabilistic classifier based on applying Bayes theorem with strong independence assumptions.
* **File**: `Naive-Bayes/naive_bayes.py`

### Perceptron
The simplest type of artificial neural network used for linear binary classification.
* **File**: `perceptron/perceptron.py`
* **Inner Project**: **Digit Reader**
  An alternative frontend application to classify handwritten digits using the linear perceptron algorithm.
