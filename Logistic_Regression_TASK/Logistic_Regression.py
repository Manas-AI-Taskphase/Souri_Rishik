import random as rand
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\SOURI\\Desktop\\Manas_TaskPhase\\train_logistic.csv')
test_data = pd.read_csv('C:\\Users\\SOURI\\Desktop\\Manas_TaskPhase\\test_logistic.csv')
X = data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']]
X["Sex"].dtype
X["Sex"]=(X["Sex"].replace("female",1)).replace("male",0)
X_test = test_data[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']]
X1 = test_data[['Pclass']]
X2 = test_data[['Sex']]
X2["Sex"].dtype
X2["Sex"]=(X2["Sex"].replace("female",1)).replace("male",0)
X3 = test_data[['Age']]
X4 = test_data[['Siblings/Spouses Aboard']]
X5 = test_data[['Parents/Children Aboard']]
Y=data['Survived']
Y_test = test_data['Survived']
iX1 = 'Pclass'
iX2 = 'Sex'
iX3 = 'Age'
iX4 = 'Siblings/Spouses Aboard'
iX5 = 'Parents/Children Aboard'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(y, y_pred):
    m = len(y)
    cost = (-1/m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    return cost


def logistic_regression(X, Y, learning_rate, iterations):
    m, n = X.shape
    coefficients = np.zeros(n)
    cost_history = []

    for iteration in range(iterations):
        z = X @ coefficients
        predictions = sigmoid(z)
        gradient = (1 / m) * (X.T @ (predictions - Y))
        coefficients -= learning_rate * gradient
        cost = cost_function(Y, predictions)
        cost_history.append(cost)

        if iteration % 100 == 0:
            print(f"Epoch {iteration}: Cost = {cost}")

    return coefficients, cost_history


def predict(X, coefficients, threshold=0.5):
    return sigmoid(X @ coefficients) >= threshold


def accuracy(y_true, y_pred):
    return (np.mean(y_true == y_pred))*100


def PLOT(X, iX, Y, n):
    X = (X - X.mean()) / X.std()
    X.insert(0, "intercept", np.ones(len(X)))

    X = X.values
    Y = Y.values

    learning_rate = 0.001
    iterations = 25000

    coefficients, cost_history = logistic_regression(X, Y, learning_rate, iterations)

    predictions = predict(X, coefficients)

    if n > 1:
        print("Accuracy: ", accuracy(Y, predictions))
        plt.plot(range(iterations), cost_history)
        plt.title("Cost Function Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()

    if n == 1:
        num = rand.randrange(310)
        sample_data = X[num]
        actual_pred = Y[num]
        pred=predict(sample_data, coefficients)
        print(f"Predictions of {iX}: {pred} , Actual Value:{actual_pred}")
        print("Accuracy: ", accuracy([actual_pred], [pred]))
        


PLOT(X, ' ', Y, len(X.columns))
PLOT(X1, iX1, Y_test, len(X1.columns))
PLOT(X2, iX2, Y_test, len(X2.columns))
PLOT(X3, iX3, Y_test, len(X3.columns))
PLOT(X4, iX4, Y_test, len(X4.columns))
PLOT(X5, iX5, Y_test, len(X5.columns))
