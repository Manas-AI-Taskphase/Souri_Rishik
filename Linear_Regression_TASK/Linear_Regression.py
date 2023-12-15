import random as rand
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\SOURI\\Desktop\\Manas_TaskPhase\\train.csv')
test_data = pd.read_csv('C:\\Users\\SOURI\\Desktop\\Manas_TaskPhase\\test.csv')
data.dropna(inplace=True)
test_data.dropna(inplace=True)
X = data[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
X_test = test_data[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd']]
X1 = test_data[['OverallQual']]
X2 = test_data[['GrLivArea']]
X3 = test_data[['GarageCars']]
X4 = test_data[['GarageArea']]
X5 = test_data[['TotalBsmtSF']]
X6 = test_data[['1stFlrSF']]
X7 = test_data[['FullBath']]
X8 = test_data[['TotRmsAbvGrd']]
X9 = test_data[['YearBuilt']]
X10 = test_data[['YearRemodAdd']]
Y = data['SalePrice']
Y_test = test_data['SalePrice']
iX1 = 'OverallQual'
iX2 = 'GrLivArea'
iX3 = 'GarageCars'
iX4 = 'GarageArea'
iX5 = 'TotalBsmtSF'
iX6 = '1stFlrSF'
iX7 = 'FullBath'
iX8 = 'TotRmsAbvGrd'
iX9 = 'YearBuilt'
iX10 = 'YearRemodAdd'


def r2_score(y_true, y_pred):
    mean_y_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y_true) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    return r2


def PLOT(X, iX, Y, n):
    X = (X - X.mean()) / X.std()
    X.insert(0, "intercept", np.ones(len(X)))

    X = X.values
    Y = Y.values

    def linear_regression(X, Y, learning_rate, iterations):
        m, n = X.shape
        coefficient = np.ones(n)
        cost_function = []

        for iteration in range(iterations):
            gradient = (1 / m) * (X.T @ (X @ coefficient - Y))
            coefficient -= learning_rate * gradient
            cost = (1 / (2 * m)) * np.sum((X @ coefficient - Y) ** 2)
            cost_function.append(cost)

            predictions = X @ coefficient

            if iteration % 100 == 0:
                print(f"Epoch {iteration}: Cost = {cost}")

        return coefficient, cost_function

    learning_rate = 0.1
    iterations = 1000

    coefficent, cost_function = linear_regression(X, Y, learning_rate, iterations)

    predictions = X @ coefficent

    if n > 1:
        print("Accuracy: ",(r2_score(Y,predictions))*100)
        plt.plot(range(iterations), cost_function)
        plt.title("Cost Function Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Cost")
        plt.show()


    if n == 1:
        num = rand.randrange(292)
        sample_Data = X[num]
        actual_pred = Y[num]
        pred=np.dot(sample_Data, coefficent)
        print("Predictions:", pred, "Actual Value:", actual_pred)
        print("Error: ", abs(actual_pred-pred))

        # plotting the data and graph
        Xaxis = test_data[iX]
        plt.scatter(Xaxis, Y_test, c='b', marker='o', label='Data Points')
        plt.plot(Xaxis, predictions, c='r', label='Linear Regression Line')
        plt.title(iX + ' vs SalePrice')
        plt.xlabel(iX)
        plt.ylabel('SalePrice')
        plt.legend()
        plt.grid(True)
        plt.show()


PLOT(X, ' ', Y, len(X.columns))
PLOT(X1, iX1, Y_test, len(X1.columns))
PLOT(X2, iX2, Y_test, len(X2.columns))
PLOT(X3, iX3, Y_test, len(X3.columns))
PLOT(X4, iX4, Y_test, len(X4.columns))
PLOT(X5, iX5, Y_test, len(X5.columns))
PLOT(X6, iX6, Y_test, len(X6.columns))
PLOT(X7, iX7, Y_test, len(X7.columns))
PLOT(X8, iX8, Y_test, len(X8.columns))
PLOT(X9, iX9, Y_test, len(X9.columns))
PLOT(X10, iX10, Y_test, len(X10.columns))
