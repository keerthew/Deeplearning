import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def mse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)
def gradient_descent(x, y, iterations=1500, lr=0.02):
    w = 0
    b = 0
    n = len(x)
    cost_list = []
    for i in range(iterations):
        y_pred = w * x + b
        cost = mse(y, y_pred)
        cost_list.append(cost)
        dw = -(2/n) * sum(x * (y - y_pred))
        db = -(2/n) * sum(y - y_pred)
        w = w - lr * dw
        b = b - lr * db
        if i % 100 == 0:
            print("Iteration:", i+1, "Cost:", cost)
    plt.plot(cost_list, 'r.')
    plt.title("Cost vs Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    return w, b
X = np.array([33.1, 52.8, 60.4, 46.9, 58.3, 54.7, 51.6, 40.5, 47.8, 53.0,
              45.9, 55.0, 43.6, 57.4, 56.0, 49.5, 45.0, 61.0, 44.9, 39.1])

Y = np.array([32.5, 67.9, 63.0, 70.8, 86.4, 77.8, 78.9, 60.0, 74.7, 72.0,
              56.3, 81.5, 61.8, 74.9, 80.7, 61.1, 82.5, 96.5, 49.0, 57.0])

scaler = StandardScaler()
X_norm = scaler.fit_transform(X.reshape(-1,1)).flatten()
weight, bias = gradient_descent(X_norm, Y)
print("Weight:", weight)
print("Bias:", bias)
Y_pred = weight * X_norm + bias
plt.scatter(X, Y, marker='o', color='blue', label='Data Points')
plt.plot(X, Y_pred, color='green', label='Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
