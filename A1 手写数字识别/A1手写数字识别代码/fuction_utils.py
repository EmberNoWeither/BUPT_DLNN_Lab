from math import *
import cupy as np

class math_fuction:
    def __init__(self):
        pass

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def softmax_derivative(self,x):
        s = self.softmax(x)
        jacobian_matrix = np.diag(s) - np.outer(s, s)
        return jacobian_matrix

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.maximum(0, x+1e-15/np.abs(x+1e-15))

    def leaky_relu(self, x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    def binary_cross_entropy_loss(self, y_true, y_pred):
        epsilon = 1e-15  # 避免log(0)的情况
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 将预测值限制在 [epsilon, 1-epsilon] 范围内
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)
    
    def L1_loss(self, y_true, y_pred):
        return np.sum(np.abs(y_true-y_pred))
        
    def L2_loss(self, y_true, y_pred):
        return np.sum(np.power(y_true-y_pred, 2))
        