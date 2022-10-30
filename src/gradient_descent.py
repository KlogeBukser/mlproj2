# Importing various packages
from random import random, seed
import numpy as np

def lin_reg(x,y):
    n = len(y)
    X = np.c_[np.ones((n,1)), x]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredict = xbnew.dot(beta)
    return ypredict

def find_gradient(X,y,beta):
    n = len(y)
    return (2.0/n)*X.T @ (X @ beta-y)

def simple_descent(x,y,condition,iterations,step_size):
    """ Simple gradient descent with learning rate 1/(max eigval of hessian matrix) """
    
    n = len(y)
    X = np.c_[np.ones((n,1)), x]
    beta = np.random.randn(2,1)
    iter = 0

    while (iter < iterations):
        gradient = find_gradient(X,y,beta)
        beta -= step_size*gradient
        iter += 1
        if abs(np.mean(gradient)) < condition:
            print('Iterations without momentum: ',iter)
            break

    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredict = xbnew.dot(beta)
    return ypredict



def momentum_descent(x,y,condition,iterations, step_size, momentum):
    """ Gradient descent, with momentum, with learning rate 1/(max eigval of hessian matrix). """
    
    n = len(y)
    X = np.c_[np.ones((n,1)), x]
    beta = np.random.randn(2,1)
    iter = 0
    change = 0

    while (iter < iterations):
        gradient = find_gradient(X, y, beta)
        change = step_size * gradient + momentum * change
        beta = beta - change

        iter += 1
        if abs(np.mean(gradient)) < condition:
            print('Iterations with momentum: ',iter)
            break

    xnew = np.array([[0],[2]])
    xbnew = np.c_[np.ones((2,1)), xnew]
    ypredict = xbnew.dot(beta)
    return ypredict
