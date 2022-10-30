# Importing various packages
from random import random, seed
import numpy as np
from numpy.random import default_rng


def lin_reg(x,y):
    n = len(y)
    X = np.c_[np.ones(n), x]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    ypredict = X @ beta
    return ypredict

def find_gradient(X,y,beta):
    n = len(y)
    return (2.0/n)*X.T @ (X @ beta-y)

def simple_descent(x,y,condition,iterations,step_size):
    """ Simple gradient descent with learning rate 1/(max eigval of hessian matrix) """
    
    rng = default_rng()
    n = len(y)
    X = np.c_[np.ones(n), x]
    beta = rng.standard_normal((2,1))
    iter = 0

    while (iter < iterations):
        gradient = find_gradient(X,y,beta)
        beta -= step_size*gradient
        iter += 1
        if abs(np.mean(gradient)) < condition:
            print('Iterations without momentum: ',iter)
            break

    ypredict = X @ beta
    return ypredict



def momentum_descent(x,y,condition,iterations, step_size, momentum):
    """ Gradient descent, with momentum, with learning rate 1/(max eigval of hessian matrix). """
    
    rng = default_rng()
    n = len(y)
    X = np.c_[np.ones(n), x]
    beta = rng.standard_normal((2,1))
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

    ypredict = X @ beta
    return ypredict
