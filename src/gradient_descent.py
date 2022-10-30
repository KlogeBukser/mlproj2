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

    for iter in range(iterations):
        gradient = find_gradient(X,y,beta)
        beta -= step_size*gradient
        if abs(np.mean(gradient)) < condition:
            print('Iterations GD: ',iter)
            break

    ypredict = X @ beta
    return ypredict



def momentum_descent(x,y,condition,iterations, step_size, momentum):
    """ Gradient descent, with momentum, with learning rate 1/(max eigval of hessian matrix). """
    
    rng = default_rng()
    n = len(y)
    X = np.c_[np.ones(n), x]
    beta = rng.standard_normal((2,1))
    change = 0

    for iter in range(iterations):
        gradient = find_gradient(X, y, beta)
        change = step_size * gradient + momentum * change
        beta = beta - change
        if abs(np.mean(gradient)) < condition:
            print('Iterations GD (momentum): ',iter)
            break

    ypredict = X @ beta
    return ypredict


def sgd(x, y, condition,iterations, step_size, M):
    """ M: size of batch """
    rng = default_rng()
    n = len(y)
    m = int(n/M) #number of minibatches
    X = np.c_[np.ones(n), x]
    beta = rng.standard_normal((2,1))
    indices = rng.permuted(np.arange(0,100,1)).reshape((m,M))
    for iter in range(iterations):
        gradient = np.zeros((2,1))
        for i in range(m):
            #Pick the k-th minibatch at random
            k = rng.integers(0,m)
            batch_indices = indices[k]
            X_b = X[batch_indices]
            y_b = y[batch_indices]
            g_b = find_gradient(X_b,y_b,beta)
            gradient = gradient + g_b/m

        beta -= step_size*gradient
        if abs(np.mean(gradient)) < condition:
            print('Iterations SDG: ', iter)
            break
    
    ypredict = X @ beta
    return ypredict


def sgd_mom(x, y, condition,iterations, step_size, M, momentum):
    """ M: size of batch """
    rng = default_rng()
    n = len(y)
    m = int(n/M) #number of minibatches
    X = np.c_[np.ones(n), x]
    beta = rng.standard_normal((2,1))
    indices = rng.permuted(np.arange(0,100,1)).reshape((m,M))
    change = 0
    for iter in range(iterations):
        gradient = np.zeros((2,1))
        for i in range(m):
            #Pick the k-th minibatch at random
            k = rng.integers(0,m)
            batch_indices = indices[k]
            X_b = X[batch_indices]
            y_b = y[batch_indices]
            g_b = find_gradient(X_b,y_b,beta)
            gradient = gradient + g_b/m

        change = step_size * gradient + momentum * change
        beta -= change
        if abs(np.mean(gradient)) < condition:
            print('Iterations SDG (momentum): ', iter)
            break
    
    ypredict = X @ beta
    return ypredict