# Importing various packages
from random import random, seed
import numpy as np
from numpy.random import default_rng



def lin_reg(X,y):
    """ linear regression """
    
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def find_gradient(X,y,theta):
    """ Function for finding the gradient """
    n = len(y)
    return (2.0/n)*X.T @ (X @ theta-y)

def simple_descent(X, y, theta, condition, n_epochs, eta):
    """ Simple gradient descent """
    
    dtheta = np.zeros((theta.shape))
    for iter in range(n_epochs):
        gradient = find_gradient(X,y,theta)
        dtheta = eta.update(gradient)
        theta += dtheta
        if abs(np.mean(dtheta)) < condition:
            print('Iterations GD: ',iter)
            break

    return theta



def momentum_descent(X, y, theta, condition, n_epochs, eta, momentum):
    """ Gradient descent with momentum """

    v = np.zeros((theta.shape))
    for iter in range(n_epochs):
        gradient = find_gradient(X,y,theta)
        v = momentum*v + eta.update(gradient)  
        theta += v
        if abs(np.mean(v)) < condition:
            print('Iterations GD (momentum): ',iter)
            break

    return theta


def sgd(X, y, theta, condition,n_epochs, eta, M):
    """ Stochastic gradient descent """
    rng = default_rng()
    n = len(y)
    m = int(n/M) #number of minibatches
     
    indices = np.arange(0,n,1).reshape((m,M))
    for iter in range(n_epochs):
        gradient = np.zeros((theta.shape))
        indices = rng.permuted(indices)
        for i in range(m):
            #Pick the k-th minibatch at random
            k = rng.integers(0,m)
            batch_indices = indices[k]
            X_b = X[batch_indices]
            y_b = y[batch_indices]
            g_b = find_gradient(X_b,y_b,theta)
            gradient += g_b

        dtheta = eta.update(gradient)/m
        theta += dtheta
        if abs(np.mean(dtheta)) < condition:
            print('Iterations SDG: ', iter)
            break
    
    return theta


def sgd_mom(X, y, theta, condition,n_epochs, eta, M, momentum):
    """ Stochastic gradient descent with momentum """

    rng = default_rng()
    n = len(y)
    m = int(n/M) #number of minibatches
     
    indices = np.arange(0,n,1).reshape((m,M))
    v = np.zeros((theta.shape))
    for iter in range(n_epochs):
        gradient = np.zeros((theta.shape))
        indices = rng.permuted(indices)
        for i in range(m):
            #Pick the k-th minibatch at random
            k = rng.integers(0,m)
            batch_indices = indices[k]
            X_b = X[batch_indices]
            y_b = y[batch_indices]
            g_b = find_gradient(X_b,y_b,theta)
            gradient += g_b

        v = eta.update(gradient)/m + momentum * v
        theta += v
        if abs(np.mean(v)) < condition:
            print('Iterations SDG (momentum): ', iter)
            break
    
    return theta