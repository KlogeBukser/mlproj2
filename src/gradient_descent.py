# Importing various packages
from random import random, seed
import numpy as np
from numpy.random import default_rng

def simple_descent(X, y, theta,n_epochs, eta):
    return gradient_descent(X, y, theta,n_epochs, eta, len(y), 0)

def momentum_descent(X, y, theta,n_epochs, eta, momentum):
    return gradient_descent(X, y, theta,n_epochs, eta, len(y), momentum)

def sgd(X, y, theta,n_epochs, eta, m):
    return gradient_descent(X, y, theta,n_epochs, eta, m, 0)

def gradient_descent(X, y, theta,n_epochs, eta, m, momentum):
    """ Gradient descent with momentum """

    rng = default_rng()
    n = len(y)
    M = int(n/m) # Size of minibatches
     
    indices = np.arange(0,n,1).reshape((m,M))
    v = np.zeros((theta.shape))
    thetas = np.zeros((n_epochs,theta.shape[0]))
    for i in range(n_epochs):
        gradient = np.zeros((theta.shape))
        indices = rng.permuted(indices)
        for j in range(m):
            #Pick the k-th minibatch at random
            k = rng.integers(0,m)
            batch_indices = indices[k]
            X_b = X[batch_indices]
            y_b = y[batch_indices]
            g_b = find_gradient_ridge(X_b,y_b,theta,0.001)
            gradient += g_b

        v = eta.update(gradient/m) + momentum * v
        theta += v
        thetas[i] = theta.ravel()
        '''if abs(np.mean(v)) < condition:
            print('Iterations SDG (momentum): ', iter)
            break'''
    
    eta.reset()
    return thetas

def lin_reg(X,y):
    """ linear regression """
    
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def find_gradient(X,y,theta):
    """ Function for finding the gradient """
    n = len(y)
    return (2/n)*X.T @ (X @ theta-y)

def find_gradient_ridge(X,y,theta,lamb):
    """ Function for finding the gradient """
    n = len(y)
    return 2*((1/n)*X.T @ (X @ theta-y) + lamb*theta)
