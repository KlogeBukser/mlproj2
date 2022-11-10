import numpy as np
from numpy.random import default_rng

def sgd_one_epoch(X,y,rng,theta,n_batches,lmbda,eta,indices):
    gradient = np.zeros((theta.shape))
    indices = rng.permuted(indices)
    for j in range(n_batches):
        k = rng.integers(0,n_batches)
        batch_indices = indices[k]
        X_b = X[batch_indices]
        y_b = y[batch_indices]
        g_b = find_gradient(X_b,y_b,theta,lmbda)
        gradient += g_b

    return eta.update(gradient/n_batches)


def gradient_descent(X, y, new_theta, lmbda, n_epochs, eta, n_batches):
    """ Gradient descent """

    rng = default_rng()
    n_datapoints = len(y)
    batch_size = int(n_datapoints/n_batches)
     
    indices = np.arange(0,n_batches*batch_size,1).reshape((n_batches,batch_size))
    for i in range(n_epochs):
        theta = np.copy(new_theta)
        change = sgd_one_epoch(X,y,rng,theta,n_batches,lmbda,eta,indices)
        new_theta = theta + change
    eta.reset()
    return 0.5*(theta + new_theta)

def lin_reg(X,y):
    """ linear regression """
    
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def find_gradient(X,y,theta,lamb):
    """ Function for finding the gradient """
    n = len(y)
    return 2*((1/n)*X.T @ (X @ theta-y) + lamb*theta)
