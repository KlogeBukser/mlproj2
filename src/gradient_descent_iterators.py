import numpy as np
from numpy.random import default_rng

"""
Classes for storing and updating hyperparameters for (stochastic) gradient descent.
# Momentum is built into RMSProp and Adam, so it's not necessary to set this parameter to anything other than 0 for these classes
"""


class Basic_grad:
    """ Basic gradient descent with constant learning rate """
    def __init__(self,X,y,theta_init,learning_rate,lmbda,logistic = False):
        self.learning_rate = learning_rate    # Initial learning rate
        self.lmbda = lmbda
        self.X = X
        self.y = y
        self.theta_init = theta_init
        self.n_datapoints = len(y)
        self.reset()

        if logistic:
            self.find_gradient = self.find_gradient_logistic
            self.predict = self.predict_logistic
        else:
            self.find_gradient = self.find_gradient_linear
            self.predict = self.predict_linear

    def predict_logistic(self,X_test):
        return np.sign(self.predict_linear(X_test))

    def find_gradient_logistic(self):
        t = self.X @ self.theta
        self.p = np.divide(1,1 + np.exp(-t))
        return 2*((1/self.n_datapoints)*self.X.T @ (self.p-self.y) + self.lmbda*self.theta)

    
    def find_gradient_linear(self):
        return 2*((1/self.n_datapoints)*self.X.T @ (self.X @ self.theta-self.y) + self.lmbda*self.theta)

    def update(self):
        gradient = self.find_gradient()
        self.change = -self.learning_rate*gradient
        self.theta += self.change

    def advance(self,n_epochs = 1):
        for epoch in range(n_epochs):
            self.update()

    def predict_linear(self,X_test,smooth = False):
        ''' "Smooth = True" makes results for RMSProp easier to read, it does not make much difference in other algos '''
        if smooth:
            return X_test @ (self.theta - 0.5*self.change)

        return X_test @ self.theta

    def reset(self):
        self.theta = self.theta_init
        self.change = np.zeros_like(self.theta)


class Momentum_grad(Basic_grad):
    def __init__(self,X,y,theta_init,learning_rate,lmbda,momentum,logistic = False):
        self.momentum = momentum
        super().__init__(X,y,theta_init,learning_rate,lmbda,logistic)

    def update(self):
        gradient = self.find_gradient()
        self.change = -(self.learning_rate*gradient + self.change*self.momentum)
        self.theta += self.change



class Basic_sgd(Basic_grad):
    def __init__(self,X,y,theta_init,learning_rate,lmbda,n_batches,logistic = False):
        super().__init__(X,y,theta_init,learning_rate,lmbda,logistic)
        self.rng = default_rng()
        self.n_batches = n_batches
        self.batch_size = int(self.n_datapoints/self.n_batches)
        self.indices = np.arange(0,self.n_batches*self.batch_size,1).reshape((self.n_batches,self.batch_size))
        

    def find_gradient(self):
        
        gradient = np.zeros((self.theta.shape))
        indices = self.rng.permuted(self.indices)
        for j in range(self.n_batches):
            k = self.rng.integers(0,self.n_batches)
            batch_indices = indices[k]
            X_b = self.X[batch_indices]
            y_b = self.y[batch_indices]
            gradient += self.find_partial_gradient(X_b,y_b)

        return gradient/self.n_batches

    def find_partial_gradient(self,X_b,y_b):
        return 2*((1/self.n_datapoints)*X_b.T @ (X_b @ self.theta-y_b) + self.lmbda*self.theta)


class Gradient_descent(Basic_sgd):
    """
    Stochastic gradient descent
    This is the base class for the algorithms ADA, RMSProp, ADAM
    """
    def __init__(self,X,y,theta_init,learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0,logistic = False):
        super().__init__(X,y,theta_init,learning_rate,lmbda,n_batches,logistic)
        self.momentum = momentum

    def update(self):
        gradient = self.find_gradient()
        self.change = -(self.learning_rate*gradient + self.change*self.momentum)
        self.theta += self.change


class ADA(Gradient_descent):
    """ Adagrad method for tuning the learning rate """
    def __init__(self, X, y, theta_init, learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0, delta = 1e-7,logistic = False):
        super().__init__(X,y,theta_init,learning_rate,lmbda,n_batches,momentum,logistic)
        self.delta = delta                                  # Small constant to avoid rounding errors/division by zero

    def update(self):
        gradient = self.find_gradient()
        self.r += np.square(gradient)
        self.change = -(np.multiply(self.learning_rate/(self.delta + np.sqrt(self.r)),gradient) + self.change*self.momentum)
        self.theta += self.change

    def reset(self):
        super().reset()
        self.r = np.zeros_like(self.theta)


class RMSProp(ADA):
    """ RMSProp method for tuning the learning rate """
    def __init__(self, X, y, theta_init, learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0, delta = 1e-6,rho = 0.9,logistic = False):
        super().__init__(X,y,theta_init,learning_rate,lmbda,n_batches,momentum,delta,logistic)
        self.rho = rho                  # Decay rate of second order momentum

    def update(self):
        gradient = self.find_gradient()
        self.r = self.rho*self.r + (1-self.rho)*np.square(gradient)
        self.change = -np.multiply(self.learning_rate/(np.sqrt(self.r + self.delta)),gradient)
        self.theta += self.change


class ADAM(ADA):
    """ ADAM method for tuning the learning rate """
    def __init__(self, X, y, theta_init, learning_rate = 0.001, lmbda = 0, n_batches = 1, momentum = 0, delta = 1e-8, rho1 = 0.9, rho2 = 0.999,logistic = False):
        super().__init__(X,y,theta_init,learning_rate,lmbda,n_batches,momentum, delta,logistic)
        self.rho1 = rho1                  # Decay rate of second order momentum
        self.rho2 = rho2                  # Decay rate of second order momentum

    def update(self):
        gradient = self.find_gradient()
        self.t += 1
        self.s = self.rho1*self.s + (1-self.rho1)*gradient
        self.r = self.rho2*self.r + (1-self.rho2)*np.square(gradient)
        s_scaled = self.s/(1-self.rho1**self.t)
        r_scaled = self.r/(1-self.rho2**self.t)
        
        self.change = -self.learning_rate*np.divide(s_scaled,self.delta + np.sqrt(r_scaled))
        self.theta += self.change

    def reset(self):
        super().reset()
        self.t = 0
        self.s = np.zeros_like(self.theta)