import numpy as np

class Basic_learning_rate:
    """ Non-scaling learning rate """
    def __init__(self,learning_rate):
        self.epsilon = learning_rate    # Initial learning rate
    
    def update(self,g):
        return -self.epsilon*g

    def reset(self):
        return

class ADA:
    """ Adagrad method for tuning the learning rate """
    def __init__(self, n_features, learning_rate = 0.001, delta = 1e-7,):
        self.epsilon = learning_rate    # Initial learning rate
        self.delta = delta              # Small constant to avoid rounding errors/division by zero
        self.feature_count = n_features # Polynomial degree/number of features
        self.reset()                    # Function for setting member values to their initial values

    def update(self,g):
        """
        Updates learning rate by changing r, returns update for theta
        g: gradient
        return: delta theta 
        """
        self.r += np.square(g)
        return -np.multiply(self.epsilon/(self.delta + np.sqrt(self.r)),g)

    def reset(self):
        self.r = np.zeros((self.feature_count,1))

class RMSProp:
    """ RMSProp method for tuning the learning rate """
    def __init__(self, n_features, learning_rate = 0.001, delta = 1e-6,rho = 0.9):
        self.epsilon = learning_rate    # Initial learning rate
        self.delta = delta              # Small constant to avoid rounding errors/division by zero
        self.feature_count = n_features # Polynomial degree/number of features
        self.rho = rho                  # Decay rate of second order momentum
        self.reset()                    # Function for setting member values to their initial values

    def update(self,g):
        """
        Updates learning rate by changing r, returns update for theta
        g: gradient
        return: delta theta 
        """
        self.r = self.rho*self.r + (1-self.rho)*np.square(g)
        return -np.multiply(self.epsilon/(np.sqrt(self.r + self.delta)),g)

    def reset(self):
        self.r = np.zeros((self.feature_count,1))

class ADAM:
    """ ADAM method for tuning the learning rate """
    def __init__(self, n_features, learning_rate = 0.001, delta = 1e-8, rho1 = 0.9, rho2 = 0.999):
        self.epsilon = learning_rate    # Initial learning rate
        self.delta = delta              # Small constant to avoid rounding errors/division by zero
        self.feature_count = n_features # Polynomial degree/number of features
        self.rho1 = rho1                # Decay rate of first order momentum
        self.rho2 = rho2                # Decay rate of second order momentum
        self.t = 0                      # Number of updates/Param for scaling momentum values
        self.reset()                    # Function for setting member values to their initial values

    def update(self,g):
        """
        Updates learning rate by changing r,s,t
        g: gradient
        return: delta theta 
        """
        self.t += 1
        self.s = self.rho1*self.s + (1-self.rho1)*g
        s_scaled = self.s/(1-self.rho1**self.t)
        self.r = self.rho2*self.r + (1-self.rho2)*np.square(g)
        r_scaled = self.r/(1-self.rho2**self.t)
        
        fraction = np.divide(s_scaled,self.delta + np.sqrt(r_scaled))
        return -self.epsilon*np.multiply(fraction,g)

    def reset(self):
        self.r = np.zeros((self.feature_count,1))
        self.s = np.zeros((self.feature_count,1))
        self.t = 0