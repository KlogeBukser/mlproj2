import numpy as np

class Basic_learning_rate:
    def __init__(self,learning_rate):
        self.epsilon = learning_rate
    
    def update(self,g):
        return -self.epsilon*g

    def reset(self):
        return

class ADA:
    """ Adagrad method for tuning the learning rate """
    def __init__(self, n_features, learning_rate, delta = 1e-7,):
        self.epsilon = learning_rate
        self.delta = delta
        self.feature_count = n_features
        #self.r = np.zeros((n_features,1))
        self.reset()

    def update(self,g):
        """
        Updates learning rate by changin r, returns update for theta
        g: gradient
        return: delta theta 
        """
        self.r += np.square(g)
        return -np.multiply(self.epsilon/(self.delta + np.sqrt(self.r)),g)

    def reset(self):
        self.r = np.zeros((self.feature_count,1))
