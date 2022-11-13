# logistic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from gradient_descent_iterators import *

def accuracy_score(y,y_pred):
    n = len(y)
    I = 0
    for i in range(n):
        if (y[i] - y_pred[i]) == 0:
            I += 1
    return I/n


def logistic_reg():
    df = load_breast_cancer()

    data = df.data
    target = df.target[:,np.newaxis]

    X_train,X_test,y_train,y_test = train_test_split(data,target, train_size=0.8, test_size=0.2)

    theta_init = np.zeros((X_train.shape[1],1))

    n_epochs = 200
    epochs = np.arange(0,n_epochs,1)

    for rate in [1e-6,1e-5,1e-4]:
        iterator = Gradient_descent(X_train,y_train,np.copy(theta_init),learning_rate = rate, lmbda = 0, n_batches = 10, momentum = 0,logistic = True)    
        scores = np.empty(n_epochs)
        for i in epochs:
            iterator.advance()
            prediction = iterator.predict(X_test)
            scores[i] = accuracy_score(y_test,prediction)

        plt.title('Accuracy score')
        plt.plot(epochs,scores,label = 'Rate: %.2e' % (rate))
    plt.legend()
    plt.show()