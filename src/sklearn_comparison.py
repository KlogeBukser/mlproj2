import numpy as np
from make_figure import *
from sklearn.datasets import load_breast_cancer


df = load_breast_cancer()

X = df.data
y = df.target[:,np.newaxis]

log_reg_sklearn(X,y,learning_rate = 1,lmbda=0, predictions = 1000)