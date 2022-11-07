import numpy as np
import seaborn as sns
from numpy.random import default_rng
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *
from learning_rates import *
from misc import *
from make_figure import make_subplot


n_datapoints = 100
n_iterations = 100
n_features = 3
n_learning_rates = 100
m = 5           # Number of minibatches
momentum = 0.3
batch_size = 5
x, y = gen_simple(n_features, n_datapoints)
X = make_design_1D(x,n_features)
rng = default_rng()
learning_rate = 0.01




eta_method = 'basic'

eta = make_adaptive_learner(eta_method,n_features,learning_rate)



mse = np.zeros(n_datapoints)
momentums = np.arange(0,1,0.2)
iterations = np.arange(0,n_datapoints,1)
batches = np.arange(1,10,1)


for n_batches in batches:
    theta0 = np.ones((n_features,1))*0.1
    thetas = gradient_descent(X, y, theta0, n_iterations, eta, batch_size, 0)
    for i in iterations:

        y_pred = X @ thetas[i]
        mse[i] = MSE(y_pred,y)

    plt.plot(iterations,mse,label = 'Batches = %d' % (n_batches))

plt.title('Momentum plot')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.legend()
plt.show()

#make_subplot(eta_method = 'adam', n_features = 3,learning_rate = 0.01, batch_size = 5)


