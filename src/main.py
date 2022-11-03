import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from generate import gen_simple
from gradient_descent import *

# the number of datapoints
n_datapoints = 100
n_iterations = 10000
learning_rate = 0.1
momentum = 0.3
cond = 1e-4
batch_size = 5
x, y = gen_simple(n_datapoints)
rng = default_rng()
beta0 = rng.standard_normal((3,1))







x_new = np.linspace(np.min(x),np.max(x),n_datapoints)
X = np.c_[np.ones(n_datapoints),x_new,np.square(x_new)]

plt.subplot(221)
plt.title(r'Stochastic gradient descent')
ypredict_sdg = X @ sgd(x, y, np.copy(beta0), cond, n_iterations, learning_rate, batch_size)
plt.plot(x_new, ypredict_sdg, '-b')
plt.plot(x, y ,'r.')
plt.ylabel('Without momentum')


plt.subplot(222)
plt.title(r'Gradient descent')
ypredict_mom = X @ momentum_descent(x, y, np.copy(beta0), cond, n_iterations, learning_rate, momentum)
plt.plot(x_new, ypredict_mom, '-b')
plt.plot(x, y ,'r.')

plt.subplot(223)
ypredict_sdg_mom = X @ sgd_mom(x, y, np.copy(beta0), cond, n_iterations, learning_rate, batch_size, momentum)
plt.plot(x_new, ypredict_sdg_mom, '-b')
plt.plot(x, y ,'r.')
plt.ylabel('With momentum')

plt.subplot(224)
ypredict = X @ simple_descent(x, y, np.copy(beta0), cond, n_iterations, learning_rate)
plt.plot(x_new, ypredict, '-b')
plt.plot(x, y ,'r.')


'''
ypredict_lin = lin_reg(x,y)
plt.plot(x, ypredict_lin, '-b')
plt.plot(x, y ,'r.')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.legend()
'''
plt.show()