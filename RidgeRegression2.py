import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

a = np.arange(1,11)
print(a)

b = np.arange(0,10)
print(b)

y = np.ones(10)
print(y)

#10x10 Hilbert Matrix 
# Newaxis used to increase the dimension of the existing array by one more dimension
X = 1. / (a + b[:, np.newaxis])

# Graph 
n_points = 200
alphas = np.logspace(-12, -3, n_points)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)


# Get current axis 
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
# Reverse axis
ax.set_xlim(ax.get_xlim()[::-1])  
plt.xlabel('alpha')
 # Changes x and y axis limits such that all data is shown.
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()