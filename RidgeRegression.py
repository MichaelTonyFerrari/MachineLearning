'''
Ridge regression addresses some of the problems of Ordinary Least Squares by imposing 
a penalty on the size of coefficients. The ridge coefficients minimize a penalized residual sum of squares:

	min||Xw - y|^2 + alpha||w||^2, where alpha >= 0 that controls shrinkage, the higher the value the higher the shrinkage
'''

from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

regr = linear_model.Ridge(alpha = .5)

xs = np.array([[0,0], [0,0], [1,1]])
ys = np.array([0,.1,1])

regr.fit(xs, ys)

print(regr.coef_)
print(regr.intercept_)

