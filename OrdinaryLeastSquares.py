'''
y(w,x) = w_0 + w_1*x_1 + ... + w_p*x_p

vector w = (w1 ... wp) as coef_
and
w_0 as intercept_ 

	LinearRegression fits a linear model with coefficients w=(w1 ... wp) to minimize the residual 
	sum of squares between the observed responses in the dataset, and the responses predicted by 
	the linear approximation. Mathematically it solves a problem of the form:

	min||Xw - y||2^2

	LinearRegression will take in its fit method arrays X, y and will store the coefficients w of the linear model in its coef_ member:

	However, coefficient estimates for Ordinary Least Squares rely on the independence of the model 
	terms. When terms are correlated and the columns of the design matrix X have an approximate linear 
	dependence, the design matrix becomes close to singular and as a result, the least-squares estimate 
	becomes highly sensitive to random errors in the observed response, producing a large variance. 
	This situation of multicollinearity can arise, for example, when data are collected without an experimental design.
''' 
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()

regr.fit([[0,0], [0,1], [0,2]], [0,1,2])
regr2.fit([[5,10], [10,10], [0,10]], [5,5,10])

print(regr.coef_)
print(regr2.coef_)