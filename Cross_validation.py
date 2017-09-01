'''
Implents ridge regression with build in cross validation, leaving one out cross validation 
'''

from sklearn import linear_model 
regr = linear_model.RidgeCV(alphas=[.1,1,10])
regr.fit([[0,0],[0,0],[1,1]],[0, .1, 1])

print(regr.alpha_)