import numpy as np

# Sigmoid: map any value between 0 and 1 
def sig(x,deriv=False):
	if(deriv==True):
		return x*(1-x)

	return 1/(1+np.exp(-x))


#input data

X = np.array([0,0,1],
			[0,1,1],
			[1,0,1],
			[1,1,1])

y = np.array([0],
			[1],
			[1],
			[0])

np.random.seed(1)

#synapses: connections between neuron from one layer to all neurons of next layer 

s0 = 2.np.random.random((3,4)) -1 
s1 = 2.np.random.random((4,1)) -1 

for j in xrange(60000)
	#input layer 
	l0 = X 
	#create predictions for next layers
	l1 = nonlin(np.dot(l0, s0))
	l2 = nonlin(np.dot(l1, s1))

	l2_error = y - l2

	if(j % 10000) == 0:
		print "Error" + str(np.mean(np.abs(l2_error)))

	# Get delta to improve data
	l2_delta = l2_error*nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(s1.T)

	l1_delta = l1_error * nonlin(l1,deriv=True)

	#update weights, gradient descent 
	s1 += l1.T.dot(l2_delta)
	s0 += l0.T.dot(l1_delta)

print("Output after training")
print(l2)