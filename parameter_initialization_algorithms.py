import numpy as np
def initialize_parameters_zeros(layer_dims):
	parameters = {}
	for i in range(len(layer_dims)-1):
		parameters['W%'%(i+1)] = np.zeros((layer_dims[i],layer_dims[i+1]))
		parameters['b%i'%(i+1)] = np.zeros((layer_dims[i],1))
	return parameters
	
def initialize_parameters_random(layer_dims):
	parameters = {}
	for i in range(len(layer_dims)-1):
		parameters['W%i'%(i+1)] = np.random.randn((layer_dims[i+1],layer_dims[i]))^10
		parameters['b%i'%(i+1)] = np.zeros((layer_dims[i+1],1))
	return parameters
	
# He initialization
def initialize_parameters_he(layer_dims):
	parameters = {}
	for i in range(len(layer_dims)-1):
		parameters['W%i'%(i+1)] = np.random.randn((layer_dims[i+1],layer_dims[i]))*np.sqrt(2/layer_dims[i])
		parameters['b%i'%(i+1)] = np.zeros((layer_dims[i+1],1))
	return parameters
	
def compute_cost_with_regularization(A3, Y, parameters, lambd):
	m = Y.shape[1]
	cross_entropy_cost = compute_cost(A3,Y)
	l2 = 0
	for i in range(len(parameters)//2):
		l2 += lambd*np.sum(np.square(parameters['W%'%(i+1)]))/2/m
	cost = cross_entropy_cost + l2
	return cost

