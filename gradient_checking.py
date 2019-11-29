import numpy as np

def forward_propagation(x, theta):
	J = np.matmul(x,theta)
	return J
	
def backward_propagation(x,theta):
	return x
	
def gradient_check(x, theta, epsilon):
	grad = backward_propagation(x,theta)
	J_plus = foward_propagation(x, theta + epsilon)
	J_minus = foward_propagation(x, theta - epsilon)
	gradapprox = (J_plus - J_minus)/2/epsilon
	
	dif = np.linalg.norm(grad - gradapprox)/(np.linalg.norm(grad)+np.linalg.norm(gradapprox))
	if dif < 1e-7:
		print('The gradient is correct!')
		
	else:
		print('The gradient is wrong!')
	return difference
	
def relu(X):
	return np.maximum(0,X)
	
def forward_propagation_n(X, Y, parameters):
	m = X.shape[1]
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	
	Z1 = np.matmul(W1,X) + b1
	A1 = relu(Z1)
	Z2 = np.matmul(W2,A1) + b2
	A2 = relu(Z2)
	Z3 = np.matmul(W3, A2) + b2
	A3 = relu(Z3)
	
	logprobs = -np.multiply(Y,np.log(A3)) - np.multiply(1-Y,np.log(1-A3))
	cost = np.sum(logprobs)/m
	
	cache = (Z1, A1, W1, b1,
			 Z2, A2, W2, b2,
			 Z3, A3, W3, b3)
	
	
	return cost, cache
	
def back_propagation_n(X, Y, cache):
	m = X.shape[1]
	(Z1, A1, W1, b1, Z2, A2,W2, b2, Z3, A3, W3, b3) = cache
	
	dZ3 = A3 - Y
	dW3 = np.matmul(dZ3,A2.T)/m
	db3 = np.sum(dZ3, axis = 1, keepdims = True)/m
	
	dA2 = np.matmul(W3.T, dZ3)
	dZ2 = np.multiply(dA2, np.int64(A2>0))
	dW2 = np.matmul(dZ2, A2.T)/m
	db2 = np.sum(dZ2, axis = 1, keepdims = True)
	
	dA1 = np.matmul(W2.T, dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1 > 0))
	dW1 = np.matmul(dZ1, X.T)/m
	db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
	
	gradients = {'dZ3': dZ3, 'dW3': dW3, 'db3': db3,
				 'dZ2': dZ2, 'dA2': dA2, 'dW2': dW2, 'db2': db2,
				 'dZ1': dZ1, 'dA1': dA1, 'dW1': dW1, 'db1': db1}
	return gradients

def parameters_to_vector(parameters):
	parameter_vector = np.array([])
	layer_dims = []
	for i in range(len(parameters)//2):
		W_cur = np.reshape(parameters['W%i'%(i+1)], -1)
		b_cur = np.reshape(parameters['b%i'%(i+1)], -1)
		np.concatenate([parameter_vector, W_cur, b_cur])
		
		# cache layer dimensions
		if i == 0:
			layer_dims.append(W_cur.shape[1])
		layer_dims.append(W_cur.shape[0])
	return parameter_vector, layer_dims
	
def gradients_to_vector(gradients):
	gradient_vector = np.array([])
	layer_dims = []
	for i in range(len(parameters)//2):
		dW_cur = np.reshape(gradients['dW%i'%(i+1)],-1)
		db_cur = np.reshape(gradients['db%i'%(i+1)],-1)
		np.concatenate([gradient_vector, dW_cur, db_cur])
		
		# cache layer dimensions
		if i == 0:
			layer_dims.append(dW_cur.shape[1])
		layer_dims.append(dW_cur.shape[0])
	return gradient_vector, layer_dims
	
	
def vector_to_parameters(vector, layer_dims):
	parameters = {}
	
	layer_count = len(layer_dims) - 1
	w_counter = 0
	for i in range(layer_count-1):
		n_next = layer_dims[i+1]
		n_cur = layer_dims[i]
		
		W_shape = (n_next,n_cur)
		b_shape = (n_next,1)
		
		w_parameter_count = W_shape[0]*W_shape[1]
		b_parameter_count = b_shape[0]
		# retrieve W; get the next w_parameter_count parameters; reshape to W_shape
		parameters['W%i'%(i+1)] = np.reshape(vector[w_counter:w_counter + w_parameter_count], W_shape)
		
		# update parameter_count
		w_counter += w_parameter_count
		
		# retrieve b; get the next b_parameter_count parameters; reshape to b_shape
		parameters['b%i'%(i+1)] = np.reshape(vector[w_counter: w_counter + b_parameter_count], b_shape)
		
		# update parameter_count
		w_counter += b_parameter_count
	return parameters
	
def gradient_check_n(parameters, gradients, X, Y, epsilon = 1e-7):
	
	grad_vector = gradients_to_vector(gradients)
	gradapprox = np.zeros_like(grad)
	
	parameter_vector, layer_dims = parameters_to_vector(parameters)
	for i in range(len(parameter_vector)):
		parameter_vector[i] += epsilon
		params = vector_to_parameters(parameter_vector)
		
		J_plus, _ = forward_propagation_n(X, Y, params)
		
		parameter_vector, _ = parameters_to_vector(params)
		parameter_vector[i] -= 2*epsilon
		params = vector_to_parameters(parameter_vector)
		
		J_minus, _ = forward_propagation_n(X, Y, params)
		
		gradapprox[i] = (J_plus - J_minus)/2/epsilon
		
		parameter_vector, _ = parameters_to_vector(params)
		parameter_vector[i] += epsilon
		
				
	dif = np.linalg.norm(grad_vector - gradapprox)/(np.linalg.norm(gradapprox)+np.linalg.norm(grad_vector))

	if dif > 1e-7:
		print('There is a mistake in the backward propagation! dif = %.10f'%dif)
	else:
		print('Your backpropagation works perfectly fine! dif = %.10f'%dif)
		
	return dif

