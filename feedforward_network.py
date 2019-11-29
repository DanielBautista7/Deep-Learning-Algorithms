import numpy as np

def sigmoid(z)
	return 1/(1+np.exp(-z))

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    parameters -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
	
	W1 = np.random.randn((n_h,n_x))
	W2 = np.random.randn((n_y,n_h))
	b1 = np.zeros((n_h,1))
	b2 = np.zeros((n_y,1))
	parameters = {"W1":W1,
			"W2":W2,
			"b1":b1,
			"b2": b2}
	return parameters
	
def forward_propagation(X, parameters):
	W1 = paramters['W1']
	W2 = paramters['W2']
	b1 = paramters['b1']
	b2 = paramters['b2']
	
	Z1 = np.matmul(W1,X)+b1
	A1 = np.tanh(Z1)
	Z2 = np.matmul(W1,A1)+b2
	A2 = sigmoid(Z2)
	cache = {'Z1': Z1,
			 'A1': A1,
			 'Z2': Z2,
			 'A2': A2}
	return A2,cache
	
def compute_cost(A2, Y, paramters):
	cost = np.multiply(Y,np.log(A2)) + np.multiply(1-Y, np.log(1-A2))
	cost = -np.sum(cost)/m
	cost = np.squeeze(cost)
	assert(isinstance(cost,float))
	return cost
	
def backward_propagation(parameters, cache, X, Y):
	"""
	Implement the backward propagation
	
	Arguments:
		parameters -- python dictionary containing our parameters
		cache -- a dictionary containing 'Z1', 'A1', 'Z2', 'A2'
		X -- input data of shape (2, number of examples)
		Y -- "true" labels vector of shape (1, number of examples)
	Returns:
		grads -- python dictionary containing your gradients with respect to different parameters
	"""
	m = X.shape[1]
	
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	A2 = cache['A2']
	A1 = cache['A1']
	
	dA2 = A2 - Y
	dW2 = np.matmul(dA2, A2)/m
	db2 = sum(dA2, axis = 1, keepdims = True)/m
	dA1 = np.matmul(W2.T,dA2)
	dZ1 = np.dot(dA1,1-np.power(A1,2))
	dW1 = np.matmul(dZ1.T,A1)/m
	db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
	
	grads = {'dW2': dW2,
			 'db2': db2,
			 'dW1': dW1,
			 'db1': db1}
	return grads
	
def update_parameters(parameters, grads, learning_rate = 1.2):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	dW1 = parameters['dW1']
	db1 = parameters['db1']
	dW2 = parameters['dW2']
	db2 = parameters['db2']
	
	W1 = W1 - learning_rate*dW1
	b1 = b1 - learning_rate*db1
	W2 = W2 - learning_rate*dW2
	b2 = b2 - learning_rate*db2
	
	parameters['W1'] = W1
	parameters['b1'] = b1
	parameters['W2'] = W2
	parameters['b2'] = b2
	return parameters
	
def nn_model(X, Y, n_h, num_iterations = 1000, print_cost = False):
	np.random.seed(3)
	n_x = X.shape[0]
	n_y = Y.shape[0]
	
	parameters = initialize_parameters(n_x, n_h, n_y)
	
	for i in range(num_iterations):
		A2, cache = forward_propagation(X, parameters)
		cost = compute_cost(A2, Y, parameters)
		grads = backward_propagation(parameters, cache, X, Y)
		parameters = update_parameters(parameters, grads)
		if print_cost and i % 1000 == 0:
			print('Cost after iteration: %i: %f' %(i, cost))
	return parameters
	
def predict(X, parameters):
	A2, cache = forward_propagation(X, parameters)
	return np.round(A2)
		
