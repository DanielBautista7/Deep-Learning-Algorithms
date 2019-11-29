def initialize_parameters(n_x, n_h, n_y):
	W1 = np.random.randn((n_h,n_x))*0.01
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn((n_y,n_h))*0.01
	b2 = np.zeros((n_y,1))
	
	assert(W1.shape == (n_h,n_x))
	assert(b1.shape == (n_h,1))
	assert(W2.shape == (n_y,n_h))
	assert(b2.shape == (n_y,1))
	
	parameters = {'W1':W1,
				  'b1':b1,
				  'W2':W2,
				  'b1': b1}
	return parameters
	
def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
	parameters = {}
	for i in range(1,len(layer_dims)):
		parameters['W%i'%i] = np.random.randn((layer_dims[i],layer_dims[i-1]))*0.01
		parameters['b%i'%i] = np.zeros((layer_dims[i],1))
		assert(parameters['W%i'%i] == (layer_dims[i],layer_dims[i-1]))
		assert(parameters['b%i'%i] == (layer_dims[i],1))
	return parameters
	
def linear_forward(A, W, b):
	Z = np.matmul(W,A)+b
	assert(Z.shape == (W.shape[0],A.shape[1]))
	cache = (A,W,b)
	return Z, cache
	
def relu(X):
	return np.maximum(np.zeros_like(X),X), X
	
def sigmoid(X):
	return 1/(1+np.exp(-X)), X
	
def linear_activation_forward(A_prev, W, b, activation):
	Z, linear_cache = linear_forward(A_prev, W, b)
	A = np.zeros_like(Z)
	if activation == 'sigmoid':
		A, activation_cache = sigmoid(Z)
	elif activation == 'relu':
		A, activation_cache = relu(Z)
	
	assert(A.shape = (W.shape[0],A_prev.shape[1]))
	cache = (linear_cache, activation_cache)
	return A, cache
	
	
def L_model_forward(X, parameters):
	caches = []
	A_cur = X.copy()
	for i in range(1,len(parameters)//2):
		W = parameters['W%i'%i]
		b = parameters['b%i'%i]
		A_cur,cache = linear_activation_forward(A_cur, W, b, activation = 'relu')
		caches.append(cache)
	A_L,cache_L = linear_activation_forward(A_cur, W['W%i'%len(parameters)],b['b%i'%len(parameters)],activation = 'sigmoid')
	caches.append(cache_L)
	assert(A_L.shape == (1,X.shape[1]))
	return A_L, caches

def compute_cost(AL, Y):
	J = -np.sum(np.multiply(Y,np.log(AL)) + np.multiply(1-Y, np.log(1-AL)))/m  
	assert(J.shape == ())
	return J

def sigmoid_backward(dA, activation_cache):
	A = activation_cache
	dZ = dA*A*(1-A)
	return dZ
	
def relu_backward(dA, activation_cache):
	A = activation_cache
	dZ = dA * np.maximum(np.zeros_like(A),A)
	return dZ

def linear_backward(dZ, cache):
	A_prev, W, b = cache
	m = dZ.shape[1]
	dW = np.matmul(dZ, A_prev.T)/m
	db = np.sum(dZ,axis  = 1, keepdims = True)/m
	dA_prev = np.matmul(W.T,dZ)
	
	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(isinstance(db,float))
	
	return dA_prev, dW, db
	
def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache
	if activation == 'sigmoid':
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW, db = linear_backward(dZ, linear_cache)
	elif activation == 'relu':
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	dAL = Y/AL - (1-Y)/(1-AL)
	max_layers = len(caches)
	
	grads = {}
	
	grads['dA%i'%(max_layers-1)], grads['dW%i'%max_layers], grads['db%i'%max_layers] = linear_activation_backward(dAL, caches[-1], 'sigmoid')
	
	for i in reversed(range(max_layers-1)):
		dAl = grads['dA%i'%(i+1)]
		grads['dA%i'%(i)], grads['dW%i'%(i+1)], grads['db%i'%(i+1)] = linear_activation_backward(dAl,caches[i],'relu')
	return grads
	
def update_parameters(parameters, grads, learning_rate):
	for i in range(len(parameters)//2)
		W_id = 'dW%i'%(i+1)
		b_id = 'db%i'%(i+1)
		parameters[W_id] =  parameters[W_id] - learning_rate*grads[W_id]
		parameters[b_id] = parameters[b_id] - learning_rate*grads[b_id]
		
	return parameters


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
	n_x, n_h, n_y = layers_dims
	parameters = initialize_parameters(n_x, n_h, n_y)
	costs = []
	for i in range(num_iterations):
		A1, cache1 = linear_activation_forward(X, parameters['W1'], parameters['b1'], 'relu')
		A2, cache2 = linear_activation_forward(A1, parameters['W2'], parameters['b2'], 'sigmoid')

		cost = compute_cost(A2, Y)
		
		grads = {}
		dA2 = Y/A2 - (1-Y)/(1-A2)
		grads['dA2'] = dA2
		grads['dA1'], grads['dW2'],grads['db2'] = linear_activation_backward(dA2, cache2, 'sigmoid')
		_, grads['dW1'],grads['db1'] = linear_activation_backward(grads['dA1'], cache1, 'relu')
		
		parameters = update_parameters(parameters, grads, learning_rate)
		if print_cost and i % 100 == 0:
			print('Cost after %i iterations: %f'%(i,cost))
		if i % 100 == 0:
			costs.append(cost)
			
	plt.plot(np.squeeze(costs))
	plt.ylabel('Cost')
	plt.xlabel('Iterations (per hundreds)')
	plt.title('Learning rate = %f'%learning_rate)
	plt.show()
	
	return parameters
	
def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
	np.random.seed(1)
	parameters = initialize_parameters_deep(layer_dims)
	costs = []
	for i in range(num_iterations):
		AL, caches = L_model_forward(X,parameters)
		cost = compute_cost(AL, Y)
		grads = L_model_backward(AL,Y,caches)
		parameters = update_parameters(parameters, grads, learning_rate)
		
		if i%100 == 0 and print_cost:
			print('Cost at iteration %i: %f'%(i,cost))
		if i%100 == 0 :
			costs.append(cost)
			
	plt.plot(np.squeeze(costs))
	plt.ylabel('Cost')
	plt.xlabel('Iterations (per hundreds)')
	plt.title('Learning rate = %f'%learning_rate)
	plt.show()
	return parameters

