import numpy as np

''' Implementations of gradient descent, momentum, and adam optimization '''

# Gradient Descent optimization
def update_parameters_with_gd(parameters, grads, learning_rate):
	for i in range(len(parameters)//2):
		W_cur = parameters['W%i'%(i+1)] 
		b_cur = parameters['b%i'%(i+1)] 
		dW_cur = grads['dW%i'(i+1)]
		db_cur = grads['db%i'(i+1)]
		
		parameters['W%i'%(i+1)] = W_cur - learning_rate*dW_cur
		parameters['b%i'%(i+1)] = b_cur - learning_rate*db_cur
		
	return parameters

def shuffle(X, Y):
	idx = list(np.random.permutation(X.shape[1]))
	X_shuffled = np.copy(X)[:,idx]
	Y_shuffled = np.copy(Y)[idx,:].reshape((1,X.shape[1]))
	return X_shuffled, Y_shuffle
	
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
	np.random.seed(seed)
	mini_batches = []
	m = X.shape[1]
	
	X_shuffled, Y_shuffled = shuffly(X,Y)
	
	mini_batch_count = m//mini_batch_size
	
	for i in range(mini_batch_count):
		mini_batch_X = X_shuffled[:,i*mini_batch_size: (i+1)*mini_batch_size]	
		mini_batch_Y = Y_shuffled[:,i*mini_batch_size: (i+1)*mini_batch_size]	
		mini_batches.append((mini_batch_X, mini_batch_Y))
	
	if m % mini_batch_size != 0:
		last_batch_size = m - mini_batch_size*mini_batch_count
		mini_batch_X = X_shuffled[:,-last_batch_size:]
		mini_batch_Y = Y_shuffled[:,-last_batch_size:]
		mini_batches.append((mini_batch_X, mini_batch_Y))
		
	return mini_batches	

# Momentum optimization
def initialize_velocity(parameters):
	v = {}
	for i in range(len(parameters)//2):
		v['dW%i'%(i+1)] = np.zeros_like(parameters['W%i'%(i+1)])
		v['db%i'%(i+1)] = np.zeros_like(parameters['b%i'%(i+1)])
	return v

def update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate)
	# larger beta takes more past gradients into account
	# usually 0.8 < beta < 0.999
	for i in range(len(parameters)//2):
		W_cur = parameters['W%i'%(i+1)]
		b_cur = parameters['b%i'%(i+1)]
		
		dW_cur = grads['dW%i'%(i+1)]
		db_cur = grads['db%i'%(i+1)]
		
		# if beta = 0, it becomes standard gradient descent
		v['dW%i'%(i+1)] = beta*v['dW%i'%(i+1)] + (1-beta)*dW_cur
		v['db%i'%(i+1)] = beta*v['db%i'%(i+1)] + (1-beta)*db_cur
		
		parameters['W%i'%(i+1)] = W_cur - learning_rate*v['dW%i'%(i+1)]
		parameters['b%i'%(i+1)] = b_cur - learning_rate*v['db%i'%(i+1)]
	return parameters, v
	
def initialize_adam(parameters):
	v = {}
	s = {}
	
	for i in range(len(parameters)//2):
		v['dW%i'%(i+1)] = np.zeros_like(parameters['W%i'%(i+1)])
		v['db%i'%(i+1)] = np.zeros_like(parameters['b%i'%(i+1)])
		s['dW%i'%(i+1)] = np.zeros_like(parameters['W%i'%(i+1)])
		s['db%i'%(i+1)] = np.zeros_like(parameters['b%i'%(i+1)])
	return v, s
	
# Adam optimization
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
	correction_factor1 = 1-np.power(beta1,t)
	correction_factor2 = 1-np.power(beta2,t)
	
	for i in range(len(parameters)//2):
		
		v['dW'%(i+1)] = beta1*v['dW'%(i+1)] + (1-beta1)*grads['dW%i'%(i+1)]
		v['db'%(i+1)] = beta1*v['db'%(i+1)] + (1-beta1)*grads['db%i'%(i+1)]
		
		s['dW'%(i+1)] = beta2*s['dW'%(i+1)] + (1-beta2)*np.square(grads['dW%i'%(i+1)])
		s['db'%(i+1)] = beta2*s['db'%(i+1)] + (1-beta2)*np.square(grads['db%i'%(i+1)])
		
		vdW_corrected = v['dW'%(i+1)]/correction_factor1
		vdb_corrected = v['db'%(i+1)]/correction_factor1
		
		sdW_corrected = s['dW'%(i+1)]/correction_factor2
		sdb_corrected = s['db'%(i+1)]/correction_factor2
		
		parameters['W%i'%(i+1)] -= learning_rate*vdW_corrected/(np.sqrt(sdW_corrected) + epsilon)
		parameters['b%i'%(i+1)] -= learning_rate*vdb_corrected/(np.sqrt(sdb_corrected) + epsilon)
	return parameters, v, s
	
def model(X, Y, layer_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
		  beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000, print_cost = True):
	
	costs = []
	t = 0 
	seed = 10
	parameters = initialize_parameters(layer_dims)
	
	if optimizer = 'gd':
		pass 
	elif optimizer = 'momentum':
		v = initialize_velocity(parameters)
	elif optimizer = 'adam':
		v, s = initialize_adam(parameters)
		
	for i in range(num_epochs):
		seed = seed + 1
		minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
		for minibatch in minibatches:
			minibatch_X, minibatch_Y = minibatch
			
			a3, caches = forward_propagation(minibatch_X, parameters)
			cost = compute_cost(a3, minibatch_Y)
			grads = backward_propagation(minibatch_X, minibatch_Y, caches)
			
			if optimizer = 'gd':
				parameters = update_parameters_with_gd(parameters, grads, learning_rate)
			elif optimizer = 'momentum':
				parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
			elif optimizer = 'adam':
				t = t + 1
				parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
		if print_cost and i % 1000 == 0:
			print('Cost after epoch%i: %.10f'%(i,cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)
			
	plt.plot('cost')
	plt.ylabel('epochs (per 100)')
	plt.title('Learning rate = %.7f'%learning_rate)
	plt.show()
	
	return parameters

