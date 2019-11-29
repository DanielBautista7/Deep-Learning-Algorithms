	
def backward_propagation_with_regularization(X, Y, cache, lambd):
	m = X.shape[1]
	(Z1, A1, W1, b1, Z2, A2, b2, Z3, A3, W3, b3) = cache
	dZ3 = A3 - Y
	dW3 = np.matmul(dZ3, A2.T)/m + lambd*W3/m
	db3 = np.sum(dZ3,axis = 1,keepdims = True)
	dA2 = np.matmul(W3.T,dZ3)
	#dZ2 = np.multiply(dA2,np.maximum(np.zeros_like(A2),(A2/np.abs(A2))))
	dZ2 = np.multiply(dA2, np.int64(A2>=0))
	dW2 = np.matmul(dZ2,A1.T)/m + lambd*W2/m
	db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
	dA1 = np.matmul(W2.T,dZ2)
	dZ1 = np.multiply(dA1, np.int64(A1>=0))
	dW1 = np.matmul(dZ1, X.T)/m + lambd*W1/m
	db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
	
	gradients = {'dZ3': dZ3, 'dW3': dW3, 'db3': db3, 'dA2': dA2,
				 'dZ2': dZ2, 'dW2': dW2, 'db2': db2, 'dA1': dA1,
				 'dZ1': dZ1, 'dW1': dW1, 'db1': db1}
				 
	return gradients
	
def relu(X):
	return np.maximum(np.zeros_like(X),X)
	
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
	
	Z1 = np.matmul(parameters['W1'],X)+parameters['b1']
	A1 = relu(Z1)
	D1 = np.random.randn(A1.shape) < keep_prob
	A1 = np.multiply(A1, np.int64(D1))/keep_prob
	
	cache_dict = {'Z1':Z1, 'A1':A1, 'D1':D1}
	
	layer_count = len(parameters)//2 
	for i in range(2,layer_count):
		W_cur = parameters['W%i'i]
		b_cur = parameters['b%i'i]
		A_prev = cache_dict['A%i'%(i-1)]
		Z_cur = np.matmul(W_cur,A_prev)+b_cur
		cache_dict['Z%'%i] = Z_cur
		A_cur_ = relu(Z_cur)
		D_cur = np.int64(np.random.randn(A_cur_.shape)<keep_prob)
		A_cur = np.multiply(A_cur_, D_cur)
		cache_dict['A%'%i] = A_cur
		cache_dict['D%i'%i] = D_cur
		
	cache = (cache_dict['Z1'], cache_dict['D1'], cache_dict['A1'], cache_dict['W1'], cache_dict['b1'],
			 cache_dict['Z2'], cache_dict['D2'], cache_dict['A2'], cache_dict['W2'], cache_dict['b2'],
			 cache_dict['Z3'], cache_dict['D3'], cache_dict['A3'], cache_dict['W3'], cache_dict['b3'])
			 
	return cache_dict['A3'], cache
	
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
	m = X.shape[1]
	(Z1, D1, A1, W1, b1,
	 Z2, D2, A2, W2, b2,
	 Z3, D3, A3, W3, b3) = cache
	 dZ3 = A3 - Y
	 dW3 = np.matmul(dZ3, A2.T)/m
	 db3 = np.sum(dZ3, axis = 1, keepdims = True)/m
	 dA2 = np.matmul(W3.T, dZ3)
	 dA2 = np.multiply(dA2,D2)/keep_prob
	 dZ2 = np.multiply(dA2, np.int64(A2>0))
	 dW2 = np.matmul(dZ2, A1.T)/m
	 db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
	 dA1 = np.matmul(W2.T, dZ2)
	 dA1 = np.multiply(dA1, D1)/keep_prob
	 dZ1 = np.multiply(dA1, np.int64(dA1 > 0))
	 dW1 = np.matmul(dZ1, X.T)/m
	 db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
	 
	 gradients = {'dZ1:' dZ1, 'dA1': dA1, 'dW1': dW1, 'db1': db1,
			'dZ2:' dZ2, 'dA2': dA2, 'dW2': dW2, 'db2': db2,
			'dZ3:' dZ3, 'dW3': dW3, 'db3': db3}
				  
	return gradients

