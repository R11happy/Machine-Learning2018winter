import numpy as np
def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    # P : 特征数量
    # N： 样本数量
    P, N = X.shape
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    w = np.zeros((P + 1, 1))
    eta = 1
    iters = 0

    # YOUR CODE HERE
    
    # begin answer
    while(True):
        J = 0
        for i in range(N):
            y_hat = np.sign(np.matmul(w.T, X_b[:, i]))
            if y_hat * y[0, i] <= 0:
                iters += 1
                grad = eta*(y[0, i]*X_b[:, i])
                w = w + grad.reshape((P+1,1))
                dist = w.T * X_b[:, i]
                dist2 = np.matmul(w.T, X_b[:, i])
                J = J + np.sum((-y[0, i]) * (w.T * X_b[:, i]))
        if J == 0:
            break
    # end answer
    return w, iters