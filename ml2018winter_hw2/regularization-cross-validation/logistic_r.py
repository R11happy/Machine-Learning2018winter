import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    lr = 0.01  # l
    maxIter = 100
    epsilon = 0.006
    # YOUR CODE HERE
    # begin answer
    for iter in range(maxIter):
        J = 0
        for i in range(N):
            ywx = np.dot(y[0, i]*w.T, X_b[:, i])
            J = J + np.log(1 + np.exp(-ywx)) + lmbda*np.sqrt(np.dot(w.T, w))
            grad = -1/(1+np.exp(ywx)) * y[0, i] * X_b[:, i] + lmbda
            w = w - lr*grad.reshape((P+1, 1))
        if (J/N) < epsilon:
            break
#         print("loss:{}".format(J))
    # end answer
    return w