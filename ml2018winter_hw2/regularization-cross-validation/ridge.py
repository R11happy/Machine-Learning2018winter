import numpy as np

from numpy.linalg import pinv
def ridge(X, y, lmbda):
    '''
    RIDGE Ridge Regression.

      INPUT:  X: training sample features, P-by-N matrix.
              y: training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w: learned parameters, (P+1)-by-1 column vector.

    NOTE: You can use pinv() if the matrix is singular.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    w = np.dot(np.dot((pinv(np.dot(X_b ,X_b.T) + lmbda * np.eye((P+1)))), X_b), y.T)
    # end answer
    return w
