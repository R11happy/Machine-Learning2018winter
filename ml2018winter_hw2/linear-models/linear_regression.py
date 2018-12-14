import numpy as np

from numpy.linalg import inv
def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    # YOUR CODE HERE
    # begin answer
    w = np.dot(np.dot((inv(np.dot(X_b ,X_b.T))), X_b), y.T)
    # end answer
    return w
