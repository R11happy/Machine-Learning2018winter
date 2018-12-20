import numpy as np
from scipy.optimize import minimize
def svm(X, y, C=1):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.
            C: > 0 , penalty for the mistake 

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    num = 0
    epsilon = 0.0001

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer
    def f(w):
        res = y * np.dot(w.T, X_b)
        return .5 * (1./C) * sum(w * w) +  sum(1 - res[res < 1])
    w = minimize(f, w).x
    
    # support vector
    res = y * np.dot(w.T, X_b)
    num = len(res[(res - 1.0) <= epsilon])
    # end answer
    return w, num

