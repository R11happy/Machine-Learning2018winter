import numpy as np
'''
y -> {-1, 1}
'''
def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''

    # YOUR CODE HERE
    # begin answer
    def sigmoid(t):
        return 1. / (1 + np.exp(-t))

    def J(w, X_b, y):
        ywx = y * (w.T.dot(X_b))  # 1 by N
        N = len(y)
        try:
            return (1. /N) * np.sum(np.log(1 + np.exp(-ywx))) + (1. /(2*N)) * lmbda * np.sum(w[1:,].T.dot(w[1:,]))
        except:
            return float('inf')

    def dJ(w, X_b, y):  # P+1 -by- 1
        ywx = y * (w.T.dot(X_b))  # 1 by N
        N = len(y)
        grad = (1./N) * (-y * X_b).dot(sigmoid(-ywx.T))
        grad[1:,] = grad[1:,] + (1./N) * lmbda *w[1:,] # do not regularize w0
        return grad

    def gradient_descent(X_b, y, initial_w, eta, n_iters=1e4, epsilon=1e-8):

        w = initial_w
        cur_iter = 0
        loss = []
        loss.append(J(w, X_b, y))
        # print('loss:{}, iter:{}'.format(loss[cur_iter], cur_iter))
        while cur_iter < n_iters:
            cur_iter += 1
            gradient = dJ(w, X_b, y)

            w = w - eta * gradient
            cur_loss = J(w, X_b, y)
            loss.append(cur_loss)

            if (abs(loss[cur_iter] - loss[cur_iter - 1]) < epsilon):
                break
            # print('loss:{}, iter:{}'.format(loss[cur_iter], cur_iter))
        return w

    P, N = X.shape
    initial_w = np.zeros((P+1, 1))
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    eta = 0.2  # l
    n_iters = 1e7
    epsilon = 1e-9
    w = gradient_descent(X_b, y, initial_w, eta, n_iters, epsilon)
    return w