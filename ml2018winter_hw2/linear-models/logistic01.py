import numpy as np
'''
y -> {0, 1}
'''

def logistic01(X, y):
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
        # y_hat = sigmoid(w.T.dot(X_b)) # 1-by-N
        try:
            # invalid value encountered in multily
            # return -np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            return -np.sum(y * (w.T.dot(X_b)) - np.log(1 + np.exp(w.T.dot(X_b))))
        except:
            return float('inf')

    def dJ(w, X_b, y): # P+1 -by- 1
        y_hat = sigmoid(w.T.dot(X_b))
        return X_b.dot((y_hat - y).T) / len(y)

    def gradient_descent(X_b, y, initial_w, eta, n_iters=1e4, epsilon=1e-8):

        w = initial_w
        cur_iter = 0
        loss = []
        loss.append(J(w, X_b, y))
        print('loss:{}, iter:{}'.format(loss[cur_iter], cur_iter))
        while cur_iter < n_iters:
            cur_iter += 1
            gradient = dJ(w, X_b, y)

            w = w - eta * gradient
            cur_loss = J(w, X_b, y)
            loss.append(cur_loss)

            if(abs(loss[cur_iter] - loss[cur_iter-1]) < epsilon):
                break
            print('loss:{}, iter:{}'.format(loss[cur_iter], cur_iter))
        return w

    P, N = X.shape
    initial_w = np.zeros((P+1, 1))
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    eta = 0.05  # l
    n_iters = 1e5
    epsilon = 1e-5
    w = gradient_descent(X_b, y, initial_w, eta, n_iters, epsilon)
    return w