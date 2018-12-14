import numpy as np
def error_compute_y(w_g, X, y):
    '''
    计算预测结果与label之间的错误个数
    X : Sample, P by N
    y : label   1 by N
    w_g: 学习到的参数 P+1 by 1
    '''
    N = X.shape[1]
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    num_error = 0
    for i in range(N):
        if np.sign(np.dot(w_g.T, X_b[:,i])) != y[0, i]:
                   num_error += 1
    return num_error