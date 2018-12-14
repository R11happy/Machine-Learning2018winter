import numpy as np

def error_compute_wf(w_g, X, w_f):
    '''
    计算错误分类的个数
    X  : 样本 P-by-N matrix.
    w_g: 学习到的参数 (P+1)-by-1 column vector
    w_f: 真正的参数   (P+1)-by-1 column vector
    '''
    N = X.shape[1]
    X_b = np.vstack((np.ones((1, X.shape[1])), X))
    num_error = 0
    for i in range(N):
        if np.sign(np.matmul(w_g.T, X_b[:,i])) != np.sign(np.matmul(w_f.T, X_b[:,i])):
            num_error += 1
    return num_error