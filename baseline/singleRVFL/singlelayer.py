

import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import KFold


def sigmoid(a, b, x):

    return 1.0 / (1 + np.exp(-1.0 * (x.dot(a) + b)))


def SingleLayer(X, Y, L, C):
    """
    Variables
    X - input : No. of samples * No. of feature 
    Y - Target : No. of samples * No. of output features
    L - Number of nodes in hidden layer 
    C - Hyper-parameter of the regularization

     H - H matrix : No. of samples * No. of hidden nodes (N*L)
    """
    # Nxd matrix
    sx = X.shape
    Nx = sx[0]
    fx = sx[1]

    sy = Y.shape
    Ny = sy[0]
    fy = sy[1]

    assert(Nx==Ny)
    
    H1 = X 

    # init the weight of hidden layers randomly
    a = np.random.normal(0, 1, (fx, L))
    b = np.random.normal(0, 1)
    
    # Why ? 
    #Y = one_hot(Y)

    H2 = sigmoid(a, b, X)
    
    H = np.concatenate((H1, H2), axis=1)

    HTH = H.transpose().dot(H)
    HTY = H.transpose().dot(Y)
    HHT = H.dot(H.transpose())
    I = np.identity(fx+L)/C_list

    # calculate the weight of output layers(beta) and output
    if (fx+L) <= Nx:
        beta = np.linalg.pinv(HTH + I).dot(HTY)
    else:
        beta = H.transpose().dot(np.linalg.pinv(HHT + I)).dot(Y)

    return beta, a, b


def one_hot(l):
    y = np.zeros([len(l), np.max(l)+1])
    for i in range(len(l)):
        y[i, l[i]] = 1
    return y


def predict(X, BETA, a, b):
    H = sigmoid(a, b, X)
    D = np.concatenate((X, H), axis=1)
    Y = D.dot(BETA)
    Y = Y.argmax(1)
    return Y


def evaluation(y_hat, goundtruth):
    y_hat = y_hat[:, np.newaxis]
    return np.sum(np.equal(y_hat, goundtruth) / len(y_hat))


