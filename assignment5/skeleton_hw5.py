#!/usr/bin/env python3
#Filename skeleton_HW5.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 01.6.2017
#Edited: 02.6.2017 -- naming conventions, comments, ...

import numpy as np
import numpy.random as rd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
from math import pi, exp
from scipy.stats import multivariate_normal


## -------------------------------------------------------    
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.
       
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """

    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)
    
    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[
    
    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1	
        y[i] = X[j]
        
    return rd.permutation(y) # permutation of all samples


def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title):
    """Show contour plot for bivariate Gaussian with given mu and cov in the range specified.

    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01], [cov_10, cov_11]]
    xmin, xmax, ymin, ymax ... range for plotting
    """
    
    npts = 500
    deltaX = (xmax - xmin) / npts
    deltaY = (ymax - ymin) / npts
    stdev = [0, 0]

    stdev[0] = np.sqrt(cov[0][0])
    stdev[1] = np.sqrt(cov[1][1])
    x = np.arange(xmin, xmax, deltaX)
    y = np.arange(ymin, ymax, deltaY)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(X, Y, stdev[0], stdev[1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline = 1, fontsize = 10)
    plt.title(title)
    #plt.show()


def likelihood_bivariate_normal(X, mu, cov):
    """Returns the likelihood of X for bivariate Gaussian specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01], ..., [x_n0, x_n1]])
    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01],[cov_10, cov_11]]
    """
    
    dist = multivariate_normal(mu, cov)
    P = dist.pdf(X)
    return P

def calc_all_r_m(X, M, alpha, mu, Sigma):
    r_m = 0.
    for i in range(M):
        r_m += alpha[i] * likelihood_bivariate_normal(X, mu[i], Sigma[i])

    return r_m

def calc_r_n_m(X, M, alpha_0, mu_0, Sigma_0):
    r_n_m = []
    for i in range(M):
        r_n_m.append((alpha_0[i] * likelihood_bivariate_normal(X, mu_0[i], Sigma_0[i])) / calc_all_r_m(X, M, alpha_0, mu_0, Sigma_0))
    return r_n_m

def initValues(X, M):

    mu_init = np.zeros((M, 2))
    alpha_init = np.zeros((M, 1))
    sigma_init = []
    x_shape = X.shape
    for i in range(M):
        N = 1000
        mu_0 = np.zeros((2, N))
        for j in range(N):
            random_row = X[int(rd.uniform(0, x_shape[0]))]
            mu_0[0][j] = random_row[0]
            mu_0[1][j] = random_row[1]

        mu_init[i] = np.mean(mu_0, axis=1)
        alpha_init[i][0] = 1/M
        cov = np.cov(mu_0).tolist()
        #cov[0][1] = 0.
        #cov[1][0] = 0.
        sigma_init.append(cov)


    return (alpha_init, mu_init, sigma_init)

def checkIfLikelihoodConverges(X, M, alpha, mu, Sigma, lastValue):
    r_ms = calc_all_r_m(X, M, alpha, mu, Sigma)
    log_value = int(np.sum(np.log(r_ms)))
    if abs(log_value - lastValue) < 1:
        return True, log_value
    else:
        return False, log_value

## -------------------------------------------------------    
## ------------- START OF  ASSIGNMENT 5 ------------------
## -------------------------------------------------------


def EM(X, M, alpha_0, mu_0, Sigma_0, max_iter):
    # TODO

    alpha = alpha_0
    mu = mu_0
    Sigma = Sigma_0
    log_value = -999999
    logs = []
    r_n_m = calc_r_n_m(X, M, alpha, mu, Sigma)
    for i in range(max_iter):
        N_ms = []

        print("iter: ", i)
        r_n_m = calc_r_n_m(X, M, alpha, mu, Sigma)

        for index, value in enumerate(r_n_m):
            N_ms.append(np.sum(value))

        #update values
        for j in range(M):

            #update alpha
            alpha[j] = N_ms[j] / X.shape[0]
            #update mu
            mu[j] = (1/ N_ms[j]) * np.dot(np.array(r_n_m[j]),np.array(X))
            #update Sigma
            tmpX = np.array(X)
            tmpX[:, 0] -= mu[j][0]
            tmpX[:, 1] -= mu[j][1]
            tmpX = tmpX.reshape((len(X), 2))
            X_r_n_m = np.array(r_n_m[j]).reshape(X.shape[0], 1) * tmpX
            tmp_Sigma = np.dot(X_r_n_m.T, tmpX)
            #tmp_Sigma = np.zeros((2,2))
            #for row in range(len(X)):
            #    tmp_Sigma += (r_n_m[j][row] * np.dot((X[row].reshape(2, 1) - mu[j].reshape((2,1))),(X[row].reshape(2, 1) - mu[j].reshape((2,1))).T))
            #tmp_Sigma[1][0] = 0.
            #tmp_Sigma[0][1] = 0.
            Sigma[j] = (1 / N_ms[j]) * tmp_Sigma

        converged, log_new = checkIfLikelihoodConverges(X, M, alpha, mu, Sigma, log_value)

        if converged:
            break
        else:
            log_value = log_new
            logs.append(log_value)

    plt.scatter(X[:, 0], X[:, 1], s=3)
    for i,value in enumerate(mu):
        print("mu", i ,":", value )
        plot_gauss_contour(value, Sigma[i], 0, 1000, 0, 3000, "Gauss")
    plt.show()
    plt.plot(np.arange(len(logs)), logs, '-')
    plt.xlabel("iteration")
    plt.ylabel("log- Likelihood")
    plt.show()
    softClassification = [[] for i in range(M)]
    for index, row in enumerate(np.array(r_n_m).T):
        max = np.argmax(row)
        softClassification[max].append(X[index])
    for sC in softClassification:
        s_C = np.array(sC)
        plt.scatter(s_C[:, 0], s_C[:, 1], c=np.random.rand(3,), s=3)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def k_means(X, M, mu_0, max_iter):
    mus = mu_0
    e = 0.001
    old_distance = -99999
    Y_Ks = [[] for i in range(M)]
    for iter in range(max_iter):
        print("iter: ", iter)
        Y_Ks = [[] for i in range(M)]
        distances = np.zeros((X.shape[0], M))
        #classification
        for i, mu in enumerate(mus):
            Dist = (X - mu)
            for j,vector in enumerate(Dist):
                distances[j][i] = np.dot(vector, vector.T)

        for index, row in enumerate(distances):
            min = np.argmin(row)
            Y_Ks[min].append(X[index])

        #recalculate clusterpoints
        mus = [np.mean(Y_K, axis=0) for Y_K in Y_Ks]

        distances = np.zeros((X.shape[0], M))
        for i, mu in enumerate(mus):
            Dist = (X - mu)
            for j, vector in enumerate(Dist):
                distances[j][i] = np.dot(vector, vector.T)

        distance = 0.
        for index, row in enumerate(distances):
            distance += row[min]

        if abs(old_distance - distance) < e:
            break
        else:
            old_distance = distance


    print(mus)
    plt.scatter(np.array(mus)[:, 0], np.array(mus)[:, 1])
    for Y_K in Y_Ks:
        Y_K = np.array(Y_K)
        plt.scatter(Y_K[:, 0], Y_K[:, 1], c=np.random.rand(3,), s=1)
    plt.scatter(np.array(mus)[:, 0], np.array(mus)[:, 1], c=(0, 0, 0))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def sample_GMM(alpha, mu, Sigma, N):
    # TODO
    pass


def main():
    # load data
    X = np.loadtxt('data/X.data', skiprows = 0) # unlabeled data
    a = np.loadtxt('data/a.data', skiprows = 0) # label: a
    e = np.loadtxt('data/e.data', skiprows = 0) # label: e
    i = np.loadtxt('data/i.data', skiprows = 0) # label: i
    o = np.loadtxt('data/o.data', skiprows = 0) # label: o
    y = np.loadtxt('data/y.data', skiprows = 0) # label: y

    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(e[:, 0], e[:, 1])
    plt.scatter(i[:, 0], i[:, 1])
    plt.scatter(o[:, 0], o[:, 1])
    plt.scatter(y[:, 0], y[:, 1])
    plt.show()

    # 1.) EM algorithm for GMM:
    # TODO	
    M = 5
    max_iter = 100
    (alpha_init, mu_init, sigma_init) = initValues(X, M)
    EM(X, M, alpha_0=alpha_init, mu_0=mu_init, Sigma_0=sigma_init, max_iter=max_iter)


    # 2.) K-means algorithm:
    # TODO
    k_means(X, M, mu_init, 100)

    # 3.) Sampling from GMM
    # TODO

    pass


def sanity_checks():
    # likelihood_bivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2], [0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_bivariate_normal(x, mu, cov)
    print(P)

    # plot_gauss_contour(mu, cov, -2, 2, -2, 2, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))


if __name__ == '__main__':
    # to make experiments replicable (you can change this, if you like)
    #rd.seed(23434345)
    rd.seed(int(np.random.uniform(0, 100000000)))
    sanity_checks()
    main()
    
