# Filename: HW4_skeleton.py
# Author: Florian Kaum
# Edited: 15.5.2017
# Edited: 19.5.2017 -- changed evth to HW4

import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import sys
from scipy.stats import multivariate_normal


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming
def plotGaussContour(mu, cov, xmin, xmax, ymin, ymax, title):
    npts = 100
    delta = 0.025
    stdev = np.sqrt(cov)  # make sure that stdev is positive definite

    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    # noinspection PyPep8Naming,PyPep8Naming
    X, Y = np.meshgrid(x, y)

    # matplotlib.mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0) -> use cov directly
    # noinspection PyPep8Naming
    Z = mlab.bivariate_normal(X, Y, stdev[0][0], stdev[1][1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+')  # plot the mean as a single point
    # noinspection PyPep8Naming
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title(title)
    plt.show()
    return


# noinspection PyPep8Naming
def ecdf(realizations):
    x = np.sort(realizations)
    # noinspection PyPep8Naming
    Fx = np.linspace(0, 1, len(realizations))
    return Fx, x

def d(para, anchor):
    return np.linalg.norm(anchor - para)

def calcVarianz(ri, point, anchors, nrAnchor, nrSample):
    sigmas = []
    for i in range(nrAnchor):
        sigma_i = 0
        for j in range(nrSample):
            sigma_i += np.power(ri[j, i] - d(point, anchors[i]), 2)
        sigma_i /= nrSample
        sigmas.append(sigma_i)
    return sigmas

def calcLambda(ri, point, anchors, nrAnchor, nrSample):
    lambdas = []
    for i in range(nrAnchor):
        lambda_i = 0
        for j in range(nrSample):
            if ri[j, i] >= d(point, anchors[i]):
                lambda_i += (ri[j, i] - d(point, anchors[i])) # replace 1 with nrSample
        lambda_i = nrSample / lambda_i
        lambdas.append(lambda_i)
    return lambdas

def derivationOfPoint(x_i, y_i, x, y):
    derivationInX = 1 / 2 * np.power((np.power((x_i - x), 2) + np.power((y_i - y), 2)), - 1 / 2) * 2 * (x_i - x)
    derivationInY = 1 / 2 * np.power((np.power((x_i - x), 2) + np.power((y_i - y), 2)), - 1 / 2) * 2 * (y_i - y)

    return (derivationInX, derivationInY)

def jacobian(p_start, p_anchor):
    jacobian_mat = np.ndarray((p_anchor.shape[0], p_start.shape[0]))
    for i, anchor in enumerate(p_anchor):
        derivation = derivationOfPoint(anchor[0], anchor[1], p_start[0], p_start[1])
        jacobian_mat[i, 0] = derivation[0]
        jacobian_mat[i, 1] = derivation[1]

    return jacobian_mat

def ds(p_start, p_anchor):
    distances = np.ndarray((p_anchor.shape[0],))
    for i, anchor in enumerate(p_anchor):
        distances[i] = d(p_start, anchor)

    return distances


def LeastSquaresGN(p_anchor, p_start, r, max_iter, tol):

    while True :
        max_iter -= 1
        p_old = p_start
        jacobian_inv = np.dot(np.linalg.inv(np.dot(jacobian(p_start, p_anchor).T, jacobian(p_start, p_anchor))), jacobian(p_start, p_anchor).T)
        diffs = (r - ds(p_start, p_anchor))
        p_start = p_start - np.dot(jacobian_inv, diffs)
        if d(p_start, p_old) < tol or max_iter <= 0:
            break
    return p_start

def getMinValues(p_anchor):
    minX = 999999
    minY = 999999
    for anchor in p_anchor:
        if minX > anchor[0]:
            minX = anchor[0]
        if minY > anchor[1]:
            minY = anchor[1]
    return (minX, minY)

def getMaxValues(p_anchor):
    maxX = -999999
    maxY = -999999
    for anchor in p_anchor:
        if maxX < anchor[0]:
            maxX = anchor[0]
        if maxY < anchor[1]:
            maxY = anchor[1]
    return (maxX, maxY)

# START OF CI ASSIGNMENT 4
# -----------------------------------------------------------------------------------------------------------------------

# positions of anchors
p_anchor = np.array([[5, 5], [-5, 5], [-5, -5], [5, -5]])
NrAnchors = np.size(p_anchor, 0)

# true position of agent
p_true = np.array([[2, -4]])

# plot anchors and true position
plt.axis([-6, 6, -6, 6])
for i in range(0, NrAnchors):
    plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
    plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
plt.xlabel("x/m")
plt.ylabel("y/m")
# plt.show()

# 1.2) maximum likelihood estimation of models---------------------------------------------------------------------------
# 1.2.1) finding the exponential anchor----------------------------------------------------------------------------------
# TODO
# insert plots

# 1.2.3) estimating the parameters for all scenarios---------------------------------------------------------------------

# scenario 1
data = np.loadtxt('HW4_1.data', skiprows=0)
NrSamples = np.size(data, 0)
# TODO
print("varianzen:   ", calcVarianz(data, p_true, p_anchor, NrAnchors, NrSamples))
print("Lambdas:     ", calcLambda(data, p_true, p_anchor, NrAnchors, NrSamples))
# scenario 2
data = np.loadtxt('HW4_2.data', skiprows=0)
NrSamples = np.size(data, 0)
# TODO

#for i in range(4):
#    plt.clf()
#    plt.hist(data.transpose()[i])
#    plt._show()


print("varianzen:   ", calcVarianz(data, p_true, p_anchor, NrAnchors, NrSamples))
print("Lambdas:     ", calcLambda(data, p_true, p_anchor, NrAnchors, NrSamples))
# scenario 3
data = np.loadtxt('HW4_3.data', skiprows=0)
NrSamples = np.size(data, 0)
# TODO
print("varianzen:   ", calcVarianz(data, p_true, p_anchor, NrAnchors, NrSamples))
print("Lambdas:     ", calcLambda(data, p_true, p_anchor, NrAnchors, NrSamples))
# 1.3) Least-Squares Estimation of the Position--------------------------------------------------------------------------
# 1.3.2) writing the function LeastSquaresGN()...(not here but in this file)---------------------------------------------
# TODO

# 1.3.3) evaluating the position estimation for all scenarios------------------------------------------------------------

# choose parameters
# tol = ... # tolerance
# maxIter = ...  # maximum number of iterations

# store all N estimated positions
p_estimated = np.zeros((NrSamples, 2))
minValues = getMinValues(p_anchor)
maxValues = getMaxValues(p_anchor)
for scenario in range(1, 5):
    if (scenario == 1):
        data = np.loadtxt('HW4_1.data', skiprows=0)
    elif (scenario == 2):
        data = np.loadtxt('HW4_2.data', skiprows=0)
    elif (scenario == 3):
        data = np.loadtxt('HW4_3.data', skiprows=0)
    elif (scenario == 4):
        # scenario 2 without the exponential anchor
        data = np.loadtxt('HW4_2.data', skiprows=0, usecols=range(1, 4))
        p_anchor = p_anchor[1:4, ]
    NrSamples = np.size(data, 0)

    # perform estimation---------------------------------------
    # #TODO
    plt.clf()
    for i in range(0, NrSamples):
        if (i % 50 == 0):
            print("sample #:", i)
        p_start = np.array([np.random.uniform(minValues[0], maxValues[0]), np.random.uniform(minValues[1], maxValues[1])])
        p_start = LeastSquaresGN(p_anchor, p_start, data[i], 50, 0.005)
        p_estimated[i][0] = p_start[0]
        p_estimated[i][1] = p_start[1]
        plt.plot(p_start[0], p_start[1], "y.")


    # calculate error measures and create plots----------------
    # TODO
    mu = np.zeros((2,1))
    cov = np.zeros((2,2))

    mu[0] = np.mean(p_estimated.transpose()[0, :])
    mu[1] = np.mean(p_estimated.transpose()[1, :])

    cov = np.zeros((2,2))
    for estimate in p_estimated:
        e = np.array([[0.],[0.]])
        e[0] = estimate[0] - mu[0]
        e[1] = estimate[1] - mu[1]
        cov += np.dot(e,e.transpose())
    cov /= NrSamples
    #print(sum)

    errors = []
    for estimate in p_estimated:
        errors.append(d(p_true, estimate))

    Fx, x = ecdf(errors)
    plt.clf()
    plt.plot(x, Fx)
    plt.show()

    plotGaussContour(mu=mu, cov=cov, xmin=-10, xmax=10, ymin=-10, ymax=10, title="Gauss - Contour")



# 1.4) Numerical Maximum-Likelihood Estimation of the Position (scenario 3)----------------------------------------------
# 1.4.1) calculating the joint likelihood for the first measurement------------------------------------------------------
# TODO

# 1.4.2) ML-Estimator----------------------------------------------------------------------------------------------------

# perform estimation---------------------------------------
# TODO

# calculate error measures and create plots----------------
# TODO

# 1.4.3) Bayesian Estimator----------------------------------------------------------------------------------------------

# perform estimation---------------------------------------
# TODO

# calculate error measures and create plots----------------
# TODO
