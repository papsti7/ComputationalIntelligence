#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #

    x_O = x.dot(theta)
    h_O = sig(x_O)
    c = - np.sum(np.dot(y, np.log(h_O)) + np.dot((1 - y), np.log(1 - h_O))) / N


    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    g = np.zeros(theta.shape)

    for row in range(g.shape[0]):
        sum = 0
        for feature in range(N):
            x_O = x[feature].dot(theta)
            h_O = sig(x_O)
            sum += (h_O - y[feature]) * x[feature][row]
        g[row] = sum / float(N)

    # END TODO
    ###########

    return g
