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

    x_O = x.dot(theta)
    h_O = sig(x_O)
    tmp = h_O - y
    g = 1. / N * tmp.dot(x)

    # END TODO
    ###########

    return g
