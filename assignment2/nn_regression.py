import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha,plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    mse = np.sum(np.power(nn.predict(x)-y, 2)) / x.shape[0]

    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    regressor = MLPRegressor(
    hidden_layer_sizes=(2,),#8,40
    solver="lbfgs",
    activation="logistic",
    alpha=0.0,
    max_iter=200,
    )
    regressor.fit(x_train, y_train)
    plot_learned_function(2,x_train, y_train,regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
    #plot_learned_function(8, x_train, y_train, regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
    #plot_learned_function(40, x_train, y_train, regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
    pass

def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    mse_trains = []
    mse_tests = []
    for index in range(0,10):
        regressor = MLPRegressor(
            hidden_layer_sizes=(2,),  # 8,40
            solver="lbfgs",
            activation="logistic",
            alpha=0.0,
            max_iter=200,
            random_state=index
        )

        regressor.fit(x_train, y_train)
        mse_trains.append(calculate_mse(regressor, x_train, y_train))
        mse_tests.append(calculate_mse(regressor, x_test, y_test))

    print("min_train:", np.min(mse_trains))
    print("min_test:", np.min(mse_tests))

    print("max_train:", np.max(mse_trains))
    print("max_text:", np.max(mse_tests))

    print("mean_train:", np.mean(mse_trains))
    print("mean_test:", np.mean(mse_tests))

    print("std_train:", np.std(mse_trains))
    print("std_test:", np.std(mse_tests))

    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    pass




def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    pass

def ex_1_2_c(x_train, x_test, y_train, y_test):
    '''
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    ## TODO
    pass