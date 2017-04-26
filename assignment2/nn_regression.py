import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt
from random import randint

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
    hidden_layer_sizes=(40,),#8,40
    solver="lbfgs",
    activation="logistic",
    alpha=0.0,
    max_iter=200,
    )
    regressor.fit(x_train, y_train)
    #plot_learned_function(2,x_train, y_train,regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
    #plot_learned_function(8, x_train, y_train, regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
    plot_learned_function(40, x_train, y_train, regressor.predict(x_train), x_test, y_test, regressor.predict(x_test))
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
            hidden_layer_sizes=(8,),
            solver="lbfgs",
            activation="logistic",
            alpha=0.0,
            max_iter=200,
            random_state=randint(1,1000)
        )

        regressor.fit(x_train, y_train)
        mse_trains.append(calculate_mse(regressor, x_train, y_train))
        mse_tests.append(calculate_mse(regressor, x_test, y_test))

    print("min_train:", np.min(mse_trains))
    print("min_test:", np.min(mse_tests))

    print("max_train:", np.max(mse_trains))
    print("max_test:", np.max(mse_tests))

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
    neuron_numbers = [1, 2, 3, 4, 6, 8, 12, 20, 40]
    mses_test = np.zeros((9,10))
    mses_train = np.zeros((9,10))
    for n in neuron_numbers:
        for i in range(0,10):
            random_seed = randint(1,1000)
            regressor = MLPRegressor(
                hidden_layer_sizes=(n,),
                solver="lbfgs",
                activation="logistic",
                alpha=0.0,
                max_iter=200,
                random_state=random_seed
            )
            regressor.fit(x_train, y_train)
            mses_train[neuron_numbers.index(n)][i] = calculate_mse(regressor, x_train, y_train)
            mses_test[neuron_numbers.index(n)][i] = calculate_mse(regressor, x_test, y_test)

    plot_mse_vs_neurons(mses_train, mses_test, neuron_numbers)

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
    neuron_numbers = [2, 8, 20]
    mses_test = np.zeros((3,1000))
    mses_train = np.zeros((3,1000))
    for n in neuron_numbers:
        regressor = MLPRegressor(
            hidden_layer_sizes=(n,),
            solver="adam",
            activation="logistic",
            alpha=0.0,
            max_iter=1,
            warm_start=True
        )
        for j in range(0, 1000):
            regressor.fit(x_train, y_train)
            mses_train[neuron_numbers.index(n)][j] = calculate_mse(regressor, x_train, y_train)
            mses_test[neuron_numbers.index(n)][j] = calculate_mse(regressor, x_test, y_test)

    plot_mse_vs_iterations(mses_train,mses_test, 1000, neuron_numbers)
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
    hidden_neurons = 40
    alphas = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    mses_train = np.zeros((len(alphas), 10))
    mses_test = np.zeros((len(alphas), 10))
    for alpha in alphas:
        for i in range(10):
            random_seed = randint(0, 1000)
            regressor = MLPRegressor(
                hidden_layer_sizes=(hidden_neurons,),
                solver="lbfgs",
                activation="logistic",
                alpha=alpha,
                max_iter=200,
                random_state=random_seed
            )
            regressor.fit(x_train, y_train)
            mses_test[alphas.index(alpha)][i] = calculate_mse(regressor, x_test, y_test)
            mses_train[alphas.index(alpha)][i] = calculate_mse(regressor, x_train, y_train)

    plot_mse_vs_alpha(mses_train, mses_test, alphas)


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

    indices = np.random.permutation(len(y_train))
    new_size = int(len(y_train) / 2)

    x_train_ = np.ndarray((new_size, 1))
    y_train_ = np.ndarray(new_size)
    indices_train = indices[:new_size]

    indices_val = indices[new_size:]
    x_val = np.ndarray((new_size, 1))
    y_val = np.ndarray(new_size)

    for i in range(new_size):
        x_train_[i] = x_train[indices_train[i]]
        y_train_[i] = y_train[indices_train[i]]

        x_val[i] = x_train[indices_val[i]]
        y_val[i] = y_train[indices_val[i]]

    x_train = x_train_
    y_train = y_train_

    min_test_errors = np.zeros(10)
    last_test_errors = np.zeros(10)
    min_val_errors = np.zeros(10)
    for i in range(10):
        regressor = MLPRegressor(
            hidden_layer_sizes=(40,),
            solver="lbfgs",
            activation="logistic",
            alpha=10e-3,
            max_iter=1,
            warm_start=True,
            random_state=randint(1,1000)
        )

        val_errors = []
        test_errors = []
        for j in range(0, 2000):
            regressor.fit(x_train, y_train)
            if j % 20 == 0:
                test_errors.append(calculate_mse(regressor, x_test, y_test))
                val_errors.append(calculate_mse(regressor, x_val, y_val))

        last_test_errors[i] = calculate_mse(regressor, x_test, y_test)
        min_val_errors[i] = test_errors[np.argmin(val_errors)]
        min_test_errors[i] = test_errors[np.argmin(test_errors)]

    plot_bars_early_stopping_mse_comparison(last_test_errors,min_val_errors, min_test_errors)
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
    indices = np.random.permutation(len(y_train))
    new_size = int(len(y_train) / 2)

    x_train_ = np.ndarray((new_size, 1))
    y_train_ = np.ndarray(new_size)
    indices_train = indices[:new_size]

    indices_val = indices[new_size:]
    x_val = np.ndarray((new_size, 1))
    y_val = np.ndarray(new_size)

    for i in range(new_size):
        x_train_[i] = x_train[indices_train[i]]
        y_train_[i] = y_train[indices_train[i]]

        x_val[i] = x_train[indices_val[i]]
        y_val[i] = y_train[indices_val[i]]

    x_train = x_train_
    y_train = y_train_

    min_test_errors = np.zeros(10)
    last_test_errors = np.zeros(10)
    min_val_errors = np.zeros(10)
    regs = []

    for i in range(10):
        regressor = MLPRegressor(
            hidden_layer_sizes=(40,),
            solver="lbfgs",
            activation="logistic",
            alpha=10e-3,
            max_iter=1,
            warm_start=True,
            random_state=randint(1, 1000)
        )

        val_errors = []
        test_errors = []
        train_errors = []
        for j in range(0, 2000):
            regressor.fit(x_train, y_train)
            if j % 20 == 0:
                if val_errors and val_errors[-1] < calculate_mse(regressor, x_val, y_val):
                    break
                test_errors.append(calculate_mse(regressor, x_test, y_test))
                val_errors.append(calculate_mse(regressor, x_val, y_val))

        last_test_errors[i] = calculate_mse(regressor, x_test, y_test)
        min_val_errors[i] = test_errors[np.argmin(val_errors)]
        min_test_errors[i] = test_errors[np.argmin(test_errors)]
        regs.append(regressor)


    #TODO things which are needed for report

    optimal_seed = np.argmin(min_val_errors)
    print(min_test_errors)
    print(min_val_errors)

    print("min validation error: ", min_val_errors[optimal_seed])
    print("min test error:       ", min_test_errors[optimal_seed])
    #print("min training error:   ", )

    pass