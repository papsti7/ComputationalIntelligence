from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
import numpy as np



__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    ## TODO

    classifier = MLPClassifier(
        hidden_layer_sizes=(6,),
        solver="adam",
        max_iter=200,
        activation="tanh"
    )
    print(input2.shape)
    print(target2.shape)
    conf_mat = confusion_matrix(input2, target2)
    print(conf_mat)
    pass


def ex_2_2(input1, target1, input2, target2):
    ## TODO
    pass

