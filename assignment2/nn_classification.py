from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc, plot_image
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

    classifier.fit(input2, target2[:,1])
    con_mat = confusion_matrix(target2[:,1], classifier.predict(input2))
    plot_hidden_layer_weights(classifier.coefs_[0])


def ex_2_2(input1, target1, input2, target2):
    ## TODO
    scores = []
    scores_train = []
    classifiers = []
    for i in range(10):
        classifier = MLPClassifier(
            hidden_layer_sizes=(20,),
            solver="adam",
            max_iter=1000,
            activation="tanh",
            random_state=i
        )
        classifier.fit(input1, target1[:, 0])
        scores.append(classifier.score(input2, target2[:,0]))
        classifiers.append(classifier)
        scores_train.append(classifier.score(input1, target1[:,0]))

    conf_mat = confusion_matrix(target2[:,0], classifiers[np.argmax(scores)].predict(input2))

    plot_histogram_of_acc(scores_train, scores)
    #plot_histogram_of_acc(classifiers[np.argmax(scores)], classifier.score(input2, target2[:, 0]))
    #plot_histogram_of_acc(classifier.score(input1, target1[:,0]), classifier.score(input2, target2[:,0]))
    predected_target = classifier.predict(input2)
    misclassified_images = []
    for i in range(len(target2[:,0])):
        if target2[:,0][i] != predected_target[i]:
            misclassified_images.append(input2[i])

    for i in range(len(misclassified_images)):
        plot_image(misclassified_images[i])
    pass

