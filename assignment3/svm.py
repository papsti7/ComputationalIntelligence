import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########

    linSVM = svm.SVC(kernel="linear")
    #print("X:", x.shape)
    #print("Y:", y.shape)
    linSVM.fit(x, y)
    plot_svm_decision_boundary(linSVM, x, y)



def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    #print("x:", x)
    #print("y:", y)
    #print("x.shape:", x.shape)
    #print("y.shape:", y.shape)
    point = np.array([4,0])
    #print("point:", point)
    #print("point_shape:", point.shape)
    x_new = np.vstack((x, point))
    #print("x_new:", x_new)
    #print("x_new.shape:", x_new.shape)
    y_new = np.hstack((y, 1))
    #print("y_new:", y_new)
    #print("y_new.shape:", y_new.shape)
    linSVM = svm.SVC(kernel="linear")
    linSVM.fit(x_new, y_new)
    plot_svm_decision_boundary(linSVM, x_new, y_new)


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]
    point = np.array([4, 0])
    x_new = np.vstack((x, point))
    y_new = np.hstack((y, 1))
    linSVM = svm.SVC(kernel="linear")
    for c in Cs:
        linSVM.set_params(C=c)
        linSVM.fit(x_new, y_new)
        plot_svm_decision_boundary(linSVM, x_new, y_new)



def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########

    linSVM = svm.SVC(kernel="linear")
    linSVM.fit(x_train, y_train)

    train_score = linSVM.score(x_train, y_train)
    test_score = linSVM.score(x_test, y_test)

    print("train_score for linear kernel: ", train_score)
    print("test_score for linear kernel: ", test_score)

    plot_svm_decision_boundary(linSVM, x_train, y_train, x_test, y_test)



def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 21)

    train_scores = []
    test_scores = []
    polySVMs = []

    for degree in degrees:
        polySVM = svm.SVC(kernel="poly", coef0=1)
        polySVM.set_params(degree=degree)
        polySVM.fit(x_train, y_train)

        train_scores.append(polySVM.score(x_train, y_train))
        test_scores.append(polySVM.score(x_test, y_test))
        polySVMs.append(polySVM)

    best_test_score_index = np.argmax(test_scores)
    print("best_train_score for poly kernel: ", train_scores[best_test_score_index])
    print("best_test_score for poly kernel: ", test_scores[best_test_score_index])
    print("degree for best test_score: ", degrees[best_test_score_index])
    plot_score_vs_degree(train_scores, test_scores, degrees)

    plot_svm_decision_boundary(polySVMs[best_test_score_index], x_train, y_train, x_test, y_test)

def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)

    train_scores = []
    test_scores = []
    rbfSVMs = []
    for gamma in gammas:
        rbfSVM = svm.SVC(kernel="rbf")
        rbfSVM.set_params(gamma=gamma)
        rbfSVM.fit(x_train, y_train)

        train_scores.append(rbfSVM.score(x_train, y_train))
        test_scores.append(rbfSVM.score(x_test, y_test))
        rbfSVMs.append(rbfSVM)

    best_test_score_index = np.argmax(test_scores)
    print("gamma of best_test_score: ", gammas[best_test_score_index])
    print("best_test_score for rbf kernel: ", test_scores[best_test_score_index])

    plot_score_vs_gamma(train_scores, test_scores, gammas)
    plot_svm_decision_boundary(rbfSVMs[best_test_score_index], x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**-3
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########
    linSVM = svm.SVC(kernel="linear", decision_function_shape='ovr')

    train_scores_rbf = []
    test_scores_rbf = []
    rbfSVMs = []

    gammas = np.linspace(10e-5, 10e-3, 10)
    for gamma in gammas:
        rbfSVM = svm.SVC(kernel="rbf", decision_function_shape='ovr')
        rbfSVM.set_params(gamma=gamma)
        rbfSVM.fit(x_train, y_train)

        train_scores_rbf.append(rbfSVM.score(x_train, y_train))
        test_scores_rbf.append(rbfSVM.score(x_test, y_test))
        rbfSVMs.append(rbfSVM)

    best_test_score__rbf_index = np.argmax(test_scores_rbf)
    print("gamma of best_test_rbf_score: ", gammas[best_test_score__rbf_index])
    print("best_test_score for rbf kernel: ", test_scores_rbf[best_test_score__rbf_index])

    linSVM.fit(x_train, y_train)

    lin_score_train = linSVM.score(x_train, y_train)
    lin_score_test = linSVM.score(x_test, y_test)

    plot_score_vs_gamma(train_scores_rbf, test_scores_rbf, gammas, lin_score_train, lin_score_test, 0)#TODO ask which baseline should be used




def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    linSVM = svm.SVC(kernel="linear", decision_function_shape='ovr')
    linSVM.fit(x_train, y_train)

    labels = range(1, 6)
    y_pred = linSVM.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    #plot_confusion_matrix(conf_matrix, labels)

    main_diagonal = np.diagonal(conf_matrix)
    i = np.argmin(main_diagonal) + 1
    print("most misclassified class: ", i)

    sel_error = np.array([])  # Numpy indices to select images that are misclassified.

    for j in range(y_pred.shape[0]):
        if y_test[j] == i and y_pred[j] != y_test[j]:
            sel_error = np.append(sel_error, j)
            if len(sel_error) == 10:
                break

    sel_error = sel_error.astype(int)
    # Plot with mnist plot
    plot_mnist(x_test[sel_error], y_pred[sel_error], labels=labels[i - 1], k_plots=10, prefix='Real class')
