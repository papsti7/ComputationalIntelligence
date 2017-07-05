#!/usr/bin/env python3
#Filename skeleton_HW6.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 20.6.2017

import numpy as np
import numpy.random as rd
import math

## -------------------------------------------------------
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def is_probability_distribution(p):
    """Check if p represents a valid probability distribution."""
    p = np.array(p)
    return np.isclose(p.sum(), 1.0) and np.logical_and(0.0 <= p, p <= 1.0).all()


def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over
    the support X.
    
    X ... Support of the RV -- (S,)
    PM ... Probabilities P(X) -- (S,)
    N ... Number of samples -- scalar
    """

    X, PM = np.asarray(X), np.asarray(PM)

    assert is_probability_distribution(PM)

    y = np.zeros(N, dtype = X.dtype)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offset = rd.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offset, 1 + offset, 1 / N) # new axis with N values in the range ]0,1[

    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1
        y[i] = X[j]

    return rd.permutation(y) # permutation of all samples


class HMM:
    def __init__(self, A, B, pi):
        """Represents a Hidden Markov Model with given parameters.

        pi ... Prior/initial probabilities -- (N_s,)
        A ... Transition probabilities -- (N_s, N_s)
        B ... Emission probabilities -- (N_s, N_o)
        """

        A, B, pi = np.asarray(A), np.asarray(B), np.asarray(pi)
        
        assert A.shape[0] == B.shape[0] == pi.shape[0]
        assert is_probability_distribution(pi)
        assert all(is_probability_distribution(p) for p in A)
        assert all(is_probability_distribution(p) for p in B)
        
        self.A = A
        self.B = B
        self.pi = pi
        self.N_s = pi.shape[0]  # number of states
        self.N_o = B.shape[1]   # number of possible observations

        
    def viterbi_discrete(self, X):
        """Viterbi algorithm for an HMM with discrete emission probabilities.
        Returns the optimal state sequence q_opt for a given observation sequence X.

        X ... Observation sequence -- (N,)
        """
        print("viterbi")
        X = np.asarray(X)

        # q_opt is the optimal state sequence; this is a default value
        q_opt = np.array([0, 0, 1, 2])

        # TODO: implement Viterbi algorithm
        print("PI:", self.pi)
        print("A:", self.A)
        print("B:", self.B)
        init_delta = self.pi*self.B.T[X[0]]
        # print("init", init_delta)
        # for n in range(1, len(X)):
        #     tmp_delta = init_delta
        #     for j in range(0, self.N_s):
        #         sum = 0
        #         for i in range(0, self.N_s):
        #             sum += self.A[i][j] * tmp_delta[i]
        #         tmp_delta[j] = sum * self.B[j][X[n]]
        #     init_delta = tmp_delta
        #
        print("alphas:", init_delta)
        p_asterisk = np.zeros((len(X), 1))
        q_asterisk = np.zeros(len(X), dtype=int)
        init_delta = self.pi*self.B.T[X[0]]
        psi = np.zeros((len(X), 3))
        print("init", init_delta)
        #p_asterisk[0] = init_delta.max()
        #q_asterisk[0] = int(init_delta.argmax())
        for n in range(1, len(X)):
            tmp_delta = np.array(init_delta)
            for j in range(0, self.N_s):
                tmp = np.zeros((3,1))
                for i in range(0, self.N_s):
                    tmp[i] = self.A[i][j] * tmp_delta[i]
                maxi = tmp.max()
                psi[n][j] = tmp.argmax()
                init_delta[j] = maxi * self.B[j][X[n]]

        print("delta:", init_delta)
        print("argmaxi:", psi)

        p_asterisk[len(X) - 1] = init_delta.max()
        q_asterisk[len(X) - 1] = int(init_delta.argmax())

        print("p_astersik:", p_asterisk)
        print("q_astersik:", q_asterisk)

        for n in range(len(X) - 2, -1, -1):
            q_asterisk[n] = int(psi[n + 1][int(q_asterisk[n + 1])])

        print("delta:", init_delta)
        print("q_asterisk:", q_asterisk)
        return q_asterisk


    def sample(self, N):
        """Draw a random state and corresponding observation sequence of length N from the model."""
        # TODO: implement sampling from HMM
        pass




## -------------------------------------------------------
## ------------- START OF  ASSIGNMENT 6 ------------------
## -------------------------------------------------------
def main():
    # define states
    states = ['s', 'r', 'f']    # 3 States: Sun, Rain, Fog

    # define HMM 1
    A1 = np.array([[0.8, 0.05, 0.15],
                   [0.2, 0.6, 0.2],
                   [0.2,0.3,0.5]]) # Transition Prob. Matrix
    B1 = np.array([[0.1, 0.9],
                   [0.8, 0.2],
                   [0.3, 0.7]]) # Emission Prob. rows correspond to states, columns to observations
    pi1 = np.array([1/3, 1/3, 1/3]) # Prior
    hmm1 = HMM(A1, B1, pi1)

    # define HMM 2
    A2 = np.array([[0.6, 0.20, 0.20],
                  [0.05, 0.7, 0.25],
                  [0.05, 0.6, 0.35]]) # Transition Prob. Matrix
    B2 = np.array([[0.3, 0.7],
                  [0.95, 0.05],
                  [0.5, 0.5]]) #Emission prob. rows correspond to states, columns to observations
    pi2 = np.array([1/3, 1/3, 1/3]) # Prior
    hmm2 = HMM(A2, B2, pi2)

    # define observation sequences
    X1 = np.array([0, 0, 1, 1, 1, 0])    # 0 = umbrella
    X2 = np.array([0, 0, 1, 1, 1, 0, 0]) # 1 = no umbrella


    # 1.1.) apply Viterbi to find the optimal state sequence and assign the corresponding states
    # TODO: implement in HMM.viterbi_discrete

    # --- example usage of viterbi_discrete:
    optimal_state_sequence = hmm1.viterbi_discrete(X1)
    print("end viterbi")
    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    optimal_state_sequence = hmm1.viterbi_discrete(X2)
    print("end viterbi")
    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    optimal_state_sequence = hmm2.viterbi_discrete(X1)
    print("end viterbi")
    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])


    optimal_state_sequence = hmm2.viterbi_discrete(X2)
    print("end viterbi")
    print(optimal_state_sequence)
    print([states[i] for i in optimal_state_sequence])

    # 1.2.) Sequence Classification
    # TODO

    

    # 1.3.) Sample from HMM
    # TODO: implement in HMM.sample


if __name__ == '__main__':
    main()
