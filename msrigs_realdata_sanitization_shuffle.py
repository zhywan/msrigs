# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Using Real Datasets (Data sanitization process - Step 2: Shuffle)
# Copyright 2017-2021 Zhiyu Wan, HIPLAB, Vanderilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1,
# Update history:
# 20201203: shuffle rows and columns in the input dataset
# 20210609: set the random number seed.

import numpy as np
import time

# main function
if __name__ == '__main__':
    np.random.seed(int(time.time()))  # reset random number generator for comparison
    Y2 = np.genfromtxt('data/Ysearch_Venter_sanitized.txt', delimiter=',').astype(int)
    Y2b = Y2.copy()
    n, m = Y2.shape
    row_indices = np.arange(n)
    row_indices = np.random.permutation(row_indices)
    col_indices = np.arange(m)
    col_indices = np.random.permutation(col_indices)
    for j in range(m):
        Y2b[:, j] = Y2[row_indices, j]
    for i in range(n):
        Y2b[i, :] = Y2b[i, col_indices]
    Y1 = np.genfromtxt('data/Venter_Venter_sanitized.txt', delimiter=',').astype(int)
    Y1b = Y1[col_indices]

    # input MU
    MU = np.genfromtxt('data/MU_Venter.txt')
    MUb = MU[col_indices]

    # input Ysearch ID
    Ysearch_ID = []
    with open("data/Ysearch_ID_Venter_sanitized.txt", "r") as f:
        for line in f.readlines():
            Ysearch_ID.append(line.rstrip("\n").upper())
    Ysearch_ID = np.array(Ysearch_ID)
    Ysearch_ID_b = Ysearch_ID[row_indices]

    np.savetxt('data/MU_Venter_shuffled.txt', MUb.reshape(m, 1), fmt="%.3f", delimiter=',')
    np.savetxt('data/Venter_Venter_sanitized_shuffled.txt', Y1b.reshape(1, m), fmt="%d", delimiter=',')
    np.savetxt('data/Ysearch_Venter_sanitized_shuffled.txt', Y2b, fmt="%d", delimiter=',')
    np.savetxt('data/Ysearch_ID_Venter_sanitized_shuffled.txt', Ysearch_ID_b, fmt="%s", delimiter=',')


