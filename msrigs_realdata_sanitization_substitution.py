# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Using Real Datasets (Data sanitization process - Step 1: Substitution)
# Copyright 2017-2021 Zhiyu Wan, HIPLAB, Vanderilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1,
# Update history:
# 20201203: input dataset substitution
# 20210609: set the random number seed.

import numpy as np
import time

# main function
if __name__ == '__main__':
    np.random.seed(int(time.time()))  # reset random number generator for comparison
    Y1 = np.genfromtxt('data/Venter_Venter.txt', delimiter=',').astype(int)
    Y1b = Y1.copy()
    Y2 = np.genfromtxt('data/Ysearch_Venter.txt', delimiter=',').astype(int)
    Y2b = Y2.copy()
    n, m = Y2.shape
    for j in range(m):
        unq, indices = np.unique(Y2[:, j], return_inverse=True)
        nz_unq = unq[1:]
        offset = np.random.randint(10)
        result = np.arange(min(nz_unq) + offset, min(nz_unq) + len(unq) - 1 + offset)
        result = np.random.permutation(result)
        result = np.concatenate((np.array([0]), result))
        new_col = result[indices]
        Y2b[:, j] = new_col
        venter_index = np.where(unq == Y1[j])
        Y1b[j] = result[venter_index[0][0]]
    np.savetxt('data/Venter_Venter_sanitized.txt', Y1b.reshape(1, m), fmt="%d", delimiter=',')
    np.savetxt('data/Ysearch_Venter_sanitized.txt', Y2b, fmt="%d", delimiter=',')

    # input Ysearch ID
    Ysearch_ID = []
    with open("data/Ysearch_ID_Venter.txt", "r") as f:
        for line in f.readlines():
            Ysearch_ID.append(line.rstrip("\n").upper())
    Ysearch_ID = np.array(Ysearch_ID)
    unq, indices = np.unique(Ysearch_ID, return_inverse=True)
    cut = np.where(unq == 'VENTER')
    cut = cut[0][0]
    part0 = unq[0:cut]
    part1 = unq[(cut + 1):]
    temp = np.concatenate((part0, part1))
    new_temp = np.random.permutation(temp)
    new_part0 = new_temp[0:cut]
    new_part1 = new_temp[cut:]
    result = np.concatenate((new_part0, np.array(['VENTER']), new_part1))
    new_col = result[indices]
    np.savetxt('data/Ysearch_ID_Venter_sanitized.txt', new_col, fmt="%s", delimiter=',')


