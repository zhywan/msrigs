# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Using Simulated Datasets (All data subjects have the same strategy)
# Copyright 2018-2021 Zhiyu Wan, HIPLAB, Vanderilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Scikit-learn 0.23.2, Pandas 1.1.3, Matplotlib 3.3.1,
# Seaborn 0.11.0, and SciPy 1.5.2
# Update history:
# Added the surname inference algorithm from the paper "Identifying personal genomes by surname inference", Science 2013
# Nov 19, 2018: Added 3 global dictionaries for distance and confidence score computation;
# Nov 19, 2018: Deleted mulist, xlist; Let etmrca function be more efficient by using cache and pre-computing
# Nov 21, 2018: Added a global dictionary for attack simulation computation
# Nov 23, 2018: Utility for each attribute is proportional to entropy
# Nov 27, 2018: Adding generalization strategies for age and state attributes
# Dec 2, 2018: Adding weight vector for utility, and adding missing data to G
# Dec 17, 2018: Individual-wise defense strategy
# Dec 17, 2018: Adding a surname inference dictionary
# Jan 23, 2019: Deleting the dictionary for defense computation
# March 18, 2019: Fixing a bug regarding tuple_x (all changed to tuple_x_selection)
# March 12, 2020: 3)save detailed results!
#                 4)add the scenario of data masking! 5)add the scenario that only opt-in or opt-out!
#                 6)add the attacker's cost (attack rate * c)! 7)add the defender's cost (L*pa)!
# March 17, 2020: 8)consider the scenario of always_attack
# March 18, 2020: 2)fix the total utility for various m
# March 19, 2020: 1)threshold for theta
# March 31, 2020: Same strategy for all individuals
# April 24, 2020: Clean the code, delete the "defense mode"
# Oct 17, 2020: Split into main program

import numpy as np
import time
import pandas as pd
import msrigs_functions as sf
import os.path
import pickle

# configuration
id_exp = '2056'  # ID for the set of experiments
n_f: int = 20  # number of firstnames
n_I: int = 20000  # size of the identified dataset (<=90000)
n_S: int = 1000  # size of the sensitive dataset (<=90000)
n_G: int = 20000  # size of the genetic genealogy dataset (<=90000)
rate_s = 0.6  # rate of sensitive
loss = 150
cost = 10
total_utility = 100
n_iter: int = 20
theta_p = 0.5
method: int = 2
m_g: int = 12  # default: 16
weight = np.concatenate((np.ones(2) * 1, np.ones(m_g)), axis=None)
missing_level = 0.3
over_confident: int = 0
alter_weight: int = 0  # 0: Based on information entropy. 1: Uniform. 2: Special (the weight of 1st 2 geno features is 10x).
algorithm: int = 0  # 0: greedy algorithm. 1: brute-force algorithm.
pruning: int = 1
participation_rate = 0.05
random_masking_rate = 0.15
save_dic = 0

# choose a scenario
scenario = 0  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking.
# 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.

# creat folders
folder_result = 'Results' + id_exp + '/Violin_oneforall'
if over_confident == 0 and alter_weight == 0 and algorithm == 0:
    folder_result += '/m'+str(method) + '/'
elif over_confident == 1 and alter_weight == 0 and algorithm == 0:
    folder_result += '_over_confident/m' + str(method) + '/'
elif alter_weight != 0 and over_confident == 0 and algorithm == 0:
    folder_result += '_multi_weight_distributions/Alter_weight_' + str(alter_weight) + '/m' + str(method) + '/'
elif algorithm == 1 and over_confident == 0 and alter_weight == 0:
    folder_result += '_bf/'
else:
    print('The configuration is not correct.')
if pruning == 1:
    folder_result += 'pruning/'
# check the existence of the directory
folders = folder_result.rstrip('/').split('/')
folder = ''
for folder_name in folders:
    folder += folder_name + '/'
    if not os.path.exists(folder):
        os.mkdir(folder)

pickle_filename = folder_result + 'result_s' + str(scenario) + '.pickle'
filename = folder_result + 'log_s' + str(scenario) + '.txt'

mu = 1e-3 * np.array([2.381, 2.081, 1.781, 2.803, 2.298, 3.081, 0.552, 0.893, 1.498, 0.425, 5.762, 1.590,
                      4.769, 6.359, 3.754, 2.180])  # updated mutation rate in 2008
tol = 0.2
Ne = 10000
inv_Ne = 1.0 / Ne
T_Max = 200
I_selection = np.array([2, 3, -2]).astype(int)

if __name__ == '__main__':
    start1 = time.time()
    surname = []
    genome = []
    ages = []
    states = []
    ID = []
    n_r = []
    array_optimal_payoff = np.empty(n_iter)
    array_optimal_attacker_payoff = np.empty(n_iter)
    array_privacy = np.empty(n_iter)
    array_utility = np.empty(n_iter)
    sum_opt_strategy = np.zeros(m_g + 2)
    dic_dist = {}
    dic_score_solo = {}
    dic_score = {}
    for i in range(3):
        n = 0
        f = open("data/simu/ped"+str(i+1)+".txt", "r")
        f2 = open("data/simu/surname" + str(i + 1) + ".txt", "r")
        f3 = open("data/simu/birth_year" + str(i + 1) + ".txt", "r")
        f4 = open("data/simu/state" + str(i + 1) + ".txt", "r")
        for line in f.readlines():
            line2 = f2.readline()
            line3 = f3.readline()
            line4 = f4.readline()
            sname = int(float(line2.rstrip("\n")))  #
            age = 2020-int(float(line3.rstrip("\n")))
            state = int(float(line4.rstrip("\n")))
            loci = line.rstrip("\n").split(" ")
            if loci[3] == 'M':
                n += 1
                y1 = []
                for j in range(len(loci)):
                    if j == 0:
                        ID.append(int(loci[j]))
                    elif j >= 5 and j % 2 == 0:
                        y1.append(int(loci[j]))
                genome.append(y1)
                surname.append(sname)
                ages.append(age)
                states.append(state)
        n_r.append(n)
        f.close()
        f2.close()
        f3.close()
        f4.close()
    elapsed1 = (time.time() - start1)
    f = open(filename, 'w')
    f.write(folder_result + '\n')
    start2 = time.time()
    for i in range(n_iter):
        print('iter: ', i)
        np.random.seed(i)  # reset random number generator for comparison
        World = sf.build_world(ID, genome, surname, ages, states, n_r, n_f, rate_s)
        (S, I, G2) = sf.generate_datasets(World, n_I, n_S, n_G)  #G2 has ground truth
        # ID, first name, ages, states, genomic attributes, surname, sensitive
        # Add missing values
        if missing_level > 0:
            np.random.seed(i)
            n_missing = int(n_G * m_g * missing_level)
            missed = np.append(np.zeros(n_missing).astype(int), np.ones(n_G * m_g - n_missing).astype(int))
            np.random.shuffle(missed)
            G1 = np.multiply(G2[:, 4:(4 + m_g)], missed.reshape(n_G, m_g))
            G = np.concatenate((G2[:, 0:4], G1, G2[:, -2:]), axis=1)
        else:
            G = G2
        # Compute entropy
        if alter_weight == 0:
            entropy = []
            for j in range(m_g + 2):
                if j == 0 or j == 1:  # entropy in demographic dataset
                    c = I[:, j + 2]
                else:  # entropy in genetic genealogy dataset
                    c = G[:, j + 2]
                entropy.append(sf.get_entropy(c))
            entropy = np.asarray(entropy)
            weighted_entropy = np.multiply(entropy, weight)
        elif alter_weight == 1:
            weighted_entropy = weight
        elif alter_weight == 2:
            weighted_entropy = np.concatenate((np.ones(2), np.ones(2) * 10, np.ones(m_g - 2)), axis=None)
        # f.write(str(i)+'-'+str(weighted_entropy) + '\n')
        dic_attack = {}
        dic_surname = {}
        (opt_payoff, opt_attacker_payoff, opt_attack, opt_success_rate, opt_utility, opt_strategy) = \
            sf.optimal_defense_all(S, I, G, weighted_entropy, m_g, dic_attack, dic_surname, loss, cost, scenario,
                               total_utility, theta_p, over_confident, mu, method, tol, dic_dist,
                               dic_score_solo, dic_score, T_Max, inv_Ne, participation_rate, random_masking_rate,
                               algorithm, pruning, I_selection)
        array_optimal_payoff[i] = opt_payoff
        array_optimal_attacker_payoff[i] = opt_attacker_payoff
        array_privacy[i] = 1 - opt_success_rate * opt_attack
        array_utility[i] = opt_utility
        sum_opt_strategy += opt_strategy
        f.write(
            '{}: {} {:f} {:d} {}\n'.format(i, np.array(list(map(int, opt_strategy))), opt_payoff, opt_attack,
                                           opt_success_rate))
    dataset = pd.DataFrame({'privacy': array_privacy,
                            'utility': array_utility,
                            'defender_optimal': array_optimal_payoff,
                            'attacker_optimal': array_optimal_attacker_payoff})
    dataset.to_pickle(pickle_filename)
    f.write('Average strategy: ' + str(sum_opt_strategy / n_iter) + '\n')
    f.write('Data subjects\' average payoff: ' + str(np.mean(array_optimal_payoff)) + '\n')
    elapsed2 = (time.time() - start2)
    f.write("Time used: " + str(elapsed1) + " seconds (loading) + " + str(elapsed2) + " seconds (computing).\n")
    f.close()
    if save_dic == 1:
        # save dictionaries
        dic_names = ['dist', 'score', 'score_solo', 'attack', 'surname']
        dics = [dic_dist, dic_score, dic_score_solo, dic_attack, dic_surname]
        for i in range(5):
            f1 = open(folder_result + 'dic_s' + str(scenario) + '_' + dic_names[i] + '.pkl', 'wb')
            pickle.dump(dics[i], f1, protocol=pickle.HIGHEST_PROTOCOL)
            f1.close()
