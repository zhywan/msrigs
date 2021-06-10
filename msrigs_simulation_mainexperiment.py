# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Using Simulated Datasets (Data subjects have different strategies)
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
# April 16, 2020: To plot privacy and utility. To plot error bar.
# April 19, 2020: clean the code
# April 21, 2020: functions been moved out, changed optimal defense output
# April 21, 2020: 1) output all data points instead of just average values. 2)deleted "defense_mode" and "n_cost"
# April 23, 2020: 1) add "reorder_genome" var, 2) add method 3&4 in functions
# May 9, 2020: add the non-zero x_nz and non-zero mu_nz and add mu1 and mu2 in the conf_score function
# May 10, 2020: tol = 0.2. theta_p = 0.45
# July 10, 2020: theta_p = 0.5
# July 11, 2020: Modify one scenario, and add three new scenarios.
# July 12, 2020: Rename saved file.
# July 28, 2020: Add alternative weight distributions.
# July 31, 2020: Add brute-force algorithm and pruning tech.
# Aug 11, 2020: Simplify arguments (scenario instead of no_defense, in_out, no_geno and random_protection)
# Aug 11, 2020: change output filename
# Aug 20, 2020: accelerate: 1) vectorize surname inference, 2) update dic_attack, 3) change the way to handle mask.
# Aug 23, 2020: update dic_attack (one for each iteration).
# Sep 23, 2020: fix the missing rate.
# Oct 10, 2020: Add the no-attack masking game.
# Oct 20, 2020: Add more options: short_memory, save_iter, and start_iter.
# March 31, 2021: Allow customized strategy.
# April 21, 2021: Change the way to compute the utility function again

import numpy as np
import time
import pandas as pd
import msrigs_functions as sf
import os.path
import pickle
import sys

# configuration
id_exp = '2058'  # ID for the set of experiments
start_iter: int = 0  # start from a particular iteration (default: 0)
n_iter: int = 100  # (default: 100)
n_f: int = 20  # number of firstnames (not used)
n_I: int = 20000  # size of the identified dataset (<=90000) (default: 20000)
n_S: int = 1000  # size of the sensitive dataset (<=90000) (default: 1000)
n_G: int = 20000  # size of the genetic genealogy dataset (<=90000) (default: 20000)
rate_s = 0.6  # rate of sensitive (not used)
loss = 150  # (default: 150)
cost = 10  # (default: 10)
base_utility = 100  # (default: 100)
theta_p = 0.5  # (default: 0.5)
method: int = 2  # (default: 2)
m_g: int = 12  # (<=16) (default: 12)
weight = np.concatenate((np.ones(2) * 1, np.ones(m_g)), axis=None)
missing_level = 0.3  # (default: 0.3)
over_confident: int = 0  # (default: 0)
alter_weight: int = 0  # *0: Based on information entropy. 1: Uniform. 2: Special (the weight of 1st 2 geno features is 10x).
algorithm: int = 0  # *0: greedy algorithm. 1: brute-force algorithm.
pruning: int = 1  # (default: 1)
participation_rate = 0.05  # (default: 0.05)
random_masking_rate = 0.15  # (default: 0.15) probability of sharing in the random masking scenario
alpha = 0  # minority-support factor. 0: original, 1:recommended, >0: minority oriented, <0: majority oriented.
save_opt_strategy: bool = True  # save optimal strategies (to csv file) for all data subjects (default: False)
log_opt_strategy: bool = False  # log optimal strategies (to pickle file) for all data subjects (default: False)
save_dic: bool = False  # save all dictionaries into files in the end (default: False)
load_dic: bool = False  # load all global dictionaries (dic_dist, dic_score, dic_score_solo) in the beginning (default: False)
short_memory_dic: bool = False  # refresh dictionaries in each iteration (no need to load/save dictionaries) (default: False)
short_local_memory_dic: bool = False  # refresh local dictionaries (attack, surname) for each subject (default: False)
save_iter: bool = False  # save results in each iteration (different file names) (default: False)

save_S: bool = True  # save S in each iteration (only work for scenario 0) (default: True)
save_G: bool = True  # save G in each iteration (only work for scenario 0) (default: True)
save_I: bool = True  # save I in each iteration (only work for tested_scenario 0) (default: True)
save_beta: bool = True  # save beta and total utility in each iteration (only work for scenario 0) (default: True)
save_weighted_entropy: bool = True  # save weighted entropy (only work for scenario 0) (default: True)

# choose a scenario
scenario = 0  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking. 3.x: custom masking
# 3.1: k-anonymity. 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.

# creat folders
folder_result = 'Results' + id_exp + '/Violin'
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

mu = 1e-3 * np.array([2.381, 2.081, 1.781, 2.803, 2.298, 3.081, 0.552, 0.893, 1.498, 0.425, 5.762, 1.590,
                      4.769, 6.359, 3.754, 2.180])  # updated mutation rate in 2008
tol = 0.2
Ne = 10000
inv_Ne = 1.0 / Ne
T_Max = 200
I_selection = np.array([2, 3, -2]).astype(int)

if __name__ == '__main__':
    start1 = time.time()
    # Enable the input of parameters including start_iter and n_iter
    if len(sys.argv) >= 2:
        start_iter = int(sys.argv[1])
    if len(sys.argv) >= 3:
        n_iter = int(sys.argv[2])
    if start_iter > 0:  # start from the middle
        save_iter = True
    # # Initialize
    surname = []
    genome = []
    ages = []
    states = []
    ID = []
    n_r = []
    # Initialize dictionaries
    if load_dic and os.path.exists(folder_result + 'dic_dist.pkl'):
        with open(folder_result + 'dic_dist.pkl', 'rb') as f1:
            dic_dist = pickle.load(f1)
    else:
        dic_dist = {}
    if load_dic and os.path.exists(folder_result + 'dic_score_solo.pkl'):
        with open(folder_result + 'dic_score_solo.pkl', 'rb') as f1:
            dic_score_solo = pickle.load(f1)
    else:
        dic_score_solo = {}
    if load_dic and os.path.exists(folder_result + 'dic_score.pkl'):
        with open(folder_result + 'dic_score.pkl', 'rb') as f1:
            dic_score = pickle.load(f1)
    else:
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
    if not save_iter:
        array_optimal_payoff = np.empty(n_iter * n_S)
        array_optimal_attacker_payoff = np.empty(n_iter * n_S)
        array_privacy = np.empty(n_iter * n_S)
        array_utility = np.empty(n_iter * n_S)
        array_success_rate = np.empty(n_iter * n_S)
        array_usefulness = np.empty(n_iter)
        array_fairness_wrt_payoff = np.empty(n_iter)
        array_fairness_wrt_privacy = np.empty(n_iter)
        array_fairness_wrt_utility = np.empty(n_iter)
        array_fairness_wrt_usefulness = np.empty(n_iter)
        sum_opt_strategy = np.zeros(m_g + 2)
        if log_opt_strategy or save_opt_strategy:
            array_attack = np.empty(n_iter * n_S).astype(bool)
            list_opt_strategy = []
        pickle_filename = folder_result + 'result_s' + str(scenario) + '.pickle'
        pickle_filename2 = folder_result + 'result2_s' + str(scenario) + '.pickle'
        filename = folder_result + 'log_s' + str(scenario) + '.txt'
        f = open(filename, 'w')
    elapsed1 = (time.time() - start1)
    start2 = time.time()
    for i in range(start_iter, start_iter + n_iter):
        if save_iter:
            start2 = time.time()
            array_optimal_payoff = np.empty(n_S)
            array_optimal_attacker_payoff = np.empty(n_S)
            array_privacy = np.empty(n_S)
            array_utility = np.empty(n_S)
            array_success_rate = np.empty(n_S)
            array_usefulness = np.empty(1)
            array_fairness_wrt_payoff = np.empty(1)
            array_fairness_wrt_privacy = np.empty(1)
            array_fairness_wrt_utility = np.empty(1)
            array_fairness_wrt_usefulness = np.empty(1)
            sum_opt_strategy = np.zeros(m_g + 2)
            if log_opt_strategy or save_opt_strategy:
                array_attack = np.empty(n_S).astype(bool)
                list_opt_strategy = []
            pickle_filename = folder_result + 'result_s' + str(scenario) + '_i' + str(i) + '.pickle'
            pickle_filename2 = folder_result + 'result2_s' + str(scenario) + '_i' + str(i) + '.pickle'
            filename = folder_result + 'log_s' + str(scenario) + '_i' + str(i) + '.txt'
            f = open(filename, 'w')
        print('iter: ', i)
        np.random.seed(i)  # reset random number generator for comparison
        World = sf.build_world(ID, genome, surname, ages, states, n_r, n_f, rate_s)
        (S, I, G2) = sf.generate_datasets(World, n_I, n_S, n_G)  #G2 has ground truth
        # ID, first name, ages, states, genomic attributes, surname, sensitive

        # save S
        if save_S and scenario == 0:
            if not os.path.exists(folder_result + 'target_data'):
                os.mkdir(folder_result + 'target_data')
            header = "YOB,State"
            for j in range(m_g):
                header += ",STR" + str(j+1)
            np.savetxt(folder_result + 'target_data/i' + str(i) + '.csv', S[:, 2:(2+m_g+2)], delimiter=',', fmt='%d',
                       header=header, comments='')

        # save I
        if save_I and scenario == 0:
            if not os.path.exists(folder_result + 'identified_data'):
                os.mkdir(folder_result + 'identified_data')
            header = "YOB,State,Surname"
            np.savetxt(folder_result + 'identified_data/i' + str(i) + '.csv', I[:, [2, 3, (4 + m_g)]], delimiter=',',
                       fmt='%d', header=header, comments='')

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

        # save G
        if save_G and scenario == 0:
            if not os.path.exists(folder_result + 'genealogy_data'):
                os.mkdir(folder_result + 'genealogy_data')
            header = "STR1"
            for j in range(m_g - 1):
                header += ",STR" + str(j + 2)
            header += ",Surname"
            np.savetxt(folder_result + 'genealogy_data/i' + str(i) + '.csv',
                       G[:, 4:(4 + m_g + 1)], delimiter=',',
                       fmt='%d', header=header, comments='')

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

        if save_weighted_entropy and scenario == 0:  # save weighted entropy
            if not os.path.exists(folder_result + 'weighted_entropy'):
                os.mkdir(folder_result + 'weighted_entropy')
            np.savetxt(folder_result + 'weighted_entropy/i' + str(i) + '.csv', weighted_entropy, delimiter=',', fmt='%f')

        # compute group-wise minority level (beta)
        dic_beta = {}
        list_values = []
        list_counts = []
        for j in range(m_g + 2):
            if j == 0 or j == 1:  # population using demographic dataset
                c = I[:, j + 2]
            else:  # population using genetic genealogy dataset
                c = G[:, j + 2]
            values, counts = np.unique(c, return_counts=True)
            # handle missing value
            if j > 1 and values[0] == 0:  # may have missing value
                values = values[1:]
                counts = counts[1:]
            list_values.append(values)
            list_counts.append(counts)
            for k in range(values.size):
                dic_beta[(j, values[k])] = np.log2(c.size/values.size/counts[k]+1)

        # load customized strategy
        custom_strategy_folder = "custom_strategy/"
        if scenario == 3.1:
            custom_strategy_folder += "k_anonymity/"
        custom_strategy_filename = folder_result + custom_strategy_folder + 'i' + str(i) + '.csv'
        if os.path.exists(custom_strategy_filename):
            custom_strategies = np.genfromtxt(custom_strategy_filename, delimiter=',').astype(bool)
        else:
            print(custom_strategy_filename + " does not exist!")
            custom_strategies = np.ones([n_S, 2 + m_g]).astype(bool)  # default
        if short_memory_dic:
            dic_dist = {}
            dic_score = {}
            dic_score_solo = {}
        dic_attack = {}
        dic_surname = {}
        total_utility_save = np.empty(n_S)
        beta_save = np.empty([n_S, m_g+2])
        for j in range(n_S):
            print('j: ', j)
            s = S[j, :]
            if short_local_memory_dic:
                dic_attack = {}
                dic_surname = {}

            # compute beta (minority level) and so on
            beta = np.empty(m_g + 2)
            for k in range(m_g + 2):
                if (k, s[k+2]) in dic_beta:
                    beta[k] = dic_beta[(k, s[k+2])]
                else:
                    temp_values = list_values[k]
                    temp_counts = list_counts[k]
                    beta[k] = np.log2((np.sum(temp_counts)+1)/(temp_values.size+1)+1)
                    dic_beta[(k, s[k+2])] = beta[k]
            beta2alpha = np.power(beta, alpha)
            w_beta2alpha = np.multiply(weighted_entropy, beta2alpha)
            beta_save[j, :] = beta
            utility_boost = np.sum(w_beta2alpha[0:(m_g + 2)]) / np.sum(weighted_entropy[0:(m_g + 2)])
            total_utility = base_utility * utility_boost
            total_utility_save[j] = total_utility

            (opt_payoff, opt_attacker_payoff, opt_attack, opt_success_rate, opt_utility, opt_strategy) = \
                sf.optimal_defense(s, I, G, w_beta2alpha, m_g, dic_attack, dic_surname, loss, cost, scenario,
                                   base_utility, theta_p, over_confident, mu, method, tol, dic_dist,
                                   dic_score_solo, dic_score, T_Max, inv_Ne, participation_rate, random_masking_rate,
                                   algorithm, pruning, I_selection, custom_strategies[j, :], utility_boost)
            if save_iter:
                index = j
            else:
                index = i * n_S + j
            array_optimal_payoff[index] = opt_payoff
            array_optimal_attacker_payoff[index] = opt_attacker_payoff
            array_privacy[index] = 1 - opt_success_rate * opt_attack
            array_utility[index] = opt_utility
            array_success_rate[index] = opt_success_rate
            sum_opt_strategy += opt_strategy
            if log_opt_strategy or save_opt_strategy:
                array_attack[index] = opt_attack
                list_opt_strategy.append(opt_strategy)
            f.write(
                '{}-{}: {} {:f} {:d} {}\n'.format(i, j, np.array(list(map(int, opt_strategy))), opt_payoff, opt_attack,
                                                  opt_success_rate))
        # # compute and save dataset-wise measures
        if save_iter:
            index = 0
        else:
            index = i

        # compute usefulness
        a = np.empty([2 + m_g])
        SS = S[:, 2:(2 + m_g + 2)]  # essential part of S
        array_opt_strategies = np.stack(list_opt_strategy)
        if scenario > 0:
            array_opt_strategy_i = array_opt_strategies[-n_S:, :]
        elif scenario == 0:
            array_opt_strategy_i = np.ones([n_S, 2 + m_g]).astype(int)
        for j in range(2 + m_g):
            S1 = SS[:, j]
            # From Age to Birth_year
            if j == 0:
                S1 = 2020 - S1
            if scenario > 0:
                array_opt_strategy1 = array_opt_strategy_i[:, j]
                S_output = S1[array_opt_strategy1 > 0]
            else:
                S_output = S1
            # compute distance
            values, counts = np.unique(S1, return_counts=True)
            counts2 = np.copy(counts)
            for k in range(values.size):
                counts2[k] = np.count_nonzero(S_output == values[k])
            if np.sum(counts2) == 0:
                S_stat = 1
                print('Undefined distance!')
            elif counts.size == 1:
                S_stat = 0
                print('One-point distribution!')
            else:
                p = counts / np.sum(counts)
                q = counts2 / np.sum(counts2)
                S_stat = 0.5 * np.sum(np.abs(p-q))  # variational_distance(p, q)
            a[j] = 1 - S_stat
        array_usefulness[index] = np.dot(a, weighted_entropy) / np.sum(weighted_entropy)

        # Compute fairness measures
        group_hr_birth_year = list(range(1910, 2000, 10))  # [1910, 1920, 1930, 1940, 1950, 1960, 1970, ..., 1990]
        group_hr_state = list(range(11, 51, 10))  # [11, 21, 31, 41]
        a_fairness_wrt_payoff = np.empty(2)
        a_fairness_wrt_privacy = np.empty(2)
        a_fairness_wrt_utility = np.empty(2)
        a_fairness_wrt_usefulness = np.empty(2)
        for targeted_attribute in range(2):
            if targeted_attribute == 0:
                group_hr = group_hr_birth_year
            else:
                group_hr = group_hr_state
            n_groups = len(group_hr) + 1
            av_optimal_payoff = np.zeros([n_groups])
            av_privacy = np.zeros([n_groups])
            av_utility = np.zeros([n_groups])
            a_usefulness = np.zeros([n_groups, 2 + m_g + 1])
            if save_iter:
                array_optimal_payoff_i = array_optimal_payoff
                array_privacy_i = array_privacy
                array_utility_i = array_utility
            else:
                array_optimal_payoff_i = array_optimal_payoff[(i * n_S):((i + 1) * n_S)]
                array_privacy_i = array_privacy[(i * n_S):((i + 1) * n_S)]
                array_utility_i = array_utility[(i * n_S):((i + 1) * n_S)]
            S_targeted = SS[:, targeted_attribute]
            # From Age to Birth_year
            if targeted_attribute == 0:
                S_targeted = 2020 - S_targeted
            memberships = np.copy(S_targeted)
            for j in range(n_S):  # for each person
                index_group = 0
                for k in range(n_groups - 1):  # for each group (excluding the 1st)
                    if S_targeted[j] < group_hr[k]:
                        break
                    else:
                        index_group += 1
                memberships[j] = index_group
            for j in range(n_groups):  # for each group
                selection = memberships == j
                av_optimal_payoff[j] = np.mean(array_optimal_payoff_i[selection])
                av_privacy[j] = np.mean(array_privacy_i[selection])
                av_utility[j] = np.mean(array_utility_i[selection])

                # usefulness compuation
                for k in range(2 + m_g):  # for each attribute
                    S1 = SS[:, k]
                    # From Age to Birth_year
                    if k == 0:
                        S1 = 2020 - S1
                    if scenario > 0:
                        array_opt_strategy1 = array_opt_strategy_i[:, k]
                        array_opt_strategy_targeted = array_opt_strategy_i[:, targeted_attribute]
                        S_output = S1[np.logical_and.reduce((array_opt_strategy1 > 0, selection,
                                                             array_opt_strategy_targeted > 0))]  # shared data in this group
                    else:  # no protection
                        S_output = S1[selection]
                    S1 = S1[selection]  # original data in this group
                    values, counts = np.unique(S1, return_counts=True)
                    counts2 = np.copy(counts)
                    for kk in range(values.size):
                        counts2[kk] = np.count_nonzero(S_output == values[kk])
                    if np.sum(counts2) == 0:
                        S_stat = 1
                        print('Undefined distance! demo_attr: ' + str(targeted_attribute) + ', group: '
                              + str(j) + ', attr: ' + str(k) + '.')
                    elif counts.size == 1:
                        S_stat = 0
                        print('One-point distribution!')
                    else:
                        p = counts / np.sum(counts)
                        q = counts2 / np.sum(counts2)
                        S_stat = 0.5 * np.sum(np.abs(p-q))  # variational_distance(p, q)
                    a_usefulness[j, k] = 1 - S_stat
                a_usefulness[j, -1] = np.dot(a_usefulness[j, 0:(2+m_g)], weighted_entropy) / np.sum(weighted_entropy)
            a_fairness_wrt_payoff[targeted_attribute] = 1 - sf.gini(av_optimal_payoff)
            a_fairness_wrt_privacy[targeted_attribute] = 1 - sf.gini(av_privacy)
            a_fairness_wrt_utility[targeted_attribute] = 1 - sf.gini(av_utility)
            a_fairness_wrt_usefulness[targeted_attribute] = 1 - sf.gini(a_usefulness[:, -1])
        array_fairness_wrt_payoff[index] = np.mean(a_fairness_wrt_payoff)
        array_fairness_wrt_privacy[index] = np.mean(a_fairness_wrt_privacy)
        array_fairness_wrt_utility[index] = np.mean(a_fairness_wrt_utility)
        array_fairness_wrt_usefulness[index] = np.mean(a_fairness_wrt_usefulness)
        if save_beta and scenario == 0:  # save total utility and beta
            if not os.path.exists(folder_result + 'minority_level'):
                os.mkdir(folder_result + 'minority_level')
            np.savetxt(folder_result + 'minority_level/beta_i' + str(i) + '.csv', beta_save, delimiter=',', fmt='%f')
            np.savetxt(folder_result + 'minority_level/total_utility_i' + str(i) + '.csv', total_utility_save,
                       delimiter=',', fmt='%f')
        if save_opt_strategy:
            array_opt_strategy = np.stack(list_opt_strategy)
            # save opt_strategy per iteration
            if not os.path.exists(folder_result + 'opt_strategy'):
                os.mkdir(folder_result + 'opt_strategy')
            np.savetxt(folder_result + 'opt_strategy/s' + str(scenario) + '_i' + str(i) + '.csv',
                       array_opt_strategy[((i - start_iter) * n_S):(((i - start_iter) + 1) * n_S), :],
                       delimiter=',', fmt='%d')
        if not save_iter and i < (start_iter + n_iter - 1):  # not the last iteration and not in save-iteration mode
            continue
        if save_iter:
            n_all = n_S
        else:
            n_all = n_S * n_iter
        if log_opt_strategy:
            # save optimal strategy to pickle file
            n_repeats = int(n_all / (m_g + 2))
            column_names = ['Data subject', 'Attribute']
            df = pd.DataFrame(columns=column_names)
            array_opt_strategy = np.stack(list_opt_strategy)
            sum_array_opt_strategy = np.sum(array_opt_strategy, axis=1)
            sort_order = (sum_array_opt_strategy, array_attack)
            for j in range(m_g+2):
                sort_order = (array_opt_strategy[:, j],) + sort_order  # have to include a comma for a single-value tuple
            order_subject = np.lexsort(sort_order)
            print('number of attacked subjects: ' + str(sum(array_attack)))
            rank_subject = order_subject.argsort()
            for i_subject in range(n_all):
                for i_attribute in range(m_g + 2):
                    if not list_opt_strategy[i_subject][i_attribute]:
                        for k in range(n_repeats):
                            id_attribute = n_repeats * (m_g + 2) - (i_attribute * n_repeats + k)
                            new_row = {column_names[0]: rank_subject[i_subject] + 0.5, column_names[1]: id_attribute}
                            df = df.append(new_row, ignore_index=True)
            df.to_pickle(folder_result + 'optimal_strategy_' + str(scenario) + '.pkl')
        dataset = pd.DataFrame({'privacy': array_privacy,
                                'utility': array_utility,
                                'defender_optimal': array_optimal_payoff,
                                'attacker_optimal': array_optimal_attacker_payoff,
                                'success_rate': array_success_rate})
        dataset2 = pd.DataFrame({'usefulness': array_usefulness,
                                'fairness_wrt_payoff': array_fairness_wrt_payoff,
                                'fairness_wrt_privacy': array_fairness_wrt_privacy,
                                'fairness_wrt_utility': array_fairness_wrt_utility,
                                'fairness_wrt_usefulness': array_fairness_wrt_usefulness})
        dataset.to_pickle(pickle_filename)
        dataset2.to_pickle(pickle_filename2)
        f.write('Average strategy: ' + str(sum_opt_strategy/n_all) + '\n')
        f.write('Average sharing rate: ' + str(np.mean(sum_opt_strategy / n_all)) + '\n')
        f.write('Data subjects\' average payoff: ' + str(np.mean(array_optimal_payoff)) + '\n')
        f.write('Data subjects\' average privacy: ' + str(np.mean(array_privacy)) + '\n')
        f.write('Data subjects\' average utility: ' + str(np.mean(array_utility)) + '\n')
        f.write('Data subjects\' average success rate: ' + str(np.mean(array_success_rate)) + '\n')
        f.write('Data usefulness: ' + str(np.mean(array_usefulness)) + '\n')
        f.write('Fairness wrt payoff: ' + str(np.mean(array_fairness_wrt_payoff)) + '\n')
        f.write('Fairness wrt privacy: ' + str(np.mean(array_fairness_wrt_privacy)) + '\n')
        f.write('Fairness wrt utility: ' + str(np.mean(array_fairness_wrt_utility)) + '\n')
        f.write('Fairness wrt usefulness: ' + str(np.mean(array_fairness_wrt_usefulness)) + '\n')
        elapsed2 = (time.time() - start2)
        f.write("Time used: " + str(elapsed1) + " seconds (loading) + " + str(elapsed2) + " seconds (computing).\n")
        f.write('\n')
        f.write('Configurations:\n')
        f.write('n_I: ' + str(n_I) + '\n')
        f.write('n_S: ' + str(n_S) + '\n')
        f.write('n_G: ' + str(n_G) + '\n')
        f.write('n_iter: ' + str(n_iter) + '\n')
        f.write('theta_p: ' + str(theta_p) + '\n')
        f.write('tol: ' + str(tol) + '\n')
        f.write('cost: ' + str(cost) + '\n')
        f.write('loss: ' + str(loss) + '\n')
        f.write('missing_level: ' + str(missing_level) + '\n')
        f.write('random_masking_rate: ' + str(random_masking_rate) + '\n')
        f.write('base_utility: ' + str(base_utility) + '\n')
        f.write('alpha: ' + str(alpha) + '\n')
        f.write('log_opt_strategy: ' + str(log_opt_strategy) + '\n')
        f.write('save_dic: ' + str(save_dic) + '\n')
        f.write('load_dic: ' + str(load_dic) + '\n')
        f.write('short_memory_dic: ' + str(short_memory_dic) + '\n')
        f.write('short_local_memory_dic: ' + str(short_local_memory_dic) + '\n')
        f.write('save_iter: ' + str(save_iter) + '\n')
        f.write('folder_result: ' + folder_result + '\n')
        f.close()
        if save_dic:
            # save dictionaries
            dic_names = ['dist', 'score', 'score_solo', 'attack', 'surname']
            dics = [dic_dist, dic_score, dic_score_solo, dic_attack, dic_surname]
            for i in range(5):
                f1 = open(folder_result + 'dic_s' + str(scenario) + '_' + dic_names[i] + '.pkl', 'wb')
                pickle.dump(dics[i], f1, protocol=pickle.HIGHEST_PROTOCOL)
                f1.close()
