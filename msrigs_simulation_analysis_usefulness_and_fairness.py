# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Analysis on Usefulness and Fairness Using Simulated Datasets
# Copyright 2018-2021 Zhiyu Wan, HIPLAB, Vanderilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Scikit-learn 0.23.2, Pandas 1.1.3, Matplotlib 3.3.1,
# Seaborn 0.11.0, and SciPy 1.5.2
# Update history:
# April 25, 2021: Add sharing rate function
# May 11, 2021: Fix bugs about weighted_entropy for stat types 14, 15.2, and 15.3.
import numpy as np
import pandas as pd
import os.path
import msrigs_functions as sf
from scipy.stats import entropy
from scipy.stats import skew
from scipy.special import rel_entr
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import wasserstein_distance
from self_implemented_distance import variational_distance, earth_movers_distance

# configuration
id_exp = '2058'  # ID for the set of experiments
n_S = 1000  # size of the sensitive dataset (<=90000) (default: 1000)
start_iter = 0  # start from a particular iteration (default: 0)
n_iter = 100  # (default: 100)
method = 2  # (default: 2)
m_g = 12  # (<=16) (default: 12)
weight = np.concatenate((np.ones(2) * 1, np.ones(m_g)), axis=None)
missing_level = 0.3  # (default: 0.3)
over_confident = 0  # (default: 0)
alter_weight = 0  # *0: Based on information entropy. 1: Uniform. 2: Special (the weight of 1st 2 geno features is 10x).
algorithm = 1  # *0: greedy algorithm. 1: brute-force algorithm.
pruning = 1  # (default: 1)

# choose a type for statistics
stat_type = 14  # 1:sharing rate, 2:number of distinct values, 3:entropy, 4:mean, 5:std, 6:skewness, 7:gini, 8:min, 9:Q1,
# 10:median, 11:Q3, 12:max, 13:KL divergence, 13.2:variational distance, 13.3:Earth mover’s distance (EMD),
# 14:pearson/spearman, 15:group-wise KL divergence, 15.2:group-wise variational distance, 15.3:group-wise Earth mover’s distance.
# 15.5:group-wise payoff, 15.6:group-wise privacy, 15.7:group-wise utility, 15.8: sharing rate

# choose a scenario
#scenario = 5  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking. 3.1: custom masking.
# 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.

# setting for stat_type 15
#explicit_usage = 0
#targeted_attribute = 1  # 0:Birth year, 1:State

save_iter = [False, False, False, False, False, True, False, False]  # save iteration for each scenario
column_names = ['defender_optimal', 'privacy', 'utility']

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

if algorithm == 0:
    scenarios = [4, 3, 3.1, 0]
else:
    scenarios = [5]

for i_scenario in range(len(scenarios)):
    scenario = scenarios[i_scenario]
    for explicit_usage in range(2):
        for targeted_attribute in range(2):
            if stat_type < 14:  # one-variable summary statistics
                a = np.empty([n_iter, 2+m_g+1])
                # load target dataset
                for i in range(start_iter, start_iter + n_iter):
                    S = np.genfromtxt(folder_result + 'target_data/i' + str(i) + '.csv', delimiter=',',
                                      skip_header=1).astype(int)
                    weighted_entropy = np.genfromtxt(folder_result + 'weighted_entropy/i' + str(i) + '.csv',
                                                     delimiter=',')
                    if scenario > 0:
                        array_opt_strategy = np.genfromtxt(folder_result + 'opt_strategy/s' + str(scenario) + '_i' + str(i) + '.csv', delimiter=',')
                    for j in range(2+m_g):
                        S1 = S[:, j]
                        # From Age to Birth_year
                        if j == 0:
                            S1 = 2020-S1
                        if scenario > 0:
                            array_opt_strategy1 = array_opt_strategy[:, j]
                            S_output = S1[array_opt_strategy1 > 0]
                        else:
                            S_output = S1
                        if stat_type == 1:  # sharing rate
                            S_stat = S_output.size/n_S
                            S1_stat = S1.size/n_S
                        elif stat_type == 2:  # number of distinct values
                            S_stat = np.unique(S_output).size
                            S1_stat = np.unique(S1).size
                        elif stat_type == 3:  # entropy
                            values, counts = np.unique(S_output, return_counts=True)
                            S_stat = entropy(counts)
                            values1, counts1 = np.unique(S1, return_counts=True)
                            S1_stat = entropy(counts1)
                        elif stat_type == 4:  # mean
                            S_stat = np.mean(S_output)
                            S1_stat = np.mean(S1)
                        elif stat_type == 5:  # standard deviation
                            S_stat = np.std(S_output)
                            S1_stat = np.std(S1)
                        elif stat_type == 6:  # skewness
                            S_stat = skew(S_output)
                            S1_stat = skew(S1)
                        elif stat_type == 7:  # gini coefficient
                            S_stat = sf.gini(S_output.astype(np.float))
                            S1_stat = sf.gini(S1.astype(np.float))
                        elif stat_type == 8:  # min
                            S_stat = np.amin(S_output)
                            S1_stat = np.amin(S1)
                        elif stat_type == 9:  # Q1
                            S_stat = np.quantile(S_output, .25)
                            S1_stat = np.quantile(S1, .25)
                        elif stat_type == 10:  # median
                            S_stat = np.median(S_output)
                            S1_stat = np.median(S1)
                        elif stat_type == 11:  # Q3
                            S_stat = np.quantile(S_output, .75)
                            S1_stat = np.quantile(S1, .75)
                        elif stat_type == 12:  # max
                            S_stat = np.amax(S_output)
                            S1_stat = np.amax(S1)
                        elif np.floor(stat_type) == 13:  # KL divergence (relative entropy), or variational distance, or EMD
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
                                p = counts/np.sum(counts)
                                q = counts2/np.sum(counts2)
                                if stat_type == 13:  # reverse KL divergence
                                    S_stat = np.sum(rel_entr(q, p))
                                elif stat_type == 13.2:  # self-implemented variational distance
                                    S_stat = variational_distance(p, q)
                                elif stat_type == 13.3:  # self-implemented EMD
                                    t = (j == 1) + 0  # 0: numerical, 1: categorical
                                    S_stat = earth_movers_distance(p, q, t)
                                elif stat_type == 13.4:  # EMD in scipy
                                    S_stat = wasserstein_distance(p, q)
                        else:
                            S_stat = 0
                            print('Invalid stat_type!')
                        if np.floor(stat_type) < 13:
                            if stat_type >= 4 and stat_type <= 12:  # numerical attributes
                                a[i, j] = np.abs(S_stat - S1_stat)
                            else:
                                a[i, j] = np.abs(S_stat - S1_stat) / S1_stat
                        else:
                            a[i, j] = S_stat
                    # calculate weighted sum
                    if stat_type >= 4 and stat_type <= 12:  #numerical attributes
                        a_temp = a[i, 0:(2 + m_g)]
                        a_temp[1] = 0
                        a[i, 2 + m_g] = np.sum(a_temp) / (a_temp.size - 1)
                    else:
                        a[i, 2 + m_g] = np.dot(a[i, 0:(2 + m_g)], weighted_entropy) / np.sum(weighted_entropy)
                # save analysis result
                if not os.path.exists(folder_result + 'analysis_result'):
                    os.mkdir(folder_result + 'analysis_result')
                filename = folder_result + 'analysis_result/type' + str(stat_type) + '_s' + str(scenario) + '.txt'
                f = open(filename, 'w')
                array_output = np.empty([2, 2+m_g+1])
                for j in range(2+m_g+1):
                    a1 = a[:, j]
                    a1_mean = np.mean(a1)
                    a1_std = np.std(a1)
                    array_output[0, j] = a1_mean
                    array_output[1, j] = a1_std
                    print('j=' + str(j) + ': mean = ' + str(a1_mean) + ', SD= ' + str(a1_std))
                    f.write('j=' + str(j) + ': mean = ' + str(a1_mean) + ', SD= ' + str(a1_std) + '\n')
                f.close()
                np.savetxt(folder_result + 'analysis_result/type' + str(stat_type) + '_s' + str(scenario) + '.csv', array_output,
                           delimiter=',')

            elif stat_type == 14:  # two-variable summary statistics
                a = np.empty([n_iter, 2 + m_g, 2 + m_g])
                b = np.empty([n_iter, 2 + m_g + 1])
                # load target dataset
                for i in range(start_iter, start_iter + n_iter):
                    S = np.genfromtxt(folder_result + 'target_data/i' + str(i) + '.csv', delimiter=',', skip_header=1)
                    weighted_entropy = np.genfromtxt(folder_result + 'weighted_entropy/i' + str(i) + '.csv',
                                                     delimiter=',')
                    if scenario > 0:
                        array_opt_strategy = np.genfromtxt(
                            folder_result + 'opt_strategy/s' + str(scenario) + '_i' + str(i) + '.csv', delimiter=',')
                    for j in range(2 + m_g):
                        for k in range(2 + m_g):
                            if j == k:
                                S_stat = 0
                                S1_stat = 0
                            else:
                                S1 = S[:, j]
                                S2 = S[:, k]
                                # From Age to Birth_year
                                if j == 0:
                                    S1 = 2020 - S1
                                if k == 0:
                                    S2 = 2020 - S2
                                if scenario > 0:
                                    array_opt_strategy1 = array_opt_strategy[:, j]
                                    array_opt_strategy2 = array_opt_strategy[:, k]
                                    S1_output = S1[np.logical_and(array_opt_strategy1 > 0, array_opt_strategy2 > 0)]
                                    S2_output = S2[np.logical_and(array_opt_strategy1 > 0, array_opt_strategy2 > 0)]
                                else:
                                    S1_output = S1
                                    S2_output = S2
                                if stat_type == 14:  # pearson correlation
                                    if S1_output.size == 0 or S2_output.size == 0:
                                        S_stat, pval = 0, 0
                                    elif S1_output.size == 1 or S2_output.size == 1:
                                        S_stat, pval = 1, 0
                                    else:
                                        #S_stat, pval = pearsonr(S1_output, S2_output)
                                        S_stat, pval = spearmanr(S1_output, S2_output)
                                    if S1.size == 0 or S2.size == 0:
                                        S1_stat, pval1 = 0, 0
                                    elif S1.size == 1 or S2.size == 1:
                                        S1_stat, pval1 = 1, 0
                                    else:
                                        S1_stat, pval1 = spearmanr(S1, S2)
                                else:
                                    S_stat, pval = 0, 0
                                    S1_stat, pval1 = 0, 0
                                    print('Invalid stat_type!')
                            if np.isnan(np.abs(S_stat - S1_stat)):
                                a[i, j, k] = 0
                            else:
                                a[i, j, k] = np.abs(S_stat - S1_stat)
                    b[i, 0:(2+m_g)] = np.mean(a[i, :, :], axis=1) * 13
                    b[i, 2+m_g] = np.dot(b[i, 0:(2+m_g)], weighted_entropy) / np.sum(weighted_entropy)

                # save analysis result
                if not os.path.exists(folder_result + 'analysis_result'):
                    os.mkdir(folder_result + 'analysis_result')
                filename = folder_result + 'analysis_result/type' + str(stat_type) + '_s' + str(scenario) + '.txt'
                f = open(filename, 'w')
                avg_array_output = np.mean(b, axis=0)
                print(str(avg_array_output))
                f.write(str(avg_array_output))
                f.close()
                np.savetxt(folder_result + 'analysis_result/type' + str(stat_type) + '_s' + str(scenario) + '.csv', avg_array_output, delimiter=',')

            elif np.floor(stat_type) == 15:  # one-variable summary statistics for each demographic group (one attribute)
                group_hr_birth_year = list(range(1910, 2000, 10))  # [1910, 1920, 1930, 1940, 1950, 1960, 1970, ..., 1990]
                group_hr_state = list(range(11, 51, 10))  # [11, 21, 31, 41]

                if targeted_attribute == 0:
                    group_hr = group_hr_birth_year
                else:
                    group_hr = group_hr_state
                n_groups = len(group_hr) + 1
                if stat_type > 15.4:  # additive utility, privacy, and payoff
                    a = np.zeros([n_iter, n_groups + 2, 2])  # [...] * [..., W_Avg, (1-Gini)] * [metric, size]
                    if algorithm == 1 and scenario >= 5:
                        folder_result = folder_result.replace('Violin/m' + str(method), 'Violin_bf')
                    if stat_type < 15.8:
                        if not save_iter[np.floor(scenario).astype(int)]:
                            dataset = pd.read_pickle(folder_result + 'result_s' + str(scenario) + '.pickle')
                        ii = np.round((stat_type - 15.5) * 10).astype(int)
                        if save_iter[np.floor(scenario).astype(int)]:
                            shaped_data = np.empty([n_iter, n_S])
                            for k in range(n_iter):  # for each iteration
                                dataset = pd.read_pickle(folder_result + 'result_s' + str(scenario) + '_i' + str(k) + '.pickle')
                                data = np.array(dataset[column_names[ii]])
                                shaped_data[k, :] = data.reshape(1, -1)
                        else:
                            data = np.array(dataset[column_names[ii]])
                            shaped_data = np.reshape(data, (n_iter, n_S))
                else:  # utility for the dataset
                    a = np.zeros([n_iter, n_groups + 5, 2 + m_g + 2])  # [.]*[., W_Avg, STD, Gini, Gini2, entropy]*[., W_Avg, size]
                # load target dataset
                for i in range(start_iter, start_iter + n_iter):
                    S = np.genfromtxt(folder_result + 'target_data/i' + str(i) + '.csv', delimiter=',',
                                      skip_header=1).astype(int)
                    weighted_entropy = np.genfromtxt(folder_result + 'weighted_entropy/i' + str(i) + '.csv', delimiter=',')
                    if scenario > 0:
                        array_opt_strategy = np.genfromtxt(
                            folder_result + 'opt_strategy/s' + str(scenario) + '_i' + str(i) + '.csv', delimiter=',').astype(int)
                    elif scenario == 0:
                        array_opt_strategy = np.ones([n_S, 2+m_g]).astype(int)
                    S_targeted = S[:, targeted_attribute]
                    # From Age to Birth_year
                    if targeted_attribute == 0:
                        S_targeted = 2020 - S_targeted
                    memberships = np.copy(S_targeted)
                    for j in range(n_S):  # for each person
                        index_group = 0
                        for k in range(n_groups-1):  # for each group (excluding the 1st)
                            if S_targeted[j] < group_hr[k]:
                                break
                            else:
                                index_group += 1
                        memberships[j] = index_group

                    for j in range(n_groups):  # for each group
                        selection = memberships == j
                        # save group size
                        if explicit_usage == 1:  # explicit usage
                            array_opt_strategy_targeted = array_opt_strategy[:, targeted_attribute]
                            a[i, j, -1] = np.sum(np.logical_and(selection, array_opt_strategy_targeted > 0))
                        else:
                            a[i, j, -1] = np.sum(selection)
                        if stat_type > 15.4:  # additive utility, privacy, and payoff
                            if explicit_usage == 1:  # explicit usage
                                array_opt_strategy_targeted = array_opt_strategy[:, targeted_attribute]
                                if np.sum(np.logical_and(selection, array_opt_strategy_targeted > 0)) == 0:
                                    print('group size is zero!')
                                    a[i, j, 0] = 0
                                else:
                                    if stat_type == 15.8:  # sharing rate
                                        a[i, j, 0] = np.mean(array_opt_strategy[
                                                             np.logical_and(selection, array_opt_strategy_targeted > 0),
                                                             :])
                                    else:
                                        a[i, j, 0] = np.mean(
                                            shaped_data[i, np.logical_and(selection, array_opt_strategy_targeted > 0)])

                            else:
                                if stat_type == 15.8:  # sharing rate
                                    a[i, j, 0] = np.mean(array_opt_strategy[selection, :])
                                else:
                                    a[i, j, 0] = np.mean(array_opt_strategy[selection, :])
                            continue

                        for k in range(2 + m_g):  # for each attribute
                            S1 = S[:, k]
                            # From Age to Birth_year
                            if k == 0:
                                S1 = 2020 - S1
                            if scenario > 0:
                                array_opt_strategy1 = array_opt_strategy[:, k]
                                if explicit_usage == 1:  # explicit usage
                                    array_opt_strategy_targeted = array_opt_strategy[:, targeted_attribute]
                                    S_output = S1[np.logical_and.reduce((array_opt_strategy1 > 0, selection,
                                                                         array_opt_strategy_targeted > 0))]  # shared data in this group
                                else:
                                    S_output = S1[np.logical_and.reduce((array_opt_strategy1 > 0, selection))]  # shared data in this group
                            else:  # no protection
                                S_output = S1[selection]
                            S1 = S1[selection]  # original data in this group
                            if np.floor(stat_type) == 15:  # KL divergence
                                values, counts = np.unique(S1, return_counts=True)
                                counts2 = np.copy(counts)
                                for kk in range(values.size):
                                    counts2[kk] = np.count_nonzero(S_output == values[kk])
                                if np.sum(counts2) == 0:
                                    S_stat = 1
                                    print('Undefined distance!')
                                elif counts.size == 1:
                                    S_stat = 0
                                    print('One-point distribution!')
                                else:
                                    p = counts / np.sum(counts)
                                    q = counts2 / np.sum(counts2)
                                    if stat_type == 15:  # reverse KL divergence
                                        S_stat = np.sum(rel_entr(q, p))
                                    elif stat_type == 15.2:  # self-implemented variational distance
                                        S_stat = variational_distance(p, q)
                                    elif stat_type == 15.3:  # self-implemented EMD
                                        t = (k == 1) + 0  # 0: numerical, 1: categorical
                                        S_stat = earth_movers_distance(p, q, t)
                                    elif stat_type == 15.4:  # EMD in scipy
                                        S_stat = wasserstein_distance(p, q)
                            else:
                                S_stat = 0
                                print('Invalid stat_type!')
                            a[i, j, k] = S_stat

                # save analysis result
                if not os.path.exists(folder_result + 'analysis_result'):
                    os.mkdir(folder_result + 'analysis_result')
                # computer statistics for each iteration
                for i in range(start_iter, start_iter + n_iter):
                    if stat_type > 15.4:  # additive utility, privacy, and payoff
                        # calculate the weighted average
                        a[i, n_groups, 0] = np.dot(a[i, 0:n_groups, 0], a[i, 0:n_groups, -1]) / np.sum(a[i, 0:n_groups, -1])
                        a[i, n_groups + 1, 0] = 1 - sf.gini(a[i, 0:n_groups, 0])  # calculate the (1 - gini coefficient)
                        a[i, n_groups, -1] = np.mean(a[i, 0:n_groups, -1])  # calculate the average
                        a[i, n_groups + 1, -1] = 1 - sf.gini(a[i, 0:n_groups, -1])  # calculate the (1 - gini coefficient)
                    else:
                        weighted_entropy = np.genfromtxt(folder_result + 'weighted_entropy/i' + str(i) + '.csv',
                                                         delimiter=',')
                        for j in range(n_groups):  # for each group
                            # calculate the weighted_average
                            a[i, j, 2+m_g] = np.dot(a[i, j, 0:(2+m_g)], weighted_entropy) / np.sum(weighted_entropy)
                        for k in range(2 + m_g + 1):  # for each attribute + 1
                            # calculate the average
                            a[i, n_groups, k] = np.dot(a[i, 0:n_groups, k], a[i, 0:n_groups, -1]) / np.sum(a[i, 0:n_groups, -1])
                            a[i, n_groups + 1, k] = np.std(a[i, 0:n_groups, k])  # calculate the standard deviation
                            a[i, n_groups + 2, k] = sf.gini(a[i, 0:n_groups, k])  # calculate the gini coefficient
                            a[i, n_groups + 3, k] = sf.gini(1-a[i, 0:n_groups, k])  # calculate the gini coefficient of (1 - distance)
                        a[i, n_groups, -1] = np.mean(a[i, 0:n_groups, -1])  # calculate the average
                        a[i, n_groups + 1, -1] = np.std(a[i, 0:n_groups, -1])  # calculate the standard deviation
                        a[i, n_groups + 2, -1] = sf.gini(a[i, 0:n_groups, -1])  # calculate the gini coefficient
                        a[i, n_groups + 4, 0:(2 + m_g)] = weighted_entropy
                        a[i, n_groups + 4, 2 + m_g] = np.mean(weighted_entropy)
                avg_array_output = np.mean(a, axis=0)
                np.savetxt(folder_result + 'analysis_result/type' + str(stat_type) + '_s' + str(scenario)
                           + '_attr' + str(targeted_attribute) + '_exp' + str(explicit_usage) + '.csv',
                           avg_array_output, delimiter=',')
