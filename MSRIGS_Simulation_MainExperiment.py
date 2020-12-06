# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.0
# Component: MSRIGS Using Simulated Datasets (Data subjects have different strategies)
# (Previous development name: A not so simple version of the simulation script)
# Copyright 2018-2020 Zhiyu Wan
# HIPLAB, Vanderilt University
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
import numpy as np
import time
import pandas as pd
import MSRIGS_Functions as sf
import os.path
import pickle

# configuration
id_exp = '2046'  # ID for the set of experiments
n_f = 20  # number of firstnames (not used)
n_I = 20000  # size of the identified dataset (<=90000) (default: 20000)
n_S = 1000  # size of the sensitive dataset (<=90000) (default: 1000)
n_G = 20000  # size of the genetic genealogy dataset (<=90000) (default: 20000)
rate_s = 0.6  # rate of sensitive (not used)
loss = 150  # (default: 150)
cost = 10  # (default: 10)
total_utility = 100  # (default: 100)
start_iter = 0  # start from a particular iteration (default: 0)
n_iter = 100  # (default: 100)
theta_p = 0.5  # (default: 0.5)
method = 2  # (default: 2)
m_g = 12  # (<=16) (default: 12)
weight = np.concatenate((np.ones(2) * 1, np.ones(m_g)), axis=None)
missing_level = 0.3  # (default: 0.3)
over_confident = 0  # (default: 0)
alter_weight = 0  # *0: Based on information entropy. 1: Uniform. 2: Special (the weight of 1st 2 geno features is 10x).
algorithm = 0  # *0: greedy algorithm. 1: brute-force algorithm.
pruning = 1  # (default: 1)
participation_rate = 0.05  # (default: 0.05)
random_mask_rate = 0.8  # (default: 0.8)
log_opt_strategy = False  # log optimal strategies for all data subjects (default: False)
save_dic = False  # save all dictionaries into files in the end (default: False)
load_dic = False  # load all global dictionaries (dic_dist, dic_score, dic_score_solo) in the beginning (default: False)
short_memory_dic = True  # refresh dictionaries in each iteration (no need to load/save dictionaries) (default: False)
short_local_memory_dic = True  # refresh local dictionaries (attack, surname) for each subject (default: False)
save_iter = True  # save results in each iteration (different file names) (default: False)

# choose a scenario
scenario = 5  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking. 4: opt-in game.
# 5: masking game. 6: no-attack masking game. 7: one-stage masking game.

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
        sum_opt_strategy = np.zeros(m_g + 2)
        if log_opt_strategy:
            array_attack = np.empty(n_iter * n_S).astype(bool)
            list_opt_strategy = []
        pickle_filename = folder_result + 'result_s' + str(scenario) + '.pickle'
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
            sum_opt_strategy = np.zeros(m_g + 2)
            if log_opt_strategy:
                array_attack = np.empty(n_S).astype(bool)
                list_opt_strategy = []
            pickle_filename = folder_result + 'result_s' + str(scenario) + '_i' + str(i) + '.pickle'
            filename = folder_result + 'log_s' + str(scenario) + '_i' + str(i) + '.txt'
            f = open(filename, 'w')
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
        if short_memory_dic:
            dic_dist = {}
            dic_score = {}
            dic_score_solo = {}
        dic_attack = {}
        dic_surname = {}
        for j in range(n_S):
            print('j: ', j)
            s = S[j, :]
            #print(s)
            if short_local_memory_dic:
                dic_attack = {}
                dic_surname = {}
            (opt_payoff, opt_attacker_payoff, opt_attack, opt_success_rate, opt_utility, opt_strategy) = \
                sf.optimal_defense(s, I, G, weighted_entropy, m_g, dic_attack, dic_surname, loss, cost, scenario,
                                   total_utility, theta_p, over_confident, mu, method, tol, dic_dist,
                                   dic_score_solo, dic_score, T_Max, inv_Ne, participation_rate, random_mask_rate,
                                   algorithm, pruning, I_selection)
            if save_iter:
                index = j
            else:
                index = i * n_S + j
            array_optimal_payoff[index] = opt_payoff
            array_optimal_attacker_payoff[index] = opt_attacker_payoff
            array_privacy[index] = 1 - opt_success_rate * opt_attack
            array_utility[index] = opt_utility
            sum_opt_strategy += opt_strategy
            if log_opt_strategy:
                array_attack[index] = opt_attack
                list_opt_strategy.append(opt_strategy)
            f.write(
                '{}-{}: {} {:f} {:d} {}\n'.format(i, j, np.array(list(map(int, opt_strategy))), opt_payoff, opt_attack,
                                                  opt_success_rate))
        if not save_iter and i < (start_iter + n_iter - 1):  # not the last iteration and not in save-iteration mode
            continue
        if save_iter:
            n_all = n_S
        else:
            n_all = n_S * n_iter
        if log_opt_strategy:
            # save optimal strategy to file
            n_repeats = int(n_all / (m_g + 2))
            column_names = ['Data subject', 'Attribute']
            df = pd.DataFrame(columns=column_names)
            array_opt_strategy = np.stack(list_opt_strategy)
            sum_array_opt_strategy = np.sum(array_opt_strategy, axis=1)
            sort_order = (sum_array_opt_strategy, array_attack)
            for i in range(m_g+2):
                sort_order = (array_opt_strategy[:, i],) + sort_order  # have to include a comma for a single-value tuple
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
                                'attacker_optimal': array_optimal_attacker_payoff})
        dataset.to_pickle(pickle_filename)
        f.write('Average strategy: ' + str(sum_opt_strategy/n_all) + '\n')
        f.write('Data subjects\' average payoff: ' + str(np.mean(array_optimal_payoff)) + '\n')
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
