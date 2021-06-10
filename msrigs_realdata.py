# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: MSRIGS Using Real Datasets
# Copyright 2017-2021 Zhiyu Wan, HIPLAB, Vanderilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Scikit-learn 0.23.2, Pandas 1.1.3, Matplotlib 3.3.1,
# Seaborn 0.11.0, and SciPy 1.5.2
# Update history:
# 20181104: remove records in Ysearch those have less than 20 non-zero loci
# 20200429: 1)infer surname of one record. 2)find the best sharing policy
# 20200507: convert to numpy objects
# 20200508: 1)add Venter genome. 2)update to clean4_matrix_all.txt. 3)delete dic_attack. 4)plot figures. 5) add entropy.
# 20201020: Delete YBase dataset.
# 20210512: sharing rate is changed to 0.15 in scenario 3.
# 20210513: fix the random number seed.

import time
import numpy as np
import pandas as pd
import msrigs_functions as sf
import os.path
import pickle

# Configuration
id_exp = '2058'  # ID for the set of experiments
folder_result = 'Results' + id_exp + '/realdata/'

# parameter settings
pop_cal_71_male = 157681  # Year 2018
pop_cal_71_male_venter = 2
pop_71_male = 1522210  # Year 2018
est_pop_71_male_venter = 19
pop_cal_male = 19663577  # Year 2018
est_pop_cal_male_venter = 249
pop_male = 161128679  # Year 2018
est_pop_male_venter = 2044

# choose a scenario
scenario = 3  # 0: no protection. (1: no genomic data sharing.) 2: random opt-in. 3: random masking. 4: opt-in game.
# 5: masking game. 6: no-attack masking game. (7: one-stage masking game.)

demo = "all"
if demo == "all":
    pop1 = pop_cal_71_male_venter
    pop0 = pop_cal_71_male
    demo_utility = 44
if demo == "age":
    pop1 = est_pop_71_male_venter
    pop0 = pop_71_male
    demo_utility = 24
elif demo == "state":
    pop1 = est_pop_cal_male_venter
    pop0 = pop_cal_male
    demo_utility = 20
elif demo == "none":
    pop1 = est_pop_male_venter
    pop0 = pop_male
    demo_utility = 0

# jaro_theta = 1  # (default: 0.9)
N_Top = 10
min_marker_com = 1  # lower bound on the number of common markers
min_marker = 17  # lower bound on the number of available markers (default: 20)
T_Max = 200
Ne = 10000
inv_Ne = 1.0 / Ne
tol = 0.2  # tolerance (0.1)
# redundancy = 49

loss = 150
cost = 10
total_utility = 100
geno_utility = 56
theta_p = 0.5
m_g = 100  # default: 16
over_confident = 0
pruning = 1
participation_rate = 0.05
random_mask_rate = 0.15
save_dic = 0

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
pickle_filename31 = folder_result + 'payoff_utility_confidence_s' + str(scenario) + '_1.pickle'
pickle_filename32 = folder_result + 'payoff_utility_confidence_s' + str(scenario) + '_2.pickle'
pickle_filename33 = folder_result + 'payoff_utility_confidence_s' + str(scenario) + '_3.pickle'
filename = folder_result + 'log_s' + str(scenario) + '.txt'


def surname_inference(y1r, Y2, Ysearch_ID, selection, MU, tol, a2, dic_dist, dic_score_solo, dic_score, T_Max, inv_Ne):
    selection = selection.astype(bool)
    if np.sum(selection) == 0:  # no computation needed
        # print("No candidates!")
        inferred_surname = -1
        p = 0
        return inferred_surname, p
    y1r = y1r[selection]
    Y2 = Y2[:, selection]
    MU = MU[selection]
    n_match = np.sum(Y2 == y1r, axis=1)
    sorted_n_match = np.sort(n_match)[::-1]
    max_n_match = sorted_n_match[0]
    tol_n_match = int(max_n_match * (1 - tol))
    # cutoff_n_match = max(1, sorted_n_match[min(N_Top * (1 + redundancy), a2) - 1])
    # lb_n_match = max(tol_n_match, cutoff_n_match)  # lowerbound
    candidates = []
    for j in range(a2):
        if n_match[j] >= tol_n_match:  # lb_n_match
            # print("cand: "+Ysearch_ID[j])
            candidates.append(j)
    if len(candidates) == 0:
        # print("No candidates!")
        inferred_surname = -1
        p = 0
        return inferred_surname, p
    D = []
    for j in range(len(candidates)):
        y2r = Y2[candidates[j], :]
        y2r_nz = y2r != 0
        x_nz = y2r[y2r_nz] == y1r[y2r_nz]
        MU_nz = MU[y2r_nz]
        tuple_x_mu = (tuple(x_nz), tuple(MU_nz))
        if tuple_x_mu in dic_dist:
            dist = dic_dist[tuple_x_mu]
        else:
            dist = sf.etmrca(x_nz, MU_nz, T_Max, inv_Ne)
            dic_dist[tuple_x_mu] = dist
        D.append(dist)

    indexD = sorted(list(range(len(D))), key=lambda k: D[k])
    c_name = []  # candidate name
    print(Ysearch_ID[candidates[indexD[0]]])  # print summary result
    for j in range(min(N_Top, len(candidates))):
        c_name.append(Ysearch_ID[candidates[indexD[j]]])
    if len(candidates) == 1:
        y2r = Y2[candidates[0], :]
        y2r_nz = y2r != 0
        x_nz = y2r[y2r_nz] == y1r[y2r_nz]
        MU_nz = MU[y2r_nz]
        tuple_x_mu = (tuple(x_nz), tuple(MU_nz))
        if tuple_x_mu in dic_score_solo:
            score = dic_score_solo[tuple_x_mu]
        else:
            score = sf.conf_score_solo(x_nz, MU_nz, T_Max, inv_Ne)
            dic_score_solo[tuple_x_mu] = score
    else:
        name1 = Ysearch_ID[candidates[indexD[0]]]
        for kk in range(1, min(N_Top, len(candidates))):
            name2 = Ysearch_ID[candidates[indexD[kk]]]
            if name1.upper() != name2.upper():
                break
        if name1.upper() == name2.upper():
            y2r = Y2[candidates[indexD[0]], :]
            y2r_nz = y2r != 0
            x_nz = y2r[y2r_nz] == y1r[y2r_nz]
            MU_nz = MU[y2r_nz]
            tuple_x_mu = (tuple(x_nz), tuple(MU_nz))
            if tuple_x_mu in dic_score_solo:
                score = dic_score_solo[tuple_x_mu]
            else:
                score = sf.conf_score_solo(x_nz, MU_nz, T_Max, inv_Ne)
                dic_score_solo[tuple_x_mu] = score
        else:
            y2r = Y2[candidates[indexD[0]], :]
            y2r_nz = y2r != 0
            x1_nz = y2r[y2r_nz] == y1r[y2r_nz]
            MU1_nz = MU[y2r_nz]
            y2r = Y2[candidates[indexD[kk]], :]
            y2r_nz = y2r != 0
            x2_nz = y2r[y2r_nz] == y1r[y2r_nz]
            MU2_nz = MU[y2r_nz]
            tuple_x1_mu1 = (tuple(x1_nz), tuple(MU1_nz))
            tuple_x2_mu2 = (tuple(x2_nz), tuple(MU2_nz))
            if (tuple_x1_mu1, tuple_x2_mu2) in dic_score:
                score = dic_score[(tuple_x1_mu1, tuple_x2_mu2)]
            else:
                score = sf.conf_score(x1_nz, x2_nz, MU1_nz, MU2_nz, T_Max, inv_Ne)
                dic_score[(tuple_x1_mu1, tuple_x2_mu2)] = score
    return c_name[0], score


def attack_SIG(s_real_name, loss, cost, inferred_surname, p, theta_p, over_confident):
    # Assume all people have same demo with Venter.
    attack = 0
    real_success_rate = 0
    # age and state
    group_size = pop0  # year 2018
    real_success_rate1 = 1 / group_size
    payoff1 = loss * real_success_rate1 - cost
    real_payoff1 = max(payoff1, 0)

    # Assume group size will never be zero.
    # age, state, and inferred surname
    if inferred_surname != -1:
        str_same = s_real_name.upper() == inferred_surname.upper()
    else:
        str_same = False
    if str_same:
        if s_real_name.upper() == "VENTER":
            group_size = pop1
        else:
            group_size = 0
    if group_size == 0 or p < theta_p:  # use age and state instead (wrong inference or no inference)
        real_success_rate = real_success_rate1
        if payoff1 > 0:
            attack = 1
        real_payoff = real_payoff1
    else:
        if over_confident == 1:
            success_rate = 1 / group_size  # expected success rate
        else:
            success_rate = 1 / group_size * p
        payoff = loss * success_rate - cost  # expected payoff
        if payoff > payoff1:  # use age, state and inferred surname
            if payoff > 0:
                attack = 1
                if str_same:
                    real_success_rate = 1 / group_size
                real_payoff = loss * real_success_rate - cost  # not expected_payoff
            else:
                real_payoff = 0
        else:  # use age and state instead
            real_success_rate = real_success_rate1
            if payoff1 > 0:
                attack = 1
            real_payoff = real_payoff1
    return int(str_same), real_success_rate, attack, real_payoff


def optimal_defense(s, s_real_name, G, G_surname, w_entropy, m_g, dic_attack, dic_surname, loss, cost, scenario,
                    total_utility, theta_p, over_confident,
                    mu, tol, n_G, dic_dist, dic_score_solo, dic_score, T_Max, inv_Ne,
                    participation_rate, random_mask_rate, pruning):
    # Only mask genomic attributes. No scenario 1 and no scenario 7.
    if scenario == 3:  # scenario 3: random masking
        random_geno = np.random.choice([0, 1], m_g, p=[1 - random_mask_rate, random_mask_rate])
        (inferred_surname, p) = surname_inference(s, G, G_surname, random_geno, mu[0:m_g], tol, n_G, dic_dist,
                                                  dic_score_solo, dic_score, T_Max, inv_Ne)
        (_, success_rate, attack, attacker_payoff) = \
            attack_SIG(s_real_name, loss, cost, inferred_surname, p, theta_p, over_confident)
        defender_loss = attacker_payoff + attack * cost
        utility = (np.dot(w_entropy[0:m_g], random_geno) / np.sum(w_entropy[0:m_g]) * geno_utility +
                   demo_utility) / total_utility * 1.0
        defender_benefit = total_utility * utility  # compute the benefit
        defender_payoff = defender_benefit - defender_loss
        optimal_attack = attack
        optimal_utility = utility
        optimal_success_rate = success_rate
        optimal_payoff = defender_payoff
        optimal_attacker_payoff = attacker_payoff
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, random_geno

    # publish all data
    all_geno = np.ones(m_g).astype(int)
    null_geno = np.zeros(m_g).astype(int)
    optimal_geno = all_geno
    tuple_geno = tuple(all_geno)
    if tuple_geno in dic_surname:
        (inferred_surname, p) = dic_surname[tuple_geno]
    else:
        (inferred_surname, p) = surname_inference(s, G, G_surname, all_geno, mu[0:m_g],
                                                  tol, n_G, dic_dist, dic_score_solo, dic_score, T_Max, inv_Ne)
        dic_surname[tuple_geno] = (inferred_surname, p)
    tuple_demo = (inferred_surname, p)
    if tuple_demo in dic_attack:
        (_, success_rate, attack, attacker_payoff) = dic_attack[tuple_demo]
    else:
        (_, success_rate, attack, attacker_payoff) = \
            attack_SIG(s_real_name, loss, cost, inferred_surname, p, theta_p, over_confident)
        dic_attack[tuple_demo] = (_, success_rate, attack, attacker_payoff)
    defender_loss = attacker_payoff + attack * cost
    utility = (geno_utility + demo_utility) / total_utility * 1.0
    defender_benefit = geno_utility + demo_utility
    defender_payoff = defender_benefit - defender_loss
    optimal_utility = utility
    optimal_payoff = defender_payoff
    if scenario == 6 and attack == 1:  # in no-attack game, a strategy will not be optimal unless there is no attack
        optimal_payoff = -10000
    optimal_attacker_payoff = attacker_payoff
    optimal_attack = attack
    optimal_p = p
    optimal_success_rate = success_rate
    if scenario == 0 or scenario == 1:  # scenario 0: no protection, or scenario 1: no genomic data sharing
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_geno
    if scenario == 2:  # scenario 2: random opt-in
        if np.random.random_sample() >= participation_rate:  # choose to opt-out
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = 0
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_geno = null_geno
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_geno

    if scenario == 4:  # scenario 4: opt-in or opt-out
        if optimal_payoff <= 0:  # do not release anything
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = 0
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_geno = null_geno
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_geno

    # scenario 5: masking game, scenario 6: no-attack masking game
    current_geno = all_geno
    height_lattice = np.sum(current_geno).astype(int)
    for _ in range(height_lattice-1):
        child_optimal_payoff = -10000
        child_optimal_p = 2
        for i in range(len(current_geno)):
            if current_geno[i] == 0:
                continue
            child_geno = current_geno.copy()
            child_geno[i] -= 1
            tuple_geno = tuple(child_geno)
            if tuple_geno in dic_surname:
                (inferred_surname, p) = dic_surname[tuple_geno]
            else:
                (inferred_surname, p) = surname_inference(s, G, G_surname, child_geno, mu[0:m_g],
                                                          tol, n_G, dic_dist,
                                                          dic_score_solo, dic_score, T_Max, inv_Ne)
                dic_surname[tuple_geno] = (inferred_surname, p)
            tuple_demo = (inferred_surname, p)
            if tuple_demo in dic_attack:
                (right, success_rate, attack, attacker_payoff) = \
                    dic_attack[tuple_demo]
            else:
                (right, success_rate, attack, attacker_payoff) = \
                    attack_SIG(s_real_name, loss, cost, inferred_surname, p, theta_p, over_confident)
                dic_attack[tuple_demo] = (right, success_rate, attack, attacker_payoff)
            defender_loss = attacker_payoff + attack * cost
            utility = (np.sum(np.dot(w_entropy[0:m_g], child_geno)) / np.sum(w_entropy[0:m_g]) * geno_utility +
                       demo_utility) / total_utility * 1.0
            defender_benefit = total_utility * utility  # compute the benefit
            defender_payoff = defender_benefit - defender_loss
            print("child_geno: " + str(child_geno))
            print("defender_payoff: " + str(defender_payoff))
            print("defender_loss: " + str(defender_loss))
            print("utility: " + str(utility))
            print("defender_benefit: " + str(defender_benefit))
            print("inferred_surname: " + str(inferred_surname))
            print("p: " + str(p))
            if right:
                rightness = 'Correct'
            else:
                rightness = 'Wrong'
            rows_list1.append({'Payoff': defender_payoff, 'Utility': utility, 'Confidence score': p,
                              'Surname inference': rightness, 'Strategy': 'Searched'})
            rows_list2.append({'Payoff': defender_payoff, 'Utility': utility, 'Confidence score': p,
                               'Surname inference': rightness, 'Strategy': 'Searched'})
            rows_list3.append({'Payoff': defender_payoff, 'Utility': utility, 'Confidence score': p,
                               'Surname inference': rightness, 'Strategy': 'Searched'})
            if defender_payoff > child_optimal_payoff or\
                    (defender_payoff == child_optimal_payoff and
                     (2 * right - 1) * p < (2 * child_optimal_right - 1) * child_optimal_p):  # privacy measure
                child_optimal_geno = child_geno
                child_optimal_payoff = defender_payoff
                child_optimal_attacker_payoff = attacker_payoff
                child_optimal_utility = utility
                child_optimal_attack = attack
                child_optimal_success_rate = success_rate
                child_optimal_right = right
                child_optimal_p = p
                print("child_optimal_geno: " + str(child_optimal_geno))
                print("child_optimal_payoff: " + str(child_optimal_payoff))
                print("child_optimal_attacker_payoff: " + str(child_optimal_attacker_payoff))
                print("child_optimal_utility: " + str(child_optimal_utility))
                print("child_optimal_attack: " + str(child_optimal_attack))
                print("child_optimal_success_rate: " + str(child_optimal_success_rate))
                print("child_optimal_right: " + str(child_optimal_right))
                print("child_optimal_p: " + str(child_optimal_p))
            # in no-attack game, update the optimal strategy only if there is no attack
            if scenario == 6 and attack == 0 and \
                    (defender_payoff > optimal_payoff or
                     (defender_payoff == optimal_payoff and
                      (2 * right - 1) * p < (2 * child_optimal_right - 1) * optimal_p)):  # privacy measure
                optimal_payoff = defender_payoff
                optimal_attacker_payoff = attacker_payoff
                optimal_attack = attack
                optimal_success_rate = success_rate
                optimal_utility = utility
                optimal_right = right
                optimal_p = p
                optimal_geno = child_geno
                print("optimal_geno: " + str(optimal_geno))
                print("optimal_payoff: " + str(optimal_payoff))
                print("optimal_attacker_payoff: " + str(optimal_attacker_payoff))
                print("optimal_utility: " + str(optimal_utility))
                print("optimal_attack: " + str(optimal_attack))
                print("optimal_success_rate: " + str(optimal_success_rate))
                print("optimal_right: " + str(optimal_right))
                print("optimal_p: " + str(optimal_p))
        # record suboptimal points
        if child_optimal_right:
            rightness = 'Correct'
        else:
            rightness = 'Wrong'
        rows_list2.append({'Payoff': child_optimal_payoff, 'Utility': child_optimal_utility,
                           'Confidence score': child_optimal_p, 'Surname inference': rightness,
                           'Strategy': 'Suboptimal'})
        rows_list3.append({'Payoff': child_optimal_payoff, 'Utility': child_optimal_utility,
                           'Confidence score': child_optimal_p, 'Surname inference': rightness,
                           'Strategy': 'Suboptimal'})
        print("Final_child_optimal_geno: " + str(child_optimal_geno))
        print("Final_child_optimal_payoff: " + str(child_optimal_payoff))
        print("Final_child_optimal_attacker_payoff: " + str(child_optimal_attacker_payoff))
        print("Final_child_optimal_utility: " + str(child_optimal_utility))
        print("Final_child_optimal_attack: " + str(child_optimal_attack))
        print("Final_child_optimal_success_rate: " + str(child_optimal_success_rate))
        print("Final_child_optimal_right: " + str(child_optimal_right))
        print("Final_child_optimal_p: " + str(child_optimal_p))
        if scenario == 6:
            if pruning == 1 and (child_optimal_attack == 0 or not child_optimal_right):  # condition for pruning
                break
        else:  # scenario 5: masking game
            if child_optimal_payoff > optimal_payoff or \
                    (child_optimal_payoff == optimal_payoff and
                     (2 * child_optimal_right - 1) * child_optimal_p < (2 * optimal_right - 1) * optimal_p):  # privacy measure
                optimal_geno = child_optimal_geno
                optimal_payoff = child_optimal_payoff
                optimal_attacker_payoff = child_optimal_attacker_payoff
                optimal_utility = child_optimal_utility
                optimal_attack = child_optimal_attack
                optimal_success_rate = child_optimal_success_rate
                optimal_right = child_optimal_right
                optimal_p = child_optimal_p
                print("optimal_geno: " + str(optimal_geno))
                print("optimal_payoff: " + str(optimal_payoff))
                print("optimal_attacker_payoff: " + str(optimal_attacker_payoff))
                print("optimal_utility: " + str(optimal_utility))
                print("optimal_attack: " + str(optimal_attack))
                print("optimal_success_rate: " + str(optimal_success_rate))
                print("optimal_right: " + str(optimal_right))
                print("optimal_p: " + str(optimal_p))
            elif pruning == 1 and child_optimal_payoff < optimal_payoff and (not child_optimal_right or
                                                                             child_optimal_attack == 0):  # shortcut
                break
        current_geno = child_optimal_geno
    if optimal_payoff < 0:  # no data release, and no attack
        optimal_payoff = 0
        optimal_attacker_payoff = 0
        optimal_attack = 0
        optimal_success_rate = 0
        optimal_utility = 0
        optimal_geno = null_geno
    # record optimal strategy
    if optimal_right:
        rightness = 'Correct'
    else:
        rightness = 'Wrong'
    rows_list3.append({'Payoff': optimal_payoff, 'Utility': optimal_utility, 'Confidence score': optimal_p,
                       'Surname inference': rightness, 'Strategy': 'Optimal'})
    return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_geno


# main function
if __name__ == '__main__':
    starttime1 = time.time()
    # # Inputs
    #  input Ysearch dataset
    Y2 = []
    NZ_Y2 = []  # available Y-STRs (Nonzero)
    with open("data/Ysearch.txt", "r") as f:
        for line in f.readlines():
            loci = line.rstrip("\n").split(",")
            y1 = []
            nz = 0
            for locus in loci:
                ystr = int(locus)
                if ystr != 0:
                    nz += 1
                y1.append(ystr)
            if nz >= min_marker:
                Y2.append(y1)
            NZ_Y2.append(nz)
    Y2 = np.array(Y2)

    # input Ysearch ID
    Ysearch_ID = []
    with open("data/Ysearch_ID.txt", "r") as f:
        i = 0
        for line in f.readlines():
            if NZ_Y2[i] >= min_marker:
                Ysearch_ID.append(line.rstrip("\n"))
            i += 1
    Ysearch_ID = np.array(Ysearch_ID)

    # configure Venter data
    Sample_ID = ['Venter']
    # input dataset
    Y1 = []
    with open("data/Venter.txt", "r") as f:
        for line in f.readlines():
            loci = line.rstrip("\n").split(",")
            y1 = []
            for locus in loci:
                y1.append(int(locus))
            Y1.append(y1)
    Y1 = np.asarray(Y1)
    m_g = Y1.shape[1]
    # input MU
    with open("data/MU.txt", "r") as f:
        MU = []
        for line in f.readlines():
            mu = float(line.rstrip("\n")) / 1000
            if mu == 0:
                mu = 0.002  # set default to 0.002
            MU.append(mu)
    MU = np.array(MU)

    a1, b1 = Y1.shape  # number of rows (records) and columns (attributes) in sample
    a2, b2 = Y2.shape  # number of rows (records) and columns (attributes) in reference

    np.random.seed(0)  # reset random number generator for comparison

    # compute entropy
    entropy = []
    for j in range(m_g):
        c = Y2[:, j]
        entropy.append(sf.get_entropy(c))
    entropy = np.asarray(entropy)

    dic_dist = {}
    dic_score_solo = {}
    dic_score = {}
    array_optimal_payoff = np.empty(a1)
    array_optimal_attacker_payoff = np.empty(a1)
    array_privacy = np.empty(a1)
    array_utility = np.empty(a1)
    sum_opt_geno = np.zeros(m_g)
    rows_list1 = []
    rows_list2 = []
    rows_list3 = []
    endtime1 = time.time()
    print("Loading time is :" + str(endtime1 - starttime1) + " seconds.")
    starttime2 = time.time()
    dic_attack = {}
    dic_surname = {}
    for i in range(a1):
        print("No.", str(i + 1))
        print(Sample_ID[i])
        (opt_payoff, opt_attacker_payoff, opt_attack, opt_success_rate, opt_utility, opt_geno) = \
            optimal_defense(Y1[i, :], Sample_ID[i], Y2, Ysearch_ID, entropy, m_g, dic_attack, dic_surname, loss, cost,
                            scenario, total_utility, theta_p, over_confident, MU, tol, a2, dic_dist, dic_score_solo,
                            dic_score, T_Max, inv_Ne, participation_rate, random_mask_rate, pruning)
        array_optimal_payoff[i] = opt_payoff
        array_optimal_attacker_payoff[i] = opt_attacker_payoff
        array_privacy[i] = 1 - opt_success_rate * opt_attack
        array_utility[i] = opt_utility
        sum_opt_geno += opt_geno
    dataset3 = pd.DataFrame(rows_list1)
    dataset3.to_pickle(pickle_filename31)
    dataset3 = pd.DataFrame(rows_list2)
    dataset3.to_pickle(pickle_filename32)
    dataset3 = pd.DataFrame(rows_list3)
    dataset3.to_pickle(pickle_filename33)
    dataset = pd.DataFrame({'privacy': array_privacy,
                            'utility': array_utility,
                            'defender_optimal': array_optimal_payoff,
                            'attacker_optimal': array_optimal_attacker_payoff})
    dataset.to_pickle(pickle_filename)
    endtime2 = time.time()
    f = open(filename, 'w')
    print("Elapsed time is :" + str(endtime2 - starttime2) + " seconds (computing).")
    f.write('Average strategy: ' + str(sum_opt_geno / a1) + '\n')
    f.write('Data subjects\' average payoff: ' + str(np.mean(array_optimal_payoff)) + '\n')
    f.write("Elapsed time is :" + str(endtime2 - starttime2) + " seconds (computing).\n")
    f.close()
    if save_dic == 1:
        # save dictionaries
        dic_names = ['dist', 'score', 'score_solo', 'attack', 'surname']
        dics = [dic_dist, dic_score, dic_score_solo, dic_attack, dic_surname]
        for i in range(5):
            f1 = open(folder_result + 'dic_s' + str(scenario) + '_' + dic_names[i] + '.pkl', 'wb')
            pickle.dump(dics[i], f1, protocol=pickle.HIGHEST_PROTOCOL)
            f1.close()
