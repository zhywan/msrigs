# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.0
# Component: Functions for MSRIGS
# © Oct 2018-2020 Zhiyu Wan, HIPLAB, Vanderbilt University
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
# April 21, 2020: Put all functions in the same file for easy editing
# May 3, 2020: Better search algorithm
# May 7, 2020: Change all tuple_x_selection to tuple_x_mu.
# July 11, 2020: Modify one scenario, and add three new scenarios.
# July 11, 2020: Delete irrational adversary.
# July 23, 2020: Add separate random masking rate.
# July 30, 2020: Changed to brute-force search algorithm.
# July 31, 2020: Add brute-force algorithm and pruning tech.
# July 31, 2020: Print optimal strategy and payoff.
# Aug 2, 2020: Entropy is calculated using the genealogy dataset or the demographic dataset.
# Aug 4, 2020: Change the order of attributes in the greedy search algorithm will not affect anything.
# Aug 5, 2020: Delete the variable: redundancy.
# Aug 11, 2020: Simplify arguments (scenario instead of no_defense, in_out, no_geno and random_protection)
# Aug 12, 2020: Adversary prefers to attack (to break tie).
# Aug 15, 2020: Adversary prefers to not attack (to break tie).
# Aug 19, 2020: Simplify: 1) vectorize surname inference, 2) update dic_attack, 3) change the way to handle mask.
# Oct 10, 2020: Add the no-attack masking game.
# Oct 15, 2020: Exchange the orders of no-attack and one-stage game.
# Oct 17, 2020: Add optimal defense function for one-for-all setting.
# Oct 21, 2020: Convert data types to save memory.
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm


def dec2bin_np_array(x, m):
    a = np.zeros(m).astype(bool)
    # One way to convert
    # temp = str(bin(x))[2:]
    # for d in range(len(temp)):
    #    a[m_g + 2 - len(temp) + d] = int(temp[d])
    # Alternative way to convert
    for j in range(m):
        a[-(j + 1)] = (x >> j) % 2
    return a


def find_offspring(current_strategy):
    # generate all offspring
    n = sum(current_strategy)
    m = len(current_strategy)
    # enumerate the 2**n offspring
    for i in range(2 ** n - 2, 0, -1):
        tmp = 0
        k = 0
        for j in range(m):
            if current_strategy[-(j+1)]:
                if (i >> k) % 2 == 1:
                    tmp += 2 ** j
                k += 1
        yield tmp


def get_entropy(column):
    (hist, bin_edges) = np.histogram(column, bins=np.arange(min(column), max(column) + 2))
    hist1 = hist[hist > 0]
    pk = hist1 / len(column)
    entropy = -np.sum(pk * np.log(pk))
    return entropy


def etmrca(x, mu, T_Max, inv_Ne):
    # x: np binary vector
    # mu: np vector
    en_mu = np.reciprocal(np.exp(mu))  # exp(-mu)
    nx = len(x)
    nmu = len(mu)
    if nx != nmu:
        print("There is an error!\nInputs to function ETMRCA have different lengths.")
        return -1
    tao = 0
    sump = 0
    z = np.sum(mu[x])
    for t in range(T_Max+1):
        tmp = np.log(1-np.power(en_mu, 2*t))
        y = np.sum(tmp[np.logical_not(x)])
        p = math.exp((-1) * t * (inv_Ne + 2 * z) + y)
        tao += t * p
        sump += p
    tao /= sump
    return tao


def conf_score_solo(x, mu, T_Max, inv_Ne):
    en_mu = np.reciprocal(np.exp(mu))  # exp(-mu)
    nx = len(x)
    nmu = len(mu)
    if nx != nmu:
        print("There is an error!\nInputs to function CONF_SCORE_SOLO have different lengths.")
        return -1
    P1 = []
    sump1 = 0
    P3 = []
    sump3 = 0
    z = np.sum(mu[x])
    for t in range(T_Max+1):
        tmp = np.log(1 - np.power(en_mu, 2 * t))
        y = np.sum(tmp[np.logical_not(x)])
        p1 = math.exp((-1) * t * (inv_Ne + 2 * z) + y)
        P1.append(p1)
        sump1 += p1
        p3 = inv_Ne * math.pow((1-inv_Ne), t-1) # inv_Ne * math.exp(-inv_Ne * t)
        P3.append(p3)
        sump3 += p3
    delta = 0
    for j1 in range(1,T_Max+1):
        sum_p3 = 0
        for j3 in range(j1 + 1, T_Max + 1):
            sum_p3 += P3[j3]
        delta += P1[j1] * sum_p3
    delta /= sump1
    delta /= sump3
    return delta


def conf_score(x1, x2, mu1, mu2, T_Max, inv_Ne):
    en_mu1 = np.reciprocal(np.exp(mu1))  # exp(-mu)
    en_mu2 = np.reciprocal(np.exp(mu2))  # exp(-mu)
    nx1 = len(x1)
    nmu1 = len(mu1)
    nx2 = len(x2)
    nmu2 = len(mu2)
    if nx1 != nmu1 or nx2 != nmu2:
        print("There is an error!\nInputs to function CONF_SCORE have different lengths.")
        return -1
    P1 = []
    sump1 = 0
    P2 = []
    sump2 = 0
    P3 = []
    sump3 = 0
    z1 = np.sum(mu1[x1])
    z2 = np.sum(mu2[x2])
    for t in range(T_Max+1):
        tmp1 = np.log(1 - np.power(en_mu1, 2 * t))
        y1 = np.sum(tmp1[np.logical_not(x1)])
        p1 = math.exp((-1) * t * (inv_Ne + 2 * z1) + y1)
        P1.append(p1)
        sump1 += p1

        tmp2 = np.log(1 - np.power(en_mu2, 2 * t))
        y2 = np.sum(tmp2[np.logical_not(x2)])
        p2 = math.exp((-1) * t * (inv_Ne + 2 * z2) + y2)
        P2.append(p2)
        sump2 += p2

        p3 = inv_Ne * math.pow((1 - inv_Ne), t-1)
        P3.append(p3)
        sump3 += p3
    delta = 0
    for j1 in range(1, T_Max+1):
        sum_p2 = 0
        for j2 in range(j1 + 1, T_Max + 1):
            sum_p2 += P2[j2]
        sum_p3 = 0
        for j3 in range(j1 + 1, T_Max + 1):
            sum_p3 += P3[j3]
        delta += P1[j1]*sum_p2*sum_p3
    delta /= sump1
    delta /= sump2
    delta /= sump3
    return delta


def build_world(ID, genome, surname, ages, states, n_r, n_f, rate_s):
    # column names:[ID, firstname, f1, f2, g1, g2, ..., g16, surname, sensitive]
    n_gen = 3
    f0 = np.random.randint(n_f, size=(n_r[0], 1))
    for i in range(1, n_gen):
        new_f0 = np.random.randint(n_f, size=(n_r[i], 1))
        f0 = np.concatenate((f0, new_f0), axis=0)
    n_r = sum(n_r)
    sensitive = (np.random.rand(n_r, 1) < rate_s).astype(int)
    ID = np.asarray(ID).reshape(len(ID), 1)
    genome = np.asarray(genome)
    surname = np.asarray(surname).reshape(len(surname), 1)
    ages = np.asarray(ages).reshape(len(ages), 1)
    states = np.asarray(states).reshape(len(states), 1)
    World = np.concatenate((ID, f0, ages, states, genome, surname, sensitive), axis=1)
    return World.astype(int)


def generate_datasets(World, n_I, n_S, n_G):
    (n_r, n_c) = World.shape
    perm = np.random.permutation(n_r)
    select_I = perm[0:n_I]
    select_S = perm[0:n_S]
    select_G = perm[-n_G:]
    S = World[select_S]
    # column names: [f1, f2, g1, g2, ..., g16, sensitive] (should be but not)
    I = World[select_I]
    # column names: [ID, firstname, f1, f2, surname] (should be but not)
    G = World[select_G]
    # column names: [g1, g2, ..., g16, surname] (should be but not)
    # ID, first name, ages, states, genomes, surname, sensitive
    return (S, I, G)


def surname_inference(s, G, m_g, selection, mu, method, tol, dic_dist, dic_score_solo, dic_score, T_Max, inv_Ne):
    selection = selection.astype(bool)
    G_surname = G[:, -2]
    if np.sum(selection) == 0:  # no computation needed
        #print("No candidates!")
        inferred_surname = -1
        p = 0
        return inferred_surname, p
    dick = np.array(range(4, (m_g+4)))
    s_genome = s[dick[selection]]
    G_genome = G[:, dick[selection]]
    #s_surname = s[-2]
    if method == 6: #svm  #too slow
        svc = svm.SVC(kernel='linear')
        #svc = svm.SVC(kernel='poly', degree = 3)
        #svc = svm.SVC(kernel='rbf')
        svc.fit(G_genome, G_surname)
        inferred_surname = svc.predict(s_genome.reshape(1, -1))[0]  # .tolist()
        # print(log.score(S_genome, S_surname))
        p = 1
    if method == 5: #logistic regression  #too slow
        log = linear_model.LogisticRegression(solver='lbfgs', C=1e5, multi_class = 'multinomial')
        log.fit(G_genome, G_surname)
        inferred_surname = log.predict(s_genome.reshape(1, -1))[0]
        # print(log.score(S_genome, S_surname))
        p = 1
        #correct_infer = inferred_surname == s_surname
        #print(correct_infer)
    if method == 4: #linear regression
        regr = linear_model.LinearRegression()
        regr.fit(G_genome, G_surname)
        inferred_surname = regr.predict(s_genome.reshape(1, -1))[0]
        #print(regr.score(S_genome, S_surname))
        p = 1
        #correct_infer = inferred_surname == s_surname
        #print(correct_infer)
    if method == 3:#knn
        knn = KNeighborsClassifier(n_neighbors=1, p=1)
        knn.fit(G_genome, G_surname)
        inferred_surname = knn.predict(s_genome.reshape(1, -1))[0]
        #print(knn.score(S_genome, S_surname))
        p = 1
    if method == 1:
        surname_cands = []
        distance = np.sum(np.absolute(G_genome - s_genome), axis=1)
        min_distance = np.min(distance)
        for j in range(len(distance)):
            if distance[j] <= min_distance:
                surname_cands.append(G_surname[j])
        inferred_surname = max(set(surname_cands), key=surname_cands.count)  # pick the first surname w/ max occurrence
        p = surname_cands.count(inferred_surname) / (len(surname_cands) * 1.0)
    if method == 2:
        mu = mu[selection]
        surname_cands = []
        candidates = []
        distance = []
        n_candidates = 0
        N_Top = 10
        n_match = np.sum(G_genome == s_genome, axis=1)  # s_genome does not have zeros.
        max_n_match = np.max(n_match)  # simpler
        tol_n_match = max_n_match * (1 - tol)
        is_candidate = n_match >= tol_n_match  # lb_n_match:
        index_candidate = np.nonzero(is_candidate)  # a tuple cell of an array
        n_candidates = np.sum(is_candidate)
        if n_candidates == 0:
            # print("no candidates!")
            inferred_surname = -1
            p = 0
            return inferred_surname, p
        for i in range(n_candidates):
            j = index_candidate[0][i]
            candidates.append(j)
            g1 = G_genome[j, :]  # simpler
            g1_nz = g1 != 0
            x_nz = (g1[g1_nz] == s_genome[g1_nz])
            mu_nz = mu[g1_nz]
            tuple_x_mu = (tuple(x_nz), tuple(mu_nz))
            if tuple_x_mu in dic_dist:
                dist = dic_dist[tuple_x_mu]
            else:
                dist = etmrca(x_nz, mu_nz, T_Max, inv_Ne)
                dic_dist[tuple_x_mu] = dist
            distance.append(dist)

        # compute confidence score
        if n_candidates == 1:
            g1 = G_genome[candidates[0], :]  # simpler
            g1_nz = g1 != 0
            x_nz = (g1[g1_nz] == s_genome[g1_nz])
            mu_nz = mu[g1_nz]
            tuple_x_mu = (tuple(x_nz), tuple(mu_nz))
            if tuple_x_mu in dic_score_solo:
                score = dic_score_solo[tuple_x_mu]
            else:
                score = conf_score_solo(x_nz, mu_nz, T_Max, inv_Ne)
                dic_score_solo[tuple_x_mu] = score
            inferred_surname = G_surname[candidates[0]]
        else:
            indexD = list(np.argsort(distance))  # simpler
            for j in range(min(N_Top, n_candidates)):
                surname_cands.append(G_surname[candidates[indexD[j]]])
            name1 = surname_cands[0]
            for kk in range(len(surname_cands)):
                name2 = surname_cands[kk]
                if name1 != name2:
                    break
            if name1 == name2:
                g1 = G_genome[candidates[indexD[0]], :]
                g1_nz = g1 != 0
                x_nz = (g1[g1_nz] == s_genome[g1_nz])
                mu_nz = mu[g1_nz]
                tuple_x_mu = (tuple(x_nz), tuple(mu_nz))
                if tuple_x_mu in dic_score_solo:
                    score = dic_score_solo[tuple_x_mu]
                else:
                    score = conf_score_solo(x_nz, mu_nz, T_Max, inv_Ne)
                    dic_score_solo[tuple_x_mu] = score
            else:
                g1 = G_genome[candidates[indexD[0]], :]
                g1_nz = g1 != 0
                x1_nz = (g1[g1_nz] == s_genome[g1_nz])
                mu1_nz = mu[g1_nz]
                g1 = G_genome[candidates[indexD[kk]], :]
                g1_nz = g1 != 0
                x2_nz = (g1[g1_nz] == s_genome[g1_nz])
                mu2_nz = mu[g1_nz]
                tuple_x1_mu1 = (tuple(x1_nz), tuple(mu1_nz))
                tuple_x2_mu2 = (tuple(x2_nz), tuple(mu2_nz))
                if (tuple_x1_mu1, tuple_x2_mu2) in dic_score:
                    score = dic_score[(tuple_x1_mu1, tuple_x2_mu2)]
                else:
                    score = conf_score(x1_nz, x2_nz, mu1_nz, mu2_nz, T_Max, inv_Ne)
                    dic_score[(tuple_x1_mu1, tuple_x2_mu2)] = score
            inferred_surname = surname_cands[0]
        p = np.float32(score)
    return inferred_surname, p


def attack_SIG(s_feature, I_feature, loss, cost, inferred_surname, p, theta_p, over_confident, mask_demo):
    mask_level = np.sum(mask_demo)
    if mask_level != 0:
        selection = np.array(range(2)).astype(int)
        masked_selection = selection[mask_demo.astype(bool)]
        s_feature1 = s_feature[masked_selection]  # age and state
        I_feature1 = I_feature[:, masked_selection]
        I_feature2 = I_feature[:, np.append(masked_selection, -1)]
        s_feature3 = np.append(s_feature1, inferred_surname)  # age, state and inferred surname
    attack = False
    real_success_rate = 0
    # age and state
    if mask_level == 0:
        group_size = len(I_feature)
    else:
        distance = np.sum(np.absolute(I_feature1 - s_feature1), axis=1)
        group_size = np.count_nonzero(distance == 0)
    real_success_rate1 = 1 / group_size
    payoff1 = loss * real_success_rate1 - cost
    if payoff1 > 0:
        attack1 = True
    else:
        attack1 = False
    real_payoff1 = max(payoff1, 0)

    # age, state, and inferred surname
    if mask_level == 0:
        group_size = np.sum(I_feature[:, -1] == inferred_surname)
    else:
        distance = np.sum(np.absolute(I_feature2 - s_feature3), axis=1)
        group_size = np.count_nonzero(distance == 0)
    if group_size == 0 or p < theta_p:  # use age and state instead (wrong inference or no inference)
        real_success_rate = real_success_rate1
        if payoff1 > 0:
            attack = True
        real_payoff = real_payoff1
    else:
        if over_confident == 1:
            success_rate = 1 / group_size  # expected success rate
        else:
            success_rate = 1 / group_size * p
        payoff = loss * success_rate - cost  # expected payoff
        if payoff > payoff1:  # use age, state and inferred surname
            if payoff > 0:
                attack = True
                if inferred_surname == s_feature[-1]:  # s_real_name = s_feature[-1]
                    real_success_rate = 1 / group_size
                real_payoff = loss * real_success_rate - cost  # not expected_payoff
            else:
                real_payoff = 0
        else:  # use age and state instead
            real_success_rate = real_success_rate1
            if payoff1 > 0:
                attack = True
            real_payoff = real_payoff1
    return real_success_rate, attack, real_payoff, real_success_rate1, attack1, real_payoff1


def optimal_defense(s, I, G, w_entropy, m_g, dic_attack, dic_surname, loss, cost, scenario,
                    total_utility, theta_p, over_confident, mu, method, tol, dic_dist, dic_score_solo, dic_score,
                    T_Max, inv_Ne, participation_rate, random_mask_rate, algorithm, pruning, I_selection):
    if scenario == 3:  # scenario 3: random masking
        random_strategy = np.random.choice([False, True], m_g + 2, p=[1 - random_mask_rate, random_mask_rate])
        random_demo = random_strategy[0:2]
        random_geno = random_strategy[2:]
        (inferred_surname, p) = surname_inference(s, G, m_g, random_geno, mu[0:m_g], method, tol, dic_dist,
                                                  dic_score_solo, dic_score, T_Max, inv_Ne)
        (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
            attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident, random_demo)
        defender_loss = attacker_payoff + attack * cost
        utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], random_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
        defender_benefit = total_utility * utility  # compute the benefit
        defender_payoff = defender_benefit - defender_loss
        optimal_attack = attack
        optimal_utility = utility
        optimal_success_rate = success_rate
        optimal_payoff = defender_payoff
        optimal_attacker_payoff = attacker_payoff
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, random_strategy

    # publish all data
    all_demo = np.ones(2).astype(bool)
    all_geno = np.ones(m_g).astype(bool)
    all_strategy = np.ones(m_g + 2).astype(bool)
    null_strategy = np.zeros(m_g + 2).astype(bool)
    optimal_strategy = all_strategy
    tuple_geno = (tuple(all_geno), tuple(s[4:(m_g+4)]))
    if tuple_geno in dic_surname:
        (inferred_surname, p) = dic_surname[tuple_geno]
    else:
        (inferred_surname, p) = surname_inference(s, G, m_g, all_geno, mu[0:m_g], method, tol, dic_dist,
                                                  dic_score_solo, dic_score, T_Max, inv_Ne)
        dic_surname[tuple_geno] = (inferred_surname, p)
    tuple_demo = (tuple(all_demo), tuple(s[2:4]), inferred_surname, p)
    if tuple_demo in dic_attack:
        (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
            dic_attack[tuple_demo]
    else:
        (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
            attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident, all_demo)
        # note: s_feature = s[I_selection], I_feature = I[:, I_selection]
        dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1)
        # update dict of attack

    if scenario == 1 or scenario == 7:  # scenario 1: no genomic data sharing, or new scenario 7: one-stage masking game
        if scenario == 1:
            no_geno_strategy = np.concatenate((np.ones(2).astype(bool), np.zeros(m_g).astype(bool)), axis=None)
            utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], no_geno_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
            optimal_strategy = no_geno_strategy
        else:
            utility = 1
        attacker_payoff = attacker_payoff1
        attack = attack1
        success_rate = success_rate1
    else:  # scenario 0: no protection, and other scenarios
        utility = 1
    defender_loss = attacker_payoff + attack * cost
    defender_benefit = total_utility * utility  # compute the benefit
    defender_payoff = defender_benefit - defender_loss
    optimal_utility = utility
    optimal_payoff = defender_payoff
    if scenario == 6 and attack:  # in no-attack game, a strategy will not be optimal unless there is no attack
        optimal_payoff = -10000
    optimal_attacker_payoff = attacker_payoff
    optimal_attack = attack
    optimal_p = p
    optimal_success_rate = success_rate
    if scenario == 0 or scenario == 1:  # scenario 0: no protection, or scenario 1: no genomic data sharing
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy
    if scenario == 2:  # scenario 2: random opt-in
        if np.random.random_sample() >= participation_rate:  # choose to opt-out
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = False
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_strategy = null_strategy
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy

    if scenario == 4:  # scenario 4: opt-in or opt-out
        if optimal_payoff <= 0:  # do not release anything
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = False
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_strategy = null_strategy
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy

    # scenario 5: masking game, scenario 6: no-attack masking game, or scenario 7: one-stage masking game
    if algorithm == 0:  # greedy algorithm
        current_strategy = all_strategy
        height_lattice = np.sum(current_strategy.astype(int))
        if scenario == 7:
            height_lattice = 3
        for _ in range(height_lattice - 1):
            child_optimal_payoff = -10000
            child_optimal_p = 2
            for i in range(m_g + 2):
                if not current_strategy[i]:
                    continue
                child_strategy = current_strategy.copy()
                child_strategy[i] = False
                child_demo = child_strategy[0:2]
                child_geno = child_strategy[2:]
                tuple_geno = (tuple(child_geno), tuple(s[4:(m_g + 4)] * child_geno))
                if tuple_geno in dic_surname:
                    (inferred_surname, p) = dic_surname[tuple_geno]
                else:
                    (inferred_surname, p) = surname_inference(s, G, m_g, child_geno, mu[0:m_g], method, tol, dic_dist,
                                                              dic_score_solo, dic_score, T_Max, inv_Ne)
                    dic_surname[tuple_geno] = (inferred_surname, p)
                tuple_demo = (tuple(child_demo), tuple(s[2:4] * child_demo), inferred_surname, p)
                if tuple_demo in dic_attack:
                    (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                        dic_attack[tuple_demo]
                else:
                    (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                        attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p,
                                   over_confident, child_demo)
                    dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1,
                                              attacker_payoff1)
                defender_loss = attacker_payoff + attack * cost
                if scenario == 7:
                    defender_loss = attacker_payoff1 + attack1 * cost
                utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], child_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
                defender_benefit = total_utility * utility  # compute the benefit
                defender_payoff = defender_benefit - defender_loss
                if defender_payoff > child_optimal_payoff or \
                        (defender_payoff == child_optimal_payoff and p < child_optimal_p):  # privacy measure
                    child_optimal_child_strategy = child_strategy
                    child_optimal_payoff = defender_payoff
                    child_optimal_attacker_payoff = attacker_payoff
                    child_optimal_utility = utility
                    child_optimal_attack = attack
                    child_optimal_success_rate = success_rate
                    child_optimal_p = p
                    if scenario == 7:
                        child_optimal_attacker_payoff = attacker_payoff1
                        child_optimal_attack = attack1
                        child_optimal_success_rate = success_rate1
                        child_optimal_p = 0
                # in no-attack game, update the optimal strategy only if there is no attack
                if scenario == 6 and not attack and \
                        (defender_payoff > optimal_payoff or
                         (defender_payoff == optimal_payoff and p < optimal_p)):  # privacy measure
                        optimal_payoff = defender_payoff
                        optimal_attacker_payoff = attacker_payoff
                        optimal_attack = attack
                        optimal_success_rate = success_rate
                        optimal_utility = utility
                        optimal_p = p
                        optimal_strategy = child_strategy
            if scenario == 6:
                if pruning == 1 and not child_optimal_attack:  # condition for pruning
                    break
            else:  # scenario 5: masking game, or scenario 7: one-stage masking game
                if child_optimal_payoff > optimal_payoff or \
                        (child_optimal_payoff == optimal_payoff and child_optimal_p < optimal_p):  # privacy measure
                    optimal_payoff = child_optimal_payoff
                    optimal_attacker_payoff = child_optimal_attacker_payoff
                    optimal_attack = child_optimal_attack
                    optimal_success_rate = child_optimal_success_rate
                    optimal_utility = child_optimal_utility
                    optimal_p = child_optimal_p
                    optimal_strategy = child_optimal_child_strategy
                elif child_optimal_payoff < optimal_payoff and pruning == 1 and not child_optimal_attack:  # condition for pruning
                    break
            current_strategy = child_optimal_child_strategy
    elif algorithm == 1:  # brute-force algorithm
        if scenario == 7:
            new_m_g = 0
        else:
            new_m_g = m_g
        visited = np.zeros(2 ** (new_m_g + 2)).astype(bool)
        visited[-1] = True
        for x in range(2 ** (new_m_g + 2) - 2, 0, -1):
            #print('x: ', x)
            if visited[x]:
                continue
            visited[x] = True
            if scenario == 7:
                current_strategy = np.append(dec2bin_np_array(x, new_m_g + 2), np.ones(m_g).astype(bool))
            else:
                current_strategy = dec2bin_np_array(x, m_g + 2)
            child_demo = current_strategy[0:2]
            child_geno = current_strategy[2:]
            tuple_geno = (tuple(child_geno), tuple(s[4:(m_g + 4)] * child_geno))
            if tuple_geno in dic_surname:
                (inferred_surname, p) = dic_surname[tuple_geno]
            else:
                (inferred_surname, p) = surname_inference(s, G, m_g, child_geno, mu[0:m_g], method, tol, dic_dist,
                                                          dic_score_solo, dic_score, T_Max, inv_Ne)
                dic_surname[tuple_geno] = (inferred_surname, p)
            tuple_demo = (tuple(child_demo), tuple(s[2:4] * child_demo), inferred_surname, p)
            if tuple_demo in dic_attack:
                (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                    dic_attack[tuple_demo]
            else:
                (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                    attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident, child_demo)
                dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1)
            if scenario == 6 and attack:  # in no-attack game, jump over if the adversary attacks
                continue
            defender_loss = attacker_payoff + attack * cost
            if scenario == 7:
                defender_loss = attacker_payoff1 + attack1 * cost
            utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], current_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
            defender_benefit = total_utility * utility  # compute the benefit
            defender_payoff = defender_benefit - defender_loss
            if defender_payoff > optimal_payoff or \
                    (defender_payoff == optimal_payoff and p < optimal_p):  # privacy measure
                optimal_payoff = defender_payoff
                optimal_attacker_payoff = attacker_payoff
                optimal_utility = utility
                optimal_attack = attack
                optimal_success_rate = success_rate
                optimal_p = p
                if scenario == 7:
                    optimal_attacker_payoff = attacker_payoff1
                    optimal_attack = attack1
                    optimal_success_rate = success_rate1
                    optimal_p = 0
                optimal_strategy = current_strategy
            elif defender_payoff < optimal_payoff and pruning == 1 and not attack and scenario != 7:  # condition for pruning
                for i in find_offspring(current_strategy):
                    if not visited[i]:
                        #print(i)
                        visited[i] = True
    if optimal_payoff < 0:  # no data release, and no attack
        optimal_payoff = 0
        optimal_attacker_payoff = 0
        optimal_attack = False
        optimal_success_rate = 0
        optimal_utility = 0
        optimal_strategy = null_strategy
    #print(optimal_strategy)
    #print(optimal_payoff)
    return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy


def optimal_defense_all(S, I, G, w_entropy, m_g, dic_attack, dic_surname, loss, cost, scenario,
                    total_utility, theta_p, over_confident, mu, method, tol, dic_dist, dic_score_solo, dic_score,
                    T_Max, inv_Ne, participation_rate, random_mask_rate, algorithm, pruning, I_selection):
    # optimal defense function for the one-for-all setting
    n_S = np.shape(S)[0]
    if scenario == 3:  # scenario 3: random masking
        random_strategy = np.random.choice([False, True], m_g + 2, p=[1 - random_mask_rate, random_mask_rate])
        random_demo = random_strategy[0:2]
        random_geno = random_strategy[2:]
        sum_defender_loss = 0
        sum_attacker_payoff = 0
        sum_attack = 0
        sum_success_rate = 0
        for j in range(n_S):
            s = S[j, :]
            (inferred_surname, p) = surname_inference(s, G, m_g, random_geno, mu[0:m_g], method, tol, dic_dist,
                                                      dic_score_solo, dic_score, T_Max, inv_Ne)
            (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident,
                           random_demo)
            defender_loss = attacker_payoff + attack * cost
            sum_defender_loss += defender_loss
            sum_attacker_payoff += attacker_payoff
            sum_attack += attack
            sum_success_rate += success_rate
        av_defender_loss = sum_defender_loss / n_S
        av_attacker_payoff = sum_attacker_payoff / n_S
        av_attack = sum_attack / n_S
        av_success_rate = sum_success_rate / n_S
        utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], random_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
        defender_benefit = total_utility * utility  # compute the benefit
        av_defender_payoff = defender_benefit - av_defender_loss
        optimal_attack = av_attack
        optimal_utility = utility
        optimal_success_rate = av_success_rate
        optimal_payoff = av_defender_payoff
        optimal_attacker_payoff = av_attacker_payoff
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, random_strategy

    # publish all data
    all_demo = np.ones(2).astype(bool)
    all_geno = np.ones(m_g).astype(bool)
    all_strategy = np.ones(m_g + 2).astype(bool)
    null_strategy = np.zeros(m_g + 2).astype(bool)
    optimal_strategy = all_strategy
    sum_defender_loss = 0
    sum_attacker_payoff = 0
    sum_attack = 0
    sum_success_rate = 0
    sum_p = 0
    for j in range(n_S):
        s = S[j, :]
        tuple_geno = (tuple(all_geno), tuple(s[4:(m_g + 4)]))
        if tuple_geno in dic_surname:
            (inferred_surname, p) = dic_surname[tuple_geno]
        else:
            (inferred_surname, p) = surname_inference(s, G, m_g, all_geno, mu[0:m_g], method, tol, dic_dist,
                                                      dic_score_solo, dic_score, T_Max, inv_Ne)
            dic_surname[tuple_geno] = (inferred_surname, p)
        tuple_demo = (tuple(all_demo), tuple(s[2:4]), inferred_surname, p)
        if tuple_demo in dic_attack:
            (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                dic_attack[tuple_demo]
        else:
            (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident,
                           all_demo)
            # note: s_feature = s[I_selection], I_feature = I[:, I_selection]
            dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1)
            # update dict of attack
        if scenario == 1 or scenario == 7:  # scenario 1: no genomic data sharing, or scenario 7: one-stage masking game
            attacker_payoff = attacker_payoff1
            attack = attack1
            success_rate = success_rate1
        else:  # scenario 0: no protection, and other scenarios
            pass
        defender_loss = attacker_payoff + attack * cost
        sum_defender_loss += defender_loss
        sum_attacker_payoff += attacker_payoff
        sum_attack += attack
        sum_success_rate += success_rate
        sum_p += p
    av_defender_loss = sum_defender_loss / n_S
    av_attacker_payoff = sum_attacker_payoff / n_S
    av_attack = sum_attack / n_S
    av_success_rate = sum_success_rate / n_S
    av_p = sum_p / n_S
    if scenario == 1 or scenario == 7:  # scenario 1: no genomic data sharing, or scenario 7: one-stage masking game
        if scenario == 1:
            no_geno_strategy = np.concatenate((np.ones(2).astype(bool), np.zeros(m_g).astype(bool)), axis=None)
            utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], no_geno_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
            optimal_strategy = no_geno_strategy
        else:
            utility = 1
    else:  # scenario 0: no protection, and other scenarios
        utility = 1
    defender_benefit = total_utility * utility  # compute the benefit
    av_defender_payoff = defender_benefit - av_defender_loss
    optimal_utility = utility
    optimal_payoff = av_defender_payoff
    if scenario == 6 and av_attack != 0:  # in no-attack game, a strategy will not be optimal unless there is no attack
        optimal_payoff = -10000
    optimal_attacker_payoff = av_attacker_payoff
    optimal_attack = av_attack
    optimal_p = av_p
    optimal_success_rate = av_success_rate
    if scenario == 0 or scenario == 1:  # scenario 0: no protection, or scenario 1: no genomic data sharing
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy
    if scenario == 2:  # scenario 2: random opt-in
        if np.random.random_sample() >= participation_rate:  # choose to opt-out
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = 0
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_strategy = null_strategy
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy

    if scenario == 4:  # scenario 4: opt-in or opt-out
        if optimal_payoff <= 0:  # do not release anything
            optimal_payoff = 0
            optimal_attacker_payoff = 0
            optimal_attack = 0
            optimal_success_rate = 0
            optimal_utility = 0
            optimal_strategy = null_strategy
        return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy

    # scenario 5: masking game, scenario 6: no-attack masking game, or scenario 7: one-stage masking game
    if algorithm == 0:  # greedy algorithm
        current_strategy = all_strategy
        height_lattice = np.sum(current_strategy.astype(int))
        if scenario == 7:
            height_lattice = 3
        for _ in range(height_lattice - 1):
            child_optimal_payoff = -10000
            child_optimal_p = 2
            for i in range(m_g + 2):
                if not current_strategy[i]:
                    continue
                child_strategy = current_strategy.copy()
                child_strategy[i] = False
                child_demo = child_strategy[0:2]
                child_geno = child_strategy[2:]
                sum_defender_loss = 0
                sum_attacker_payoff = 0
                sum_attack = 0
                sum_success_rate = 0
                sum_p = 0
                sum_attacker_payoff1 = 0
                sum_attack1 = 0
                sum_success_rate1 = 0
                for j in range(n_S):
                    s = S[j, :]
                    tuple_geno = (tuple(child_geno), tuple(s[4:(m_g + 4)] * child_geno))
                    if tuple_geno in dic_surname:
                        (inferred_surname, p) = dic_surname[tuple_geno]
                    else:
                        (inferred_surname, p) = surname_inference(s, G, m_g, child_geno, mu[0:m_g], method, tol,
                                                                     dic_dist, dic_score_solo, dic_score, T_Max, inv_Ne)
                        dic_surname[tuple_geno] = (inferred_surname, p)
                    tuple_demo = (tuple(child_demo), tuple(s[2:4] * child_demo), inferred_surname, p)
                    if tuple_demo in dic_attack:
                        (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                            dic_attack[tuple_demo]
                    else:
                        (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                            attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p,
                                          over_confident, child_demo)
                        dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1,
                                                  attacker_payoff1)
                    defender_loss = attacker_payoff + attack * cost
                    if scenario == 7:
                        defender_loss = attacker_payoff1 + attack1 * cost
                    sum_defender_loss += defender_loss
                    sum_attacker_payoff += attacker_payoff
                    sum_attack += attack
                    sum_success_rate += success_rate
                    sum_p += p
                    sum_attacker_payoff1 += attacker_payoff1
                    sum_attack1 += attack1
                    sum_success_rate1 += success_rate1
                av_defender_loss = sum_defender_loss / n_S
                av_attacker_payoff = sum_attacker_payoff / n_S
                av_attack = sum_attack / n_S
                av_success_rate = sum_success_rate / n_S
                av_p = sum_p / n_S
                av_attacker_payoff1 = sum_attacker_payoff1 / n_S
                av_attack1 = sum_attack1 / n_S
                av_success_rate1 = sum_success_rate1 / n_S
                utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], child_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
                defender_benefit = total_utility * utility  # compute the benefit
                av_defender_payoff = defender_benefit - av_defender_loss
                if av_defender_payoff >= child_optimal_payoff or \
                        (av_defender_payoff == child_optimal_payoff and av_p < child_optimal_p):  # privacy measure
                    child_optimal_payoff = av_defender_payoff
                    child_optimal_child_strategy = child_strategy
                    child_optimal_attacker_payoff = av_attacker_payoff
                    child_optimal_utility = utility
                    child_optimal_attack = av_attack
                    child_optimal_success_rate = av_success_rate
                    child_optimal_p = av_p
                    if scenario == 7:
                        child_optimal_attacker_payoff = av_attacker_payoff1
                        child_optimal_attack = av_attack1
                        child_optimal_success_rate = av_success_rate1
                        child_optimal_p = 0
                # in no-attack game, update the optimal strategy only if there is no attack
                if scenario == 6 and av_attack == 0 and \
                        (av_defender_payoff > optimal_payoff or
                         (av_defender_payoff == optimal_payoff and av_p < optimal_p)):  # privacy measure
                    optimal_payoff = av_defender_payoff
                    optimal_attacker_payoff = av_attacker_payoff
                    optimal_attack = av_attack
                    optimal_success_rate = av_success_rate
                    optimal_utility = utility
                    optimal_p = av_p
                    optimal_strategy = child_strategy
            if scenario == 6:
                if pruning == 1 and child_optimal_attack == 0:  # condition for pruning
                    break
            else:  # scenario 5: masking game, or scenario 7: one-stage masking game
                if child_optimal_payoff > optimal_payoff or \
                        (child_optimal_payoff == optimal_payoff and child_optimal_p < optimal_p):  # privacy measure
                    optimal_payoff = child_optimal_payoff
                    optimal_attacker_payoff = child_optimal_attacker_payoff
                    optimal_attack = child_optimal_attack
                    optimal_success_rate = child_optimal_success_rate
                    optimal_utility = child_optimal_utility
                    optimal_p = child_optimal_p
                    optimal_strategy = child_optimal_child_strategy
                elif child_optimal_payoff < optimal_payoff and pruning == 1 and child_optimal_attack == 0:  # condition for pruning
                    break
            current_strategy = child_optimal_child_strategy
    elif algorithm == 1:  # brute-force algorithm
        if scenario == 7:
            new_m_g = 0
        else:
            new_m_g = m_g
        visited = np.zeros(2 ** (new_m_g + 2)).astype(bool)
        visited[-1] = True
        for x in range(2 ** (new_m_g + 2) - 2, 0, -1):
            #print('x: ', x)
            if visited[x]:
                continue
            visited[x] = True
            if scenario == 7:
                current_strategy = np.append(dec2bin_np_array(x, new_m_g + 2), np.ones(m_g).astype(bool))
            else:
                current_strategy = dec2bin_np_array(x, m_g + 2)
            child_demo = current_strategy[0:2]
            child_geno = current_strategy[2:]
            sum_defender_loss = 0
            sum_attacker_payoff = 0
            sum_attack = 0
            sum_success_rate = 0
            sum_p = 0
            sum_attacker_payoff1 = 0
            sum_attack1 = 0
            sum_success_rate1 = 0
            for j in range(n_S):
                s = S[j, :]
                tuple_geno = (tuple(child_geno), tuple(s[4:(m_g + 4)] * child_geno))
                if tuple_geno in dic_surname:
                    (inferred_surname, p) = dic_surname[tuple_geno]
                else:
                    (inferred_surname, p) = surname_inference(s, G, m_g, child_geno, mu[0:m_g], method, tol, dic_dist,
                                                              dic_score_solo, dic_score, T_Max, inv_Ne)
                    dic_surname[tuple_geno] = (inferred_surname, p)
                tuple_demo = (tuple(child_demo), tuple(s[2:4] * child_demo), inferred_surname, p)
                if tuple_demo in dic_attack:
                    (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                        dic_attack[tuple_demo]
                else:
                    (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1) = \
                        attack_SIG(s[I_selection], I[:, I_selection], loss, cost, inferred_surname, p, theta_p, over_confident, child_demo)
                    dic_attack[tuple_demo] = (success_rate, attack, attacker_payoff, success_rate1, attack1, attacker_payoff1)
                if scenario == 6 and attack:  # in no-attack game, jump over if the adversary attacks
                    continue
                defender_loss = attacker_payoff + attack * cost
                if scenario == 7:
                    defender_loss = attacker_payoff1 + attack1 * cost
                sum_defender_loss += defender_loss
                sum_attacker_payoff += attacker_payoff
                sum_attack += attack
                sum_success_rate += success_rate
                sum_p += p
                sum_attacker_payoff1 += attacker_payoff1
                sum_attack1 += attack1
                sum_success_rate1 += success_rate1
            av_defender_loss = sum_defender_loss / n_S
            av_attacker_payoff = sum_attacker_payoff / n_S
            av_attack = sum_attack / n_S
            av_success_rate = sum_success_rate / n_S
            av_p = sum_p / n_S
            av_attacker_payoff1 = sum_attacker_payoff1 / n_S
            av_attack1 = sum_attack1 / n_S
            av_success_rate1 = sum_success_rate1 / n_S
            utility = np.sum(np.dot(w_entropy[0:(m_g + 2)], current_strategy)) / np.sum(w_entropy[0:(m_g + 2)])
            defender_benefit = total_utility * utility  # compute the benefit
            av_defender_payoff = defender_benefit - av_defender_loss
            if av_defender_payoff > optimal_payoff or \
                    (av_defender_payoff == optimal_payoff and av_p < optimal_p):  # privacy measure
                optimal_payoff = av_defender_payoff
                optimal_attacker_payoff = av_attacker_payoff
                optimal_utility = utility
                optimal_attack = av_attack
                optimal_success_rate = av_success_rate
                optimal_p = av_p
                if scenario == 7:
                    optimal_attacker_payoff = av_attacker_payoff1
                    optimal_attack = av_attack1
                    optimal_success_rate = av_success_rate1
                    optimal_p = 0
                optimal_strategy = current_strategy
            elif av_defender_payoff < optimal_payoff and pruning == 1 and av_attack == 0 and scenario != 7:  # condition for pruning
                for i in find_offspring(current_strategy):
                    if not visited[i]:
                        #print(i)
                        visited[i] = True
    if optimal_payoff < 0:  # no data release, and no attack
        optimal_payoff = 0
        optimal_attacker_payoff = 0
        optimal_attack = 0
        optimal_success_rate = 0
        optimal_utility = 0
        optimal_strategy = null_strategy
    return optimal_payoff, optimal_attacker_payoff, optimal_attack, optimal_success_rate, optimal_utility, optimal_strategy


def dic_compare(d1, d2):
    # Compare Two dictionaries
    # Example:
    # dic_name = 'score'
    # with open('Results2043/mchanging/pruning/dic_p0_s5_' + dic_name + '.pkl', 'rb') as f:
    #     dic1 = pickle.load(f)
    # with open('Results2043/Violin/m2/pruning/dic_s5_' + dic_name + '.pkl', 'rb') as f:
    #     dic2 = pickle.load(f)
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    shared_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o: (d1[o], d2[o]) for o in shared_keys if d1[o] != d2[o]}
    same = set(o for o in shared_keys if d1[o] == d2[o])
    return added, removed, modified, same
