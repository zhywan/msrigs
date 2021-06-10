# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing sensitivity analysis results in line plots (usefulness, privacy, utility, payoff, fairness wrt
# usefulness, fairness wrt privacy, fairness wrt utility, fairness wrt payoff) wrt minority-support factor
# © Oct 2021-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
# Update history:
# May 3, 2021: sensitivity of usefulness and fairness
# May 19, 2021: 8 figures.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

start1 = time.time()

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=9)    # fontsize of the x and y labels (original: MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

id_exp = '2058'  # ID for the set of experiments
n_iter = 100
n_S = 1000
n_scenario = 5
pruning = 1
order = [0, 1, 2, 3, 4]  # 0: no protection. 1: k-anonymity. 2: random masking.
# 3: opt-in game. 4: masking game.
scenario_id = ['0', '3.1', '3', '4', '5']  # 0: no protection. 3.1: k-anonymity. 2: random masking.
# 3: opt-in game. 4: masking game.
n_fig = 8  # Number of lineplot figures
n_row = 2  # number of rows of subplots in each figure
n_col = 4  # number of collums of subplots in each figure
fig, axes = plt.subplots(n_row, n_col, figsize=(15, 6.9))


#metric_name = ['payoff', 'privacy', 'utility']
column_names = ['usefulness', 'privacy', 'utility', 'defender_optimal', 'fairness_wrt_usefulness', 'fairness_wrt_privacy',
                'fairness_wrt_utility', 'fairness_wrt_payoff']
scenario_name = ['No-protection', 'K-anonymity', 'Random masking', 'Opt-in game', 'Masking game']
scenario_name = np.array(scenario_name)
scenario_name_in_order = scenario_name[order]
experiment_names = ['c-2', 'c-1', '', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8']
# c-2: alpha=-1, c-1: alpha=-0.5, c1: alpha=0.5, c2: alpha=1, c3: alpha=1.5, c4: alpha=2, c5: alpha=2.5, c6: alpha=3,...
ylabels = ['Usefulness', 'Privacy', 'Utility', 'Payoff ($)', 'Fairness with respect to usefulness',
           'Fairness with respect to privacy', 'Fairness with respect to utility', 'Fairness with respect to payoff']
style_order = ['No-protection', 'K-anonymity', 'Random masking', 'Opt-in game', 'Masking game']

m_g_start = -2
m_g_end = 9
step = 0.5
default_x = 0
for fig_i in range(n_fig):  # each subplot

    fig_row = fig_i // n_col
    fig_col = fig_i % n_col

    # plot the default vertical lines
    axes[fig_row, fig_col].axvline(x=default_x, label='Default value', color='0.5', linestyle='--')

    
    # input result data (usefulness, privacy, payoff, fairness with respect to usefulness,
    # fairness with respect to privacy, fairness with respect to payoff)
    metric = []
    for experiment_name in experiment_names:
        folder_result = 'Results' + id_exp + experiment_name + '/Violin/m2/'
        folder_result2 = 'Results' + id_exp + experiment_name + '/Violin_bf/'
        if pruning == 1:
            folder_result += 'pruning/'
            folder_result2 += 'pruning/'
        folder_output = 'Results' + id_exp + '/'
        for j in range(n_scenario):
            if j < 4:
                if fig_i >= 1 and fig_i <= 3:  # metric is privacy, utility, or payoff
                    dataset = pd.read_pickle(folder_result + 'result_s' + scenario_id[j] + '.pickle')
                    data = np.array(dataset[column_names[fig_i]])
                    shaped_dataset = np.reshape(data, (n_iter, n_S))
                    av_dataset = np.mean(shaped_dataset, axis=1)
                    metric.extend(av_dataset)
                else:  # metric is not privacy, utility, and payoff
                    dataset = pd.read_pickle(folder_result + 'result2_s' + scenario_id[j] + '.pickle')
                    data = np.array(dataset[column_names[fig_i]])
                    metric.extend(data)
            else:  # j == 5
                if fig_i >= 1 and fig_i <= 3:  # metric is privacy, utility, or payoff
                    av_data = np.empty([n_iter])
                    for k in range(n_iter):  # for each iteration
                        dataset = pd.read_pickle(folder_result2 + 'result_s' + scenario_id[j] + '_i' + str(k) + '.pickle')
                        data = np.array(dataset[column_names[fig_i]])
                        av_data[k] = np.mean(data)
                else:  # metric is not privacy, utility, and payoff
                    av_data = np.empty([n_iter])
                    for k in range(n_iter):  # for each iteration
                        dataset = pd.read_pickle(folder_result2 + 'result2_s' + scenario_id[j] + '_i' + str(k) + '.pickle')
                        data = np.array(dataset[column_names[fig_i]])
                        av_data[k] = data[0]
                metric.extend(av_data)

    scenarios = []
    for j in range(m_g_start, m_g_end):
        for i in range(n_scenario):
            label = [scenario_name_in_order[i] for k in range(n_iter)]
            scenarios.extend(label)

    m_g = []
    for j in range(m_g_start, m_g_end):
        label = [j * step for k in range(n_iter * n_scenario)]
        m_g.extend(label)

    # plot each figure
    colors = ["tab:red", "tab:orange", "tab:green", "tab:cyan", "tab:blue"]
    colors = np.array(colors)
    colors_in_order = colors[order]
    customPalette = sns.set_palette(sns.color_palette(colors_in_order))
    markers_base = ["X", "^", "v", "o", "s"]
    markers = dict(zip(scenario_name, markers_base))

    dataset = pd.DataFrame({ylabels[fig_i]: metric,
                            'Scenario': scenarios,
                            'Minority-support factor': m_g,
                            })
    sns.lineplot(data=dataset, x='Minority-support factor', y=ylabels[fig_i], hue='Scenario', markers=markers, style='Scenario',
                 palette=customPalette, style_order=style_order, ax=axes[fig_row, fig_col], ci='sd')
    axes[fig_row, fig_col].legend_.remove()
    axes[fig_row, fig_col].text(-0.2, 0.98, str(chr(ord('@')+fig_i+1)), fontfamily='sans-serif',
                                size=9, weight='bold', transform=axes[fig_row, fig_col].transAxes)

# Adjust, show and save each figure

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
ax_pos = axes[1, 1].get_position()
lines, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(ax_pos.x0 + 0.27, ax_pos.y0 - 0.02), borderaxespad=0.,
           ncol=6)
fig.show()
fig.savefig(folder_output + 'sensitivity_minority_support_result.png', bbox_inches='tight',
            pad_inches=0.011, transparent=True, dpi=300)
