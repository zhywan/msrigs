# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing sensitivity analysis results in line plots (payoff, privacy, and utility) and violin plots (payoff)
# © Oct 2018-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
# Update history:
# April 21, 2020: plot multiple figure
# Aug 1, 2020: plot all sub-figures
# Oct 13, 2020: Add two variations of masking game
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

id_exp = '2056'  # ID for the set of experiments
n_iter = 20
n_S = 1000
n_scenario = 8
pruning = 1
order = [0, 1, 2, 3, 7, 4, 5, 6]  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking.
# 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.
n_fig = 8  # Number of lineplot figures
n_row = [4, 3, 3]  # number of rows of subplots in each figure
n_col = [3, 3, 3]  # number of collums of subplots in each figure
fig_row = np.zeros(3).astype(int)
fig_col = np.zeros(3).astype(int)
fig0, axes0 = plt.subplots(n_row[0], n_col[0], figsize=(10.28, 12.6))
fig1, axes1 = plt.subplots(n_row[1], n_col[1], figsize=(10.4, 9.6))
fig2, axes2 = plt.subplots(n_row[2], n_col[2], figsize=(10.4, 9.6))
fig = [fig0, fig1, fig2]
axes = [axes0, axes1, axes2]

metric_name = ['payoff', 'privacy', 'utility']
column_names = ['defender_optimal', 'privacy', 'utility']
scenario_name = ['No-protection', 'Demographics-only', 'Random opt-in', 'Random masking', 'Opt-in game', 'Masking game',
                 'No-attack masking game', 'One-stage masking game']
scenario_name = np.array(scenario_name)
scenario_name_in_order = scenario_name[order]
experiments_name = ['m', 'missinglevel', 'theta', 'nG', 'nI', 'loss', 'benefit', 'cost']
experiments_name = [experiments_name[i] + 'changing' for i in range(len(experiments_name))]
xlabels = ['Number of genomic attributes', 'Proportion of missing genomic data', 'Threshold for confidence score',
           'Number of records in the genetic genealogy dataset', 'Number of records in the identified dataset',
           'Loss from being re-identified ($)', 'Maximal benefit of sharing all data ($)', 'Cost of attack ($)']
style_order = ['Masking game', 'One-stage masking game', 'No-attack masking game', 'Opt-in game', 'Random masking',
               'Random opt-in', 'Demographics-only', 'No-protection']
if n_scenario == 7:
    style_order = ['Masking game', 'One-stage masking game', 'No-protection', 'Opt-in game', 'Random masking',
                   'Random opt-in', 'Demographics-only']
ms_g_start = [2, 0, 0, 1, 1, 0, 0, 0]
n_ms = [15, 10, 11, 20, 20, 17, 17, 17]
ms_g_end = [17, 10, 11, 21, 21, 17, 17, 17]
steps = [1, 0.1, 0.1, 2000, 2000, 25, 25, 10]
default_xs = [12, 0.3, 0.5, 20000, 20000, 150, 100, 10]

for fig_i in range(n_fig):  # each subplot
    experiment_name = experiments_name[fig_i]
    xlabel = xlabels[fig_i]
    m_g_start = ms_g_start[fig_i]
    m_g_end = ms_g_end[fig_i]
    step = steps[fig_i]
    default_x = default_xs[fig_i]
    for i in range(3):
        fig_row[i] = fig_i // n_col[i]
        fig_col[i] = fig_i % n_col[i]

    # plot the default vertical lines
    for i in range(3):
        axes[i][fig_row[i], fig_col[i]].axvline(x=default_x, label='Default value', color='0.5', linestyle='--')
    folder_result = 'Results' + id_exp + '/' + experiment_name + '/'
    if pruning == 1:
        folder_result += 'pruning/'
    folder_output = 'Results' + id_exp + '/'
    
    # input result data (payoff, privacy, and utility)
    n_m = m_g_end - m_g_start
    payoffs = []
    privacy = []
    utility = []
    for jj in range(n_scenario):
        j = order[jj]  # re-order for plotting
        dataset = pd.read_pickle(folder_result + 'result_p' + str(fig_i) + '_s' + str(j) + '.pickle')
        for i in range(3):
            data = np.array(dataset[column_names[i]])
            shaped_dataset = np.reshape(data, (n_iter * n_m, n_S))
            av_dataset = np.mean(shaped_dataset, axis=1)
            if i == 0:
                payoffs.extend(av_dataset)
            elif i == 1:
                privacy.extend(av_dataset)
            else:
                utility.extend(av_dataset)

    m_g = []
    for i in range(n_scenario):
        for j in range(m_g_start, m_g_end):
            label = [j * step for k in range(n_iter)]
            m_g.extend(label)

    scenarios = []
    for i in range(n_scenario):
        label = [scenario_name_in_order[i] for j in range(n_iter * n_m)]
        scenarios.extend(label)

    # plot each figure
    results = [payoffs, privacy, utility]
    ylabels = ['Average payoff ($)', 'Privacy', 'Utility']
    colors = ["tab:red", "tab:orange", "tab:olive", "tab:green", "tab:cyan", "tab:blue", "tab:purple", "tab:gray"]
    colors = np.array(colors)
    colors_in_order = colors[order]
    customPalette = sns.set_palette(sns.color_palette(colors_in_order))
    markers_base = ["X", "d", "^", "v", "o", "s", "P", "D"]
    markers = dict(zip(scenario_name, markers_base))
    for i in range(3):
        dataset = pd.DataFrame({ylabels[i]: results[i],
                                'Scenario': scenarios,
                                xlabel: m_g,
                                })
        sns.lineplot(data=dataset, x=xlabel, y=ylabels[i], hue='Scenario', markers=markers, style='Scenario',
                     palette=customPalette, style_order=style_order, ax=axes[i][fig_row[i], fig_col[i]], ci='sd')
        if fig_i == 1:
            axes[i][fig_row[i], fig_col[i]].set(xlim=(-0.05, 0.95), xticks=np.array(range(10)) / 10)
        elif fig_i == 0:
            axes[i][fig_row[i], fig_col[i]].set(xlim=(1, 17), xticks=np.array(range(2, 17)))
        if i != 0:
            axes[i][fig_row[i], fig_col[i]].set(ylim=(-0.05, 1.05), yticks=np.array(range(11)) / 10)
        axes[i][fig_row[i], fig_col[i]].legend_.remove()
        axes[i][fig_row[i], fig_col[i]].text(-0.2, 0.98, str(chr(ord('@')+fig_i+1)), fontfamily='sans-serif',
                                             size=9, weight='bold', transform=axes[i][fig_row[i], fig_col[i]].transAxes)

# plots for sensitivity analyses on changing settings
for fig_i in range(n_fig, n_fig+3):
    fig_row[0] = (fig_i + 1) // n_col[0]
    fig_col[0] = (fig_i + 1) % n_col[0]

    if fig_i == (n_fig + 1):
        n_setting = 2
        method = 2
        name = 'oneforall'
        xlabel = 'Homogeneity constraint for adopted strategies'
        settings = ['Without', 'With']
    elif fig_i == (n_fig + 2):
        n_setting = 3
        name = 'multi_methods'
        xlabel = 'Surname inference approach'
        settings = ['TMRCA-based', 'KNN', 'Linear regression']
    elif fig_i == (n_fig + 0):
        n_setting = 3
        method = 2
        name = 'multi_weight_distributions'
        xlabel = 'Weight distribution of attributes'
        settings = ['Entropy-based', 'Uniform', 'Highly biased']
    folder_result_base = 'Results' + id_exp + '/Violin'
    folder_result = folder_result_base + '_' + name + '/m' + str(method) + '/'  # for alternative setting
    folder_result_head = folder_result_base + '_over_confident/m'  # for alternative inference methods
    folder_result_head_input = folder_result_base + '_' + name + '/Alter_weight_'  # for alternative weight distribution
    folder_result_base += '/m' + str(method) + '/'  # for default setting

    col1 = []
    rvs = []
    for j in range(n_setting):
        if fig_i == (n_fig + 2):  # Surname inference approach
            method = j + 2
            folder = folder_result_head + str(method) + '/'
        elif fig_i == (n_fig + 0) and j > 0:  # Weight distribution of attributes
            folder = folder_result_head_input + str(j) + '/m' + str(method) + '/'
        elif j == 0:
            folder = folder_result_base
        else:  # Homogeneity constraint for adopted strategies
            folder = folder_result
        if pruning == 1:
            folder += 'pruning/'
        if j == 1 and fig_i == (n_fig + 1):  # Homogeneity constraint for adopted strategies
            for i in range(3):  # 3 game scenarios
                scenario = i + 4
                dataset = pd.read_pickle(folder + 'result_s' + str(scenario) + '.pickle')
                col1.extend(dataset['defender_optimal'])
                rvs.append(dataset['defender_optimal'])
            continue  # different data format
        for i in range(3):  # 3 game scenarios
            scenario = i + 4
            if scenario == 4 and fig_i == (n_fig + 0):
                continue  # no need to display the opt-in game
            dataset = pd.read_pickle(folder + 'result_s' + str(scenario) + '.pickle')
            data = np.array(dataset['defender_optimal'])
            shaped_dataset = np.reshape(data, (n_iter, n_S))
            av_dataset = np.mean(shaped_dataset, axis=1)
            col1.extend(av_dataset)
            rvs.append(av_dataset)

    col2 = []
    if fig_i != (n_fig + 0):  # Not the setting on the weight distribution of attributes
        for i in range(n_setting):
            for j in range(3):  # 3 game scenarios
                col2.extend([scenario_name[j + 4] for k in range(n_iter)])
    else:
        for i in range(n_setting):
            for j in range(2):  # 2 game scenarios
                col2.extend([scenario_name[j + 5] for k in range(n_iter)])

    col3 = []
    for i in range(n_setting):
        if fig_i == (n_fig + 0):
            n_game = 2
        else:
            n_game = 3
        col3.extend([settings[i] for j in range(n_iter * n_game)])

    dataset2 = pd.DataFrame({'Average payoff ($)': col1,
                             'Games': col2,
                             xlabel: col3})
    if fig_i == (n_fig + 0):
        colors1 = colors[5:7]
    else:
        colors1 = colors[4:7]
    customPalette = sns.set_palette(sns.color_palette(colors1))
    sns.violinplot(data=dataset2, x=xlabel, y='Average payoff ($)', hue='Games', scale='width', palette=customPalette,
                   ax=axes[0][fig_row[0], fig_col[0]])
    axes[0][fig_row[0], fig_col[0]].legend_.remove()
    axes[0][fig_row[0], fig_col[0]].text(- 0.2, 0.98, str(chr(ord('@') + fig_i + 1)), size=9, fontfamily='sans-serif',
                                         transform=axes[0][fig_row[0], fig_col[0]].transAxes, weight='bold')
# Adjust, show and save each figure
for i in range(3):
    fig[i].subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax_pos = axes[i][2, 2].get_position()
    axes[i][2, 2].axis('off')
    lines, labels = axes[i][0, 0].get_legend_handles_labels()
    fig[i].legend(lines, labels, loc='upper left',
                  bbox_to_anchor=(ax_pos.x0 + 0.04, ax_pos.y1 - 0.02), borderaxespad=0.)
    if i == 0:
        lines, labels = axes[i][3, 2].get_legend_handles_labels()
        fig[i].legend(lines, labels, loc='lower right', bbox_to_anchor=(ax_pos.x1 - 0.015, ax_pos.y0 + 0.01), borderaxespad=0.)
    fig[i].show()
    if i == 0:
        pad_inch = 0.008
    else:
        pad_inch = 0.007
    fig[i].savefig(folder_output + 'sensitivity_result_' + str(metric_name[i]) + '.png', bbox_inches='tight',
                   pad_inches=pad_inch, transparent=True, dpi=600)
