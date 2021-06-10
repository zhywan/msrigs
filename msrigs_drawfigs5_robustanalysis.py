# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing robust analysis results in line plots (prob. of success, payoff, privacy)
# © Oct 2021-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
# Update history:
# April 16, 2021: changed for the robust analysis
# April 18, 2021: 10 scenarios
# May 12, 2021: 9 sub-figures in one plot
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
n_scenario = 10
pruning = 1
scenario_id = [0, 3, 4, 4, 4, 5, 5, 5, 4, 5]
# 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking.
# 3.2: random masking 2 (sharing rate = 0.15).
# 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.
order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_fig = 9  # Number of lineplot figures
n_row = 3  # number of rows of subplots in each figure
n_col = 3  # number of collums of subplots in each figure
fig, axes = plt.subplots(n_row, n_col + 1, figsize=(16.5, 12))

metric_name = ['payoff', 'privacy', 'utility', 'success_rate']
column_names = ['success_rate', 'privacy', 'defender_optimal']
experiments_name = ['cost', 'nG', 'nI']
experiments_name = [experiments_name[i] + '_robust' for i in range(len(experiments_name))]
xlabels = ['Actual cost of attack ($)', 'Actual number of records in the genetic genealogy dataset',
           'Actual number of records in the identified dataset']

ms_g_start = [0, 1, 1]
n_ms = [17, 20, 20]
ms_g_end = [17, 21, 21]
steps = [10, 2000, 2000]
default_xs = [10, 20000, 20000]

for fig_i in range(3):  # each subplot
    if fig_i == 0:
        scenario_name = ['No-protection', 'Random masking', r'Opt-in game ($C$=\$10)', r'Opt-in game ($C$=\$20)', r'Opt-in game ($C$=\$30)',
                         r'Masking game ($C$=\$10)', r'Masking game ($C$=\$20)', r'Masking game ($C$=\$30)', 'Opt-in game', 'Masking game']
    elif fig_i == 1:
        scenario_name = ['No-protection', 'Random masking', r'Opt-in game ($n_G$=10,000)', r'Opt-in game ($n_G$=20,000)',
                         r'Opt-in game ($n_G$=30,000)', r'Masking game ($n_G$=10,000)', r'Masking game ($n_G$=20,000)',
                         r'Masking game ($n_G$=30,000)', 'Opt-in game', 'Masking game']
    else:  # fig_i ==2:
        scenario_name = ['No-protection', 'Random masking', r'Opt-in game ($n_I$=10,000)', r'Opt-in game ($n_I$=20,000)',
                         r'Opt-in game ($n_I$=30,000)', r'Masking game ($n_I$=10,000)', r'Masking game ($n_I$=20,000)',
                         r'Masking game ($n_I$=30,000)', 'Opt-in game', 'Masking game']
    scenario_name = np.array(scenario_name)
    scenario_name_in_order = scenario_name[order]

    experiment_name = experiments_name[fig_i]
    xlabel = xlabels[fig_i]
    m_g_start = ms_g_start[fig_i]
    m_g_end = ms_g_end[fig_i]
    step = steps[fig_i]
    default_x = default_xs[fig_i]


    folder_result = 'Results' + id_exp + '/' + experiment_name + '/'
    if pruning == 1:
        folder_result += 'pruning/'
    folder_output = 'Results' + id_exp + '/'
    
    # input result data (payoff, privacy, and utility)
    n_m = m_g_end - m_g_start
    payoffs = []
    privacy = []
    utility = []
    success_rate = []
    for jj in range(n_scenario):
        j = scenario_id[jj]  # re-order for plotting (filename)
        if jj in [0, 1]:
            dataset = pd.read_pickle(folder_result + 'result_p' + str(fig_i) + '_ts' + str(j) + '.pickle')
        elif jj in [3, 6]:
            if fig_i == 0:  # cost robust
                dataset = pd.read_pickle('Results' + id_exp + '_cost20/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
            else:
                dataset = pd.read_pickle(folder_result + 'result_p' + str(fig_i) + '_ts' + str(j) + '.pickle')
        elif jj in [2, 5]:
            if fig_i == 0:  # cost robust
                dataset = pd.read_pickle(folder_result + 'result_p' + str(fig_i) + '_ts' + str(j) + '.pickle')
            if fig_i == 1:  # nG robust
                dataset = pd.read_pickle('Results' + id_exp + '_nG10000/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
            if fig_i == 2:  # nI robust
                dataset = pd.read_pickle('Results' + id_exp + '_nI10000/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
        elif jj in [4, 7]:
            if fig_i == 0:  # cost robust
                dataset = pd.read_pickle('Results' + id_exp + '_cost30/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
            if fig_i == 1:  # nG robust
                dataset = pd.read_pickle('Results' + id_exp + '_nG30000/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
            if fig_i == 2:  # nI robust
                dataset = pd.read_pickle('Results' + id_exp + '_nI30000/' + experiment_name + '/pruning/result_p'
                                         + str(fig_i) + '_ts' + str(j) + '.pickle')
        else:  # jj in [8, 9]
            if fig_i == 0:  # cost robust
                dataset = pd.read_pickle('Results' + id_exp + '/costchanging/pruning/result_p7_s' + str(j) + '.pickle')
            if fig_i == 1:  # nG robust
                dataset = pd.read_pickle('Results' + id_exp + '/nGchanging/pruning/result_p3_s' + str(j) + '.pickle')
            if fig_i == 2:  # nI robust
                dataset = pd.read_pickle('Results' + id_exp + '/nIchanging/pruning/result_p4_s' + str(j) + '.pickle')
        for i in range(len(column_names)):
            data = np.array(dataset[column_names[i]])
            shaped_dataset = np.reshape(data, (n_iter * n_m, n_S))
            av_dataset = np.mean(shaped_dataset, axis=1)
            if i == 0:
                payoffs.extend(av_dataset)
            elif i == 1:
                privacy.extend(av_dataset)
            elif i == 2:
                utility.extend(av_dataset)
            else:  # i == 3
                success_rate.extend(av_dataset)

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
    results = [payoffs, privacy, utility, success_rate]
    ylabels = ["Probability of an attack's success", 'Privacy', 'Average payoff ($)']
    colors = ["tab:red", "tab:orange", "tab:olive", "tab:cyan", "tab:green", "tab:pink", "tab:blue", "tab:purple",
              "tab:brown", "tab:gray"]
    colors = np.array(colors)
    colors_in_order = colors[order]
    customPalette = sns.set_palette(sns.color_palette(colors_in_order))
    markers_base = ["X", "v", "o", "o", "o", "s", "s", "s", "o", "s"]
    markers = dict(zip(scenario_name, markers_base))
    for i in range(len(column_names)):
        fig_row = (fig_i * 3 + i) // n_col
        fig_col = (fig_i * 3 + i) % n_col
        dataset = pd.DataFrame({ylabels[i]: results[i],
                                'Scenario': scenarios,
                                xlabel: m_g,
                                })
        # plot the default vertical lines
        axes[fig_row, fig_col].axvline(x=default_x, label='Default value', color='0.5', linestyle='--')

        sns.lineplot(data=dataset, x=xlabel, y=ylabels[i], hue='Scenario', markers=markers, style='Scenario',
                          palette=customPalette, ax=axes[fig_row, fig_col], ci='sd')  # style_order=style_order,
        axes[fig_row, fig_col].legend_.remove()
        axes[fig_row, fig_col].text(-0.1, 0.98, str(chr(ord('@') + fig_i * 3 + i + 1)), size=9,
                                    fontfamily='sans-serif', weight='bold',
                                    transform=axes[fig_row, fig_col].transAxes)
# Adjust, show and save each figure
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.15)
for i in range(3):
    axes[i, -1].axis('off')
    ax_pos = axes[i, -1].get_position()
    lines, labels = axes[i, -2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(ax_pos.x0, ax_pos.y1 + 0.035 - i * 0.015), borderaxespad=0.)
fig.show()
fig.savefig(folder_output + 'robust_result.png', #bbox_inches='tight',
            pad_inches=0.006, transparent=True, dpi=300)
