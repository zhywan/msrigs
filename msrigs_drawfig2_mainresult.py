# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing results in violin plot (payoff) and in scatter plot (privacy, and utility)
# © Oct 2018-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

id_exp = '2058'  # ID for the set of experiments
n_iter = 100
n_S = 1000
n_scenario = 8
method = 2
pruning = 1
algorithm = 1  # 0: greedy algorithm. 1: brute-force algorithm.
save_iter = [False, False, False, False, False, True, True, False]  # save iteration for each scenario
folder_result = 'Results' + id_exp + '/Violin/m'+str(method) + '/'
if pruning == 1:
    folder_result += 'pruning/'
order = [0, 1, 2, 3, 4, 5, 6, 7]  # 0: no protection. 1: no genomic data sharing. 2: random opt-in. 3: random masking.
# 4: opt-in game. 5: masking game. 6: no-attack masking game. 7: one-stage masking game.
fig, axes = plt.subplots(2, 1, figsize=(5.35, 11.4), gridspec_kw={'height_ratios': [0.95, 1]})

# input result data
payoffs = []
privacy = []
utility = []
column_names = ['defender_optimal', 'privacy', 'utility']
for jj in range(n_scenario):
    j = order[jj]  # re-order for plotting
    if algorithm == 1 and j >= 5:
        folder_result = folder_result.replace('Violin/m' + str(method), 'Violin_bf')
    if not save_iter[j]:
        dataset = pd.read_pickle(folder_result + 'result_s' + str(j) + '.pickle')
    for i in range(3):  # for each metric
        if save_iter[j]:
            shaped_data = np.empty([n_iter, n_S])
            for k in range(n_iter):  # for each iteration
                dataset = pd.read_pickle(folder_result + 'result_s' + str(j) + '_i' + str(k) + '.pickle')
                data = np.array(dataset[column_names[i]])
                shaped_data[k, :] = data.reshape(1, -1)
        else:
            data = np.array(dataset[column_names[i]])
            shaped_data = np.reshape(data, (n_iter, n_S))
        av_data = np.mean(shaped_data, axis=1)
        std_data = np.std(shaped_data, axis=1)
        print('s' + str(j) + ' ' + column_names[i] + ' av: ' + str(av_data[0]))
        print('std: ' + str(std_data[0]))
        if i == 0:
            payoffs.extend(av_data)
        elif i == 1:
            privacy.extend(av_data)
        else:
            utility.extend(av_data)

# plot the subfigure 1
scenario_name = np.array(['No-protection', 'Demographics-only', 'Random opt-in', 'Random masking', 'Opt-in game',
                          'Masking game', 'No-attack masking game', 'One-stage masking game'])
scenario_name_in_order = scenario_name[order]
scenarios = []
for i in range(n_scenario):
    label = [scenario_name_in_order[i] for j in range(n_iter)]
    scenarios.extend(label)
colors = ["tab:red", "tab:orange", "tab:olive", "tab:green", "tab:cyan", "tab:blue", "tab:purple", "tab:gray"]
colors = np.array(colors)
colors_in_order = colors[order]

customPalette = sns.set_palette(sns.color_palette(colors))
dataset2 = pd.DataFrame({'Average payoff ($)': payoffs, 'Scenario': scenarios})
sns.violinplot(data=dataset2, x='Scenario', y='Average payoff ($)', scale='width', saturation=1,
               palette=customPalette, gridsize=100, width=0.8, ax=axes[0])  # linewidth=2,
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

# plot the subfigure 2
scenario_name = np.array(['No-protection', 'Demographics-only', 'Random opt-in', 'Random masking', 'Opt-in game',
                          'Masking game', 'No-attack masking game', 'One-stage masking game'])
scenario_name_in_order = scenario_name[order]
scenarios = []
for i in range(n_scenario):
    label = [scenario_name_in_order[i] for j in range(n_iter)]
    scenarios.extend(label)
markers_base = ["X", "d", "^", "v", "o", "s", "P", "D"]
markers = dict(zip(scenario_name, markers_base))
dataset2 = pd.DataFrame({'Privacy': privacy, 'Utility': utility, 'Scenario': scenarios})
sns.scatterplot(data=dataset2, x='Utility', y='Privacy', s=50, hue='Scenario', style='Scenario', markers=markers,
                palette=customPalette, ax=axes[1], alpha=0.8)
axes[1].legend(loc='lower left')
axes[1].set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xticks=np.array(range(11))/10, yticks=np.array(range(11))/10)
for i in range(2):
    axes[i].text(-0.12, 1, str(chr(ord('@') + i + 1)), fontfamily='sans-serif',
                 size=11, weight='bold', transform=axes[i].transAxes)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig.show()
if algorithm == 0:
    fig.savefig(folder_result + 'result_payoff_privacy_utility.png',  bbox_inches='tight',
                pad_inches=0.01, transparent=True, dpi=600)
else:
    fig.savefig(folder_result.replace('Violin/m' + str(method), 'Violin_bf') + 'result_payoff_privacy_utility.png',
                bbox_inches='tight',
                pad_inches=0.005, transparent=True, dpi=600)
