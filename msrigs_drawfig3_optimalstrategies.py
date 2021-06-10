# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing first 700 data subjects' strategies in first run of an experiment
# © Oct 2018-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Numpy 1.19.1, Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 8
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=9)    # fontsize of the x and y labels (original: MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels

id_exp = '2056b'  # ID for the set of experiments
n_S = 700
n_iter = 1
m_g = 12
ID_sub = [0, 0, 0, 0, 1, 1, 2, 3]  # scenario 3: Random masking; Scenario 5: Masking game;
# Scenario 6: No-attack masking game; Scenario 7: One-stage masking game

scenario = 6  # 3, 5, 6, 7
ns_attacks = [11, 66, 0, 47]
n_attack = ns_attacks[ID_sub[scenario]]
method = 2
pruning = 1
algorithm = 1  # 0: greedy algorithm. (3) 1: brute-force algorithm. (5, 6, 7)
folder_result = 'Results' + id_exp + '/Violin/m'+str(method) + '/'
if algorithm == 1:
    folder_result = folder_result.replace('Violin/m'+str(method), 'Violin_bf')
if pruning == 1:
    folder_result += 'pruning/'
features = ['YOB', 'State', 'DYS19', 'DYS385', 'DYS389I', 'DYS389II', 'DYS390', 'DYS391', 'DYS392',
            'DYS393', 'DYS437', 'DYS438', 'DYS439', 'DYS448']
features.reverse()
n_repeats = int(n_S * n_iter / (m_g + 2))

colors = ["tab:olive", "tab:green", "tab:cyan", "tab:blue", "tab:purple", "tab:gray"]
colors[0] = colors[scenario-2]
customPalette = sns.set_palette(sns.color_palette(colors))
df = pd.read_pickle(folder_result + 'optimal_strategy_' + str(scenario) + '.pkl')
ax = sns.jointplot(data=df, x='Data subject', y='Attribute', kind='hist', marginal_kws=dict(bins=n_S, linewidth=0,
                                                                                            binwidth=1,
                                                                                            binrange=(0, n_S+0.5)),
                   joint_kws=dict(bins=n_S, binwidth=1, binrange=(0, n_S+0.5)), space=-10, palette=customPalette)
ax.fig.set_figwidth(6)
ax.fig.set_figheight(3)
plt.yticks(np.arange(len(features)) * n_repeats + int(n_repeats/2) + 1, features)

# mark the attacked data subjects
if n_attack > 0:
    x1 = [n_S-n_attack, n_S]
    y1 = [n_S+1, n_S+1]
    # Shade the area between y1 and line y=0
    plt.fill_between(x1, y1, 0,
                     facecolor="r",  # The fill color
                     color='r',       # The outline color
                     alpha=0.2)          # Transparency of the fill

# input result data
payoffs = []
privacy = []
utility = []
column_names = ['defender_optimal', 'privacy', 'utility']
dataset = pd.read_pickle(folder_result + 'result_s' + str(scenario) + '.pickle')
for i in range(3):
    data = np.array(dataset[column_names[i]])
    av_data = np.mean(data)
    if i == 0:
        plt.text(int(5/14*n_S), n_S+12, 'Payoff: ${:.2f}'.format(av_data), fontfamily='sans-serif', fontweight='bold',
                 size=8, alpha=0.8)
    elif i == 1:
        plt.text(int(0.65*n_S), n_S+12, 'Privacy risk: {:.4f}'.format(1-av_data), fontfamily='sans-serif', fontweight='bold',
                 size=8, color='tab:red', alpha=0.8)
    else:
        plt.text(0, n_S+12, 'Utility loss: {:.4f}'.format(1-av_data), fontfamily='sans-serif', fontweight='bold',
                 size=8, color=colors[0], alpha=0.8)

plt.text(int(-2/7*n_S), int(n_S), str(chr(ord('@') + ID_sub[scenario] + 1)), fontfamily='sans-serif',
         size=10, weight='bold')
plt.show()
ax.savefig(folder_result + 'optimal_strategy_' + str(scenario) + '.png', bbox_inches='tight', pad_inches=0.01, transparent=True, dpi=525)
