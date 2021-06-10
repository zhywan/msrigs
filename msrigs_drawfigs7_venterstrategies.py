# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing searched strategies in the greedy algorithm with pruning using Venter and Ysearch datasets
# © Oct 2018-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
id_exp = '2058'  # ID for the set of experiments
scenario = 5
pruning = 1
draw_region = 1
folder_result = 'Results' + id_exp + '/realdata/'
if pruning == 1:
    folder_result += 'pruning/'
fig, axes = plt.subplots(2, 1, figsize=(5.3, 10.4))

if draw_region == 1:
    # Parameters for plotting the region
    group_size1 = 2  # number of the query results of Craig Venter’s demographic attributes with the surname
    group_size0 = 157681  # number of the query results of Craig Venter’s demographic attributes without the surname
    loss = 150
    cost = 10
    total_utility = 100
    theta_p = 0.5
    # Compute the region bounds
    U_max = 1  # U is the utility of the shared data
    U_min = max(0, 1 - (1/group_size1 - 1/group_size0) * loss /total_utility)
    r_hat_min = 0  # r_hat is the confidence score as the estimated correctness of the inferred surname
    r_hat_max = min(1, max(theta_p, group_size0 * cost / loss), max(theta_p, group_size1 * cost / loss))

# input result data
dataset = pd.read_pickle(folder_result + 'payoff_utility_confidence_s' + str(scenario) + '_3.pickle')

# plot the subfigure 1
colors = ["tab:blue", "tab:orange", "tab:red"]
customPalette = sns.set_palette(sns.color_palette(colors))
markers = {"Correct": "o", "Wrong": "X"}
sns.scatterplot(data=dataset, y='Utility', x='Confidence score', hue='Strategy',
                style='Surname inference', markers=markers, palette=customPalette, ax=axes[0], alpha=0.8)
# axes[0].legend(loc='lower left')
axes[0].set(xlim=(0.425, 0.725), ylim=(0.93, 1.00))

if draw_region == 1:
    # Plot the region
    x1 = [max(r_hat_min, 0.425), min(r_hat_max, 0.725)]
    y1 = [U_max, U_max]
    # Shade the area between y1 and line y=0
    axes[0].fill_between(x1, y1, 0,
                         facecolor="tab:blue",  # The fill color
                         color='tab:blue',       # The outline color
                         alpha=0.1)          # Transparency of the fill

# plot the subfigure 2
sns.scatterplot(data=dataset, y='Payoff', x='Confidence score', hue='Strategy',
                     style='Surname inference', markers=markers, palette=customPalette, ax=axes[1], alpha=0.8)
# axes[1].legend(loc='lower left')
axes[1].set(xlim=(0.425, 0.725))
axes[1].set(ylabel='Payoff ($)')

for i in range(2):
    axes[i].text(-0.12, 1, str(chr(ord('@') + i + 1)), fontfamily='sans-serif',
                 size=11, weight='bold', transform=axes[i].transAxes)
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig.show()
fig.savefig(folder_result + 'Venter_searched_strategies_s' + str(scenario) + '.png',  bbox_inches='tight',
            pad_inches=0.02, transparent=True, dpi=600)
