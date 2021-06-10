# Multi-Stage Re-Identification (MSRI) Game Solver (GS) v1.1
# Component: Showing results in scatter plot (fairness wrt privacy, and fairness wrt usefulness)
# © Oct 2021-2021 Zhiyu Wan, HIPLAB, Vanderbilt University
# Compatible with python 3.8.5. Package dependencies: Pandas 1.1.3, Matplotlib 3.3.1, Seaborn 0.11.0
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pruning = 1
folder_result = 'Results2058/Violin/m2/'
if pruning == 1:
    folder_result += 'pruning/'
column_names = ['privacy', 'utility']
scenario_name = ['No-protection', 'K-anonymity', 'Random masking',
                 'Opt-in game', r'Masking game ($\alpha$=0)', r'Masking game ($\alpha$=1)']
privacy_fairness = [0.822, 0.993, 0.995, 0.99, 0.998, 0.997]
usefulness_fairness = [1, 0.752, 0.801, 0.922, 0.783, 0.977]

scenarios = scenario_name

dataset2 = pd.DataFrame({'Fairness with respect to privacy': privacy_fairness, 'Fairness with respect to usefulness': usefulness_fairness, 'Scenario': scenarios})
colors = ["tab:red", "tab:orange", "tab:green", "tab:cyan", "tab:blue", "tab:purple"]
# ‘tab:blue’,‘tab:orange’,‘tab:green’,‘tab:red’,‘tab:purple’,‘tab:brown’,‘tab:pink’,‘tab:gray’,‘tab:olive’,‘tab:cyan’
customPalette = sns.set_palette(sns.color_palette(colors))
markers = {"No-protection": "X", "K-anonymity": "^", "Random masking": "v", "Opt-in game": "o",
           r"Masking game ($\alpha$=0)": "s", r"Masking game ($\alpha$=1)": "D"}
ax = sns.scatterplot(data=dataset2, x='Fairness with respect to usefulness', y='Fairness with respect to privacy', s=100, hue='Scenario', style='Scenario', markers=markers,
                     palette=customPalette)
ax.legend(loc='lower left')
fig = plt.gcf()
fig.set_size_inches(5.25, 5.25)
plt.show()
ax.figure.savefig(folder_result + 'result_figure_S2.png', bbox_inches='tight',
                pad_inches=0.0285, transparent=True, dpi=600)
