import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

data = pd.read_csv('online_comparison_5.csv')
stds = np.array(range(1000, 4010, 1000))
# plt.ticklabel_format(style='plain')
# number_of_machines = list(range(20,90,20))
for alg, marker, color in zip(['WSVF','SVF','WSPT','SPT'],
                                         ['.', 'v', 'x', '^', '<', '>'],
                                         ['r', 'c', 'm', 'g', 'b']):
    results = data[alg] / data['WSVF']
    plt.plot(stds, results, marker=marker, label=alg, color=color)
# plt.title('Constant number of machines = 50')
plt.ylabel('Average weighted flow time')
# plt.xlabel('Errors interval')
plt.xlabel('Standard deviation')
plt.legend(loc='upper left')
plt.savefig(f'plot_ratios_online.png')
# with open(file=f'json_results/{experiment_name}.json', mode='w') as f:
#     json.dump(fp=f, obj=all_results)
