import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('wsvf_opt_compare_11.csv')
alphas = np.array(range(5, 55, 5))
alphas = alphas / 100
# const_job_df = df[df['N']==25]
# const_job_df.drop_duplicates(subset=['M','N'], inplace=True)
# for col, marker, color in zip(['WSVF', 'OPT'], ['.', 'v', 'x', '^', '<', '>'], ['r', 'c', 'y', 'g', 'b']):
#     plt.plot(df['M'], df[col], marker=marker, label=col, color=color)
plt.plot(alphas, df['WSVF'] / df['OPT'], marker='.', color='g')
# plt.title('Constant ratio of jobs per machine = 5')
# plt.title('Constant number of machines = 3 and jobs =21')
plt.ylabel('wsvf / opt ratio')
plt.xlabel('alpha')
# plt.ylabel('Total weighted completion time')
# plt.xlabel('number of machines')
# plt.legend()
plt.savefig(f'figures_to_show/OPT_WSVF_comp_new_alpha_2.png')
# plt.close()
# plt.plot(const_job_df['N']/const_job_df['M'], const_job_df['WSVF'] / const_job_df['OPT'], marker='x', label='ratio',
#          color='g')
# plt.ylabel('WSVF / OPT ratio')
# plt.xlabel('jobs per machine')
# plt.legend()
# plt.savefig(f'figures_to_show/ratio_const_jobs.png')
# plt.close()
#
# const_machines_df = df[df['M']==3]
# const_machines_df = const_machines_df.drop_duplicates(subset=['M','N'], keep='last').sort_values('N')
# for col, marker, color in zip(['WSVF', 'OPT'], ['.', 'v', 'x', '^', '<', '>'], ['r', 'c', 'y', 'g', 'b']):
#     plt.plot(const_machines_df['N'], const_machines_df[col], marker=marker, label=col, color=color)
#
# plt.title('Constant number of machines = 3')
# plt.ylabel('Total weighted completion time')
# plt.xlabel('Number of jobs')
# plt.legend()
# plt.savefig(f'figures_to_show/OPT_WSVF_comp_const_machines.png')
# plt.close()
# plt.plot(const_machines_df['N']/const_machines_df['M'], const_machines_df['WSVF'] / const_machines_df['OPT'], marker='x', label='ratio',
#          color='g')
# plt.ylabel('WSVF / OPT ratio')
# plt.xlabel('jobs per machine')
# plt.legend()
# plt.savefig(f'figures_to_show/ratio_const_machines.png')