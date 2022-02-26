import json

import numpy as np
import pandas as pd
# with open ('json_results/test_4.json', 'r') as f:
#     data= json.load(f)
# df = pd.DataFrame(columns=range(15000, 31000, 5000))
# for i, alg in enumerate(data.keys()):
#     df = df.append({}, ignore_index=True)
#     df.iloc[i] = np.array(data[alg][1:]) / 10
# df.index = data.keys()
# df = df.applymap(int)
# df.to_csv('table_4.csv')
df = pd.read_csv('expirements_results/online_comparison_stds_normal_2.csv')
df= df.transpose()
df = df.applymap(int)

df.to_csv('table_8.csv')