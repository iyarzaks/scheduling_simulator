import pandas as pd
from simulator import simulate_scheduling
import ast
import pickle

jobs_df = pd.read_csv('integer_problem_jobs.csv').fillna('')
with open('integer_problem_servers.pkl', 'rb') as f:
    servers_df = pickle.load(f)
result = simulate_scheduling(jobs_df=jobs_df, servers_df=servers_df, algorithm='WSVF')
print (jobs_df.shape[0]* result)