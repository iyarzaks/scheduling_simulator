from mip import *
import random
import pandas as pd
import pickle
from constants import RUN_IN, STARTED_AT
import numpy as np
from mip.constants import GRB


def create_problem(problem_name, number_of_machines=3, number_of_jobs=8, write=True, alpha=0.5, prediction_std=0,
                   estimation_errors='normal'):
    print(f'prediction_std: {prediction_std}')
    machines_capacity = 1
    weights = [random.randint(1, 10) for _ in range(number_of_jobs)]
    demands = [round(random.uniform(0.1, alpha), 2) for _ in range(number_of_jobs)]
    processing_times = [random.randint(1, 6) for _ in range(number_of_jobs)]
    if estimation_errors == 'normal':
        estimated_processing_times = [max(round(random.normalvariate(mu=p, sigma=p*prediction_std)),
                                          1) for p in processing_times]
    else:
        estimated_processing_times = [max(round(random.uniform(a=p-prediction_std,
                                                               b=p+prediction_std)), 1) for p in processing_times]
    arriving_times = [random.randint(1, 30) for _ in range(number_of_jobs)]

    servers = [{'server_name': f'server_{i}', 'D': [(pd.Interval(left=0, right=np.inf),
                                                     machines_capacity)]} for i in range(1, number_of_machines+1)]
    servers_df = pd.DataFrame(servers)
    if write:
        with open(f'{problem_name}/servers.pkl', 'wb') as f:
            pickle.dump(servers_df, f)

    problem_df = pd.DataFrame(columns=['p', 'd', 'w', 'a', RUN_IN, STARTED_AT, 'finished_at'])
    problem_df['job_name'] = [f'job_{j}' for j in range(1, number_of_jobs+1)]
    problem_df[RUN_IN] = ''
    problem_df[STARTED_AT] = ''
    problem_df['finished_at'] = ''
    problem_df['p'] = estimated_processing_times
    problem_df['r_p'] = processing_times
    problem_df['d'] = demands
    problem_df['w'] = weights
    problem_df['a'] = arriving_times
    if write:
        problem_df.to_csv(f'{problem_name}/jobs.csv')
    return servers_df, problem_df


def create_integer_problem(problem_name, servers_df=None, jobs_df=None, write=True):
    if jobs_df is None:
        jobs_df = pd.read_csv(f'{problem_name}/jobs.csv').fillna('')
    if servers_df is None:
        with open(f'{problem_name}/servers.pkl', 'rb') as pkl_f:
            servers_df = pickle.load(pkl_f)
    m = Model(sense=MINIMIZE, solver_name=GRB)  # use GRB for Gurobi
    number_of_machines = servers_df.shape[0]
    number_of_jobs = jobs_df.shape[0]
    processing_times = jobs_df['p'].to_list()
    demands = jobs_df['d'].to_list()
    weights = jobs_df['w'].to_list()
    arriving_times = jobs_df['a'].to_list()
    machines_capacity = servers_df.iloc[0]['D'][0][1]
    T_old = sum(processing_times) // number_of_machines + 1
    T = max(arriving_times) + int(np.ceil((sum(processing_times[j] * demands[j] for j in range(number_of_jobs))
                     // number_of_machines) * (1/(1-max(demands)))))
    # a_variables = []
    # for j in range(1, number_of_jobs+1):
    #     job_list = []
    #     for i in range(1, number_of_machines+1):
    #         job_machine_list = []
    #         for t in range(1, T+1):
    #             job_machine_list.append(m.add_var(name=f'a_{j},{i},{t}', var_type=BINARY))
    #         job_list.append(job_machine_list)
    #     a_variables.append(job_list)

    # A_variables = []
    #
    # for j in range(1, number_of_jobs+1):
    #     job_list = []
    #     for i in range(1, number_of_machines+1):
    #         job_list.append(m.add_var(name=f'A_{j},{i}', var_type=BINARY))
    #     A_variables.append(job_list)

    starting_times = []
    for j in range(1, number_of_jobs+1):
        job_list = []
        for i in range(1, number_of_machines+1):
            job_machine_list = []
            for t in range(0, T):
                job_machine_list.append(m.add_var(name=f's_{j},{i},{t}', var_type=BINARY))
            job_list.append(job_machine_list)
        starting_times.append(job_list)





    completion_times = []
    for j in range(1, number_of_jobs+1):
        completion_times.append(m.add_var(name=f'c_{j}', var_type=CONTINUOUS, ub=T))

    # c_j >= t * s_j_i_t + p_j

    for j in range(number_of_jobs):
        for i in range(number_of_machines):
            for t in range(T):
                m += t * starting_times[j][i][t] + processing_times[j] <= completion_times[j]


    for j in range(number_of_jobs):
        for i in range(number_of_machines):
            for t in range(T):
                if t < arriving_times[j]:
                    m += starting_times[j][i][t] == 0



    # sum over t,j (s_j_i_t) = 1
    for j in range(number_of_jobs):
        sums_list = []
        for i in range(number_of_machines):
            sums_list.append(sum(starting_times[j][i]))
        m += sum(sums_list) == 1

    #

    for i in range(number_of_machines):
        for t in range(0, T):
            total = []
            for j in range(number_of_jobs):
                total.append(demands[j] * sum(starting_times[j][i][max(0, t-processing_times[j]+1):t+1]))
            m += sum(total) <= machines_capacity


    # c_j >= t * a_j_i_t

    # for j in range(number_of_jobs):
    #     for i in range(number_of_machines):
    #         for t in range(T):
    #             m += (t+1) * a_variables[j][i][t] <= completion_times[j]

    # p_j * a_j_i_t - p_j * a_j_i_t+1 + sum over v>=t+2 (a_j_i_v) <= p_j

    # for j in range(number_of_jobs):
    #     for i in range(number_of_machines):
    #         for t in range(T-2):
    #             m += processing_times[j] * a_variables[j][i][t] - processing_times[j] * a_variables[j][i][t+1] + \
    #                 sum(a_variables[j][i][t+2:]) <= processing_times[j]

    # sum over all i,t (a_j_i_t) >= p_j
    # for j in range(number_of_jobs):
    #     sums_list = []
    #     for i in range(number_of_machines):
    #         sums_list.append(sum(a_variables[j][i]))
    #     m += processing_times[j] <= sum(sums_list)

    # sum over j (d_j * a_j_i_t) <= D_i
    # for i in range(number_of_machines):
    #     for t in range(T):
    #         m += xsum(demands[j] * a_variables[j][i][t] for j in range(number_of_jobs)) <= machines_capacity

    # sum over i (A_j_i) = 1
    # for j in range(number_of_jobs):
    #     m += sum(A_variables[j]) == 1

    #  sum over t (a_j_i_t) <= p_j * A_j_i
    # for j in range(number_of_jobs):
    #     for i in range(number_of_machines):
    #         m += sum(a_variables[j][i]) <= processing_times[j] * A_variables[j][i]

    m.objective = minimize(xsum(weights[j] * (completion_times[j]-arriving_times[j]) for j in range(number_of_jobs)))

    if write:
        m.write(f'model.lp')
    return m




