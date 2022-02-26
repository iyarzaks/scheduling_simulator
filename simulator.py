import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from copy import deepcopy
from constants import RUN_IN, STARTED_AT
import os
import json
import matplotlib.pyplot as plt

import time

def measure_time(f):

  def timed(*args, **kw):
    ts = time.time()
    result = f(*args, **kw)
    te = time.time()

    print ('%r (%r, %r) %2.2f sec' % (f.__name__, args, kw, te-ts))
    return result
  return timed


def make_data(real_data=False, N=3000, M=50):
    if real_data:
        servers_size = 512
        jobs_df = pd.read_csv('jobs_data.csv')
        jobs_df = jobs_df[jobs_df['p']<7000]
        jobs_df = jobs_df.sample(N)
        jobs_df['job_name'] = [f'job_{i+1}' for i in jobs_df.index]

    else:
        servers_size = 200
        jobs = list()
        for i in range(1, N + 1):
            rand = random.uniform(0, 1)
            if rand < 0.7:
                p = random.randint(1, 100)
            elif rand < 0.85:
                p = random.randint(300, 350)
            else:
                p = random.randint(450, 500)
            jobs.append({'job_name': f'job_{i}', 'p': p,
                         'd': random.randint(1, 100) if i < 0.75 * N else random.randint(100, 200)})
        jobs_df = pd.DataFrame(jobs)
    # jobs_df['w'] = [random.uniform(0.1, 5) for i in range(jobs_df.shape[0])]
    # jobs_df['s'] = [random.choice([0, 50]) for i in range(jobs_df.shape[0])]
    jobs_df['w'] = [random.uniform(jobs_df['d'].min(), jobs_df['d'].max())/20 for i in range(jobs_df.shape[0])]
    jobs_df[RUN_IN] = ''
    jobs_df[STARTED_AT] = ''
    jobs_df['finished_at'] = ''
    jobs_df = shuffle(jobs_df)
    servers = [{'server_name': f'server_{i}', 'D': [(pd.Interval(left=0, right=np.inf), servers_size)]} for i in range(1, M+1)]
    servers_df = pd.DataFrame(servers)
    return jobs_df, servers_df


def build_I_star(jobs_df, machines_size):
    jobs_df['p'] = (jobs_df['p'] * jobs_df['d']) / machines_size
    jobs_df['d'] = machines_size
    return jobs_df


def check_avail_start_time(server, demand, time, current_time):
    first_start_time = current_time
    remaining_time = time
    for section in server['D']:
        times_interval, capacity = section
        if capacity >= demand and current_time < times_interval.right:
            if current_time < times_interval.left:
                remaining_time = remaining_time - times_interval.length
            else:
                remaining_time = remaining_time - (times_interval.right - current_time)
            if remaining_time <= 0:
                break
        else:
            first_start_time = max(times_interval.right, current_time)
            remaining_time = time
    return first_start_time


def find_last_interval(x):
    try:
        return x[-2][1]
    except:
        return x[-1][1]



def find_first_fit(servers_df, job,current_time, specific_server=None, randomize=False, dynamic=False):
    servers_df['optional_start_time'] = servers_df.apply(check_avail_start_time,axis=1, current_time=current_time,
                                                         demand=job['d'],
                                                         time=job['p'])
    if dynamic and min(servers_df['optional_start_time'] ) != current_time:
        return None
    if specific_server is not None:
        return servers_df[servers_df['server_name'] == specific_server].iloc[0]
    minimum_starting_time = min(servers_df['optional_start_time'])
    servers_with_minimal = servers_df[servers_df['optional_start_time'] == minimum_starting_time]
    servers_with_minimal['free_space'] = servers_with_minimal['D'].apply(find_last_interval)
    if randomize:
        return servers_with_minimal.sort_values('free_space').iloc[0]
        # return servers_with_minimal.iloc[random.randint(0, servers_with_minimal.shape[0]-1)]
    return servers_with_minimal.iloc[0]


def update_servers(server, start_time, finish_time, demand, servers_df):
    new_sections = []
    index_to_change = servers_df[servers_df['server_name'] == server].index[0]
    sections_before = servers_df[servers_df['server_name'] == server].iloc[0]['D']
    for section in sections_before:
        interval, capacity = section
        if start_time >= interval.right:
            new_sections.append((interval, capacity))
        elif start_time == interval.left and finish_time <= interval.right:
            new_sections.extend([(pd.Interval(start_time, finish_time), capacity - demand),
                                 (pd.Interval(finish_time, interval.right), capacity)])
        elif start_time == interval.left and finish_time > interval.right:
            new_sections.extend([(pd.Interval(start_time, interval.right), capacity - demand)])
        elif start_time < interval.left and interval.left < finish_time and  finish_time < interval.right:
            new_sections.extend([(pd.Interval(interval.left, finish_time), capacity - demand),
                                 (pd.Interval(finish_time, interval.right), capacity)])
        elif start_time < interval.left < interval.right <= finish_time:
            new_sections.extend([(pd.Interval(interval.left, interval.right), capacity - demand)])
        elif finish_time <= interval.left:
            new_sections.append((interval, capacity))
        elif start_time > interval.left and finish_time<interval.right:
            new_sections.extend([(pd.Interval(interval.left, start_time), capacity),
                                 (pd.Interval(start_time, finish_time), capacity-demand),
                                 (pd.Interval(finish_time, interval.right), capacity)])
        elif start_time > interval.left and start_time < interval.right and finish_time > interval.right:
            new_sections.extend([(pd.Interval(interval.left, start_time), capacity),
                                 (pd.Interval(start_time, interval.right), capacity-demand)])
    if new_sections[-1][0].right != np.inf:
        print(new_sections)
    servers_df.at[index_to_change, 'D'] = new_sections
    return servers_df


def simulate_scheduling(jobs_df=None, servers_df=None, data_tuple=None, algorithm='RANDOM', opt=False, prints=True,
                        randomize_server_fit=False):
    if data_tuple is not None:
        jobs_df, servers_df = data_tuple
    servers_size = servers_df.iloc[0]['D'][0][1]
    jobs_df['v'] = jobs_df['p'] * jobs_df['d']
    jobs_df['v/w'] = jobs_df['v'] / jobs_df['w']
    jobs_df['p/w'] = jobs_df['p'] / jobs_df['w']

    if algorithm == 'HYBRID':
        high_demand_jobs = jobs_df[jobs_df['d']/servers_size > 0.5]
        low_demand_jobs = jobs_df[jobs_df['d']/servers_size <= 0.5]
        servers_for_low_count = int(np.ceil(2*(servers_df.shape[0]-2)/3) + 1)
        servers_for_low_df = servers_df.iloc[:servers_for_low_count]
        servers_for_high_df = servers_df.iloc[servers_for_low_count:]
        high_demand_jobs_cost = simulate_scheduling(jobs_df=high_demand_jobs, servers_df=servers_for_high_df.copy(), algorithm='WSPT', prints=False)
        low_demand_jobs_cost = simulate_scheduling(jobs_df=low_demand_jobs, servers_df=servers_for_low_df.copy(), algorithm='WSVF_batch_dispatch', prints=False)
        return low_demand_jobs_cost + high_demand_jobs_cost

    # if algorithm == 'TWO_STEP_SVF':
    #     I_star_df = build_I_star(jobs_df=jobs_df.copy(), machines_size=servers_size)
    #     simulate_scheduling(jobs_df=I_star_df, servers_df=servers_df.copy(), algorithm='SVF', prints=False)
    #     jobs_df[RUN_IN] = I_star_df[RUN_IN]
    # if algorithm == 'RANDOM_TWO_STEP_SVF' or algorithm == 'RANDOM_TWO_STEP_WSVF':
    #     jobs_df[RUN_IN] = np.random.choice(servers_df['server_name'], size=jobs_df.shape[0])
    # if algorithm == 'TWO_STEP_WSVF':
    #     I_star_df = build_I_star(jobs_df=jobs_df.copy(), machines_size=servers_df.iloc[0]['D'][0][1])
    #     simulate_scheduling(jobs_df=I_star_df, servers_df=servers_df.copy(), algorithm='WSVF_batch_dispatch', prints=False)
    #     jobs_df[RUN_IN] = I_star_df[RUN_IN]

    i = 0
    current_time = 0
    min_available_time = 0
    print(f'\nRun algorithm {algorithm}\n')
    with tqdm(total=jobs_df.shape[0]) as pbar:
        while '' in list(jobs_df[STARTED_AT]):
            if min_available_time > current_time:
                current_time = min_available_time
            unscheduled_available_jobs_cond = (jobs_df[STARTED_AT] == '') & (jobs_df['a'] <= current_time)
            unscheduled_available_jobs = jobs_df[unscheduled_available_jobs_cond]['job_name']
            if len(unscheduled_available_jobs) == 0:
                current_time += 1
                continue
            if algorithm == 'RANDOM':
                first_in_line = unscheduled_available_jobs.iloc[0]
            elif algorithm == 'SPT':
                first_in_line = jobs_df[unscheduled_available_jobs_cond].sort_values(by='p').iloc[0]['job_name']
            elif algorithm == 'SVF':
                first_in_line = jobs_df[unscheduled_available_jobs_cond].sort_values(by='v').iloc[0]['job_name']
            elif algorithm == 'WSPT':
                first_in_line = jobs_df[unscheduled_available_jobs_cond].sort_values(by='p/w').iloc[0]['job_name']
            elif algorithm == 'WSVF':
                first_in_line = jobs_df[unscheduled_available_jobs_cond].sort_values(by='v/w').iloc[0]['job_name']

            alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0],
                                          current_time, randomize=randomize_server_fit, dynamic=True)
            if alloc_server is None:
                current_time += 1
                continue

            elif algorithm == 'WSVF_DINAMIC_with_bypassing':
                for i in range(min(jobs_df[unscheduled_available_jobs_cond].shape[0], 20)):
                    first_in_line = jobs_df[unscheduled_available_jobs_cond].sort_values(by='v/w').iloc[i]['job_name']
                    alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0],
                                                  current_time, randomize=randomize_server_fit, dynamic=True)
                    if alloc_server is not None:
                        break
                if alloc_server is None:
                    current_time += 1
                    continue

            # if algorithm not in ['WSVF_DINAMIC','WSVF_DINAMIC_with_bypassing']:
            #     alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0],
            #                                   current_time, randomize=randomize_server_fit)


            # updating

            jobs_df.loc[jobs_df['job_name'] == first_in_line, RUN_IN] = alloc_server['server_name']
            jobs_df.loc[jobs_df['job_name'] == first_in_line, STARTED_AT] = alloc_server['optional_start_time']
            finish_time = alloc_server['optional_start_time'] + \
                          jobs_df[jobs_df['job_name'] == first_in_line].iloc[0]['r_p']
            jobs_df.loc[jobs_df['job_name'] == first_in_line, 'finished_at'] = finish_time

            servers_df = update_servers(server=alloc_server['server_name'], start_time=alloc_server['optional_start_time'],
                                        finish_time=finish_time,
                                        demand=jobs_df[jobs_df['job_name'] == first_in_line].iloc[0]['d'],
                                        servers_df=servers_df)
            # min_available_time = min(servers_df['optional_start_time'])
            i += 1
            if i % 100 == 0:
                pbar.update(n=100)
        pbar.update(n=(i % 100))
    # servers_df['utilization'] = servers_df['D'].apply(calc_utilization, server_size=servers_size)
    # if prints:
    #     if opt:
    #         print(f"\nOPT* {algorithm} RESULTS")
    #     else:
    #         print(f"\n{algorithm} RESULTS")
    #     print("-"*20)
    #     print(f"Average waiting time: {(jobs_df['finished_at']-jobs_df['p']).mean()}\n")
    #     print(f"Average weighted waiting time: {(jobs_df['w']*(jobs_df['finished_at'] - jobs_df['p'])).mean()}\n")
    #     print(f"Average flow time: {jobs_df['finished_at'].mean()}\n")
    #     print(f"Average weighted flow time: {(jobs_df['w']*jobs_df['finished_at']).mean()}\n")
    #     # print(f"Utilization: {servers_df['utilization'] .mean()}\n")
    #     plot_utilization(servers_df['D'], server_size=servers_size, algorithm=algorithm)
    # if 'WS' in algorithm or algorithm == 'RANDOM':
    return (jobs_df['w']*(jobs_df['finished_at'] - jobs_df['a'])).mean()
    # return jobs_df['finished_at'].sum()


def find_interval(D_over_time, x):
    for i in range(len(D_over_time)):
        if x in D_over_time[i][0] or x == D_over_time[i][0].left:
            return i
    return None


def plot_utilization(D_over_time_series, server_size, algorithm):
    points = 1000  # Number of points
    for i in range(len(D_over_time_series)):
        xmin, xmax = 0, D_over_time_series[i][-2][0].right
        xlist = [float(xmax - xmin) * i / points for i in range(points + 1)]
        ylist = list(map(lambda y: server_size - D_over_time_series[i][find_interval(D_over_time_series[i], y)][1],
                         xlist))
        plt.plot(xlist, ylist, label=f"server_{i+1}")
    plt.title(algorithm)
    plt.ylabel('memory_use')
    plt.xlabel('time')
    plt.axhline(y=server_size, color='r')
    plt.legend()
    plt.show()


def calc_utilization(D_over_time, server_size):
    return round(np.array([step[0].length * (server_size - step[1]) / server_size
                           for step in D_over_time[:-2]]).sum() / D_over_time[-2][0].right, 3)


def make_summary(results: np.array, experiment_name, algorithm, opt=False):
    try:
        f = open(f'results/{experiment_name}', 'a')
    except FileNotFoundError:
        f = open(f'results/{experiment_name}', 'w')
    if opt:
        print('Upper bound', file=f)
    else:
        print(algorithm, file=f)
    print("-" * 20, file=f)
    print(f'Mean: {results.mean()}', file=f)
    print(f'Recurrences: {len(results)}', file=f)
    print(f'STD: {results.std()}', file=f)
    print(f'Worst case: {results.max()}', file=f)
    print("-" * 20, file=f)


def run_experiments(algorithm, data_list, experiment_name, opt=False):
    results = np.array([simulate_scheduling(data_tuple=deepcopy(data_list[i]),
                                            algorithm=algorithm, prints=False, opt=opt) for i in range(len(data_list))])
    if opt:
        upper_bound_results = np.array([4 * results[i] + (data_list[i][0]['w'] * data_list[i][0]['p']).sum() for i in range(len(results))])
        make_summary(upper_bound_results, algorithm=algorithm, experiment_name=experiment_name, opt=opt)
        return upper_bound_results.mean()
    else:
        # make_summary(results, algorithm=algorithm, experiment_name=experiment_name, opt=opt)
        return results.mean()

def main():
    recurrences = 1
    experiment_name = 'test'
    while experiment_name + '.json' in os.listdir('expirements_results/json_results'):
        if experiment_name[-1].isdigit():
            experiment_name = experiment_name[:-1] + str(int(experiment_name[-1]) + 1)
        else:
            experiment_name += '_2'
    all_results = {}
    data = {}
    # numbers_of_jobs = list(range(5000, 30000, 5000))
    numbers_of_machines = list(range(20, 140, 20))
    for number_of_machine in numbers_of_machines:
        data[str(number_of_machine)] = [make_data(real_data=False,N=10000, M=number_of_machine) for i in range(recurrences)]
    for alg in ['RANDOM', 'WSPT', 'WSVF_batch_dispatch', 'SVF', 'SPT']:
        results = list()
        for number_of_machine in numbers_of_machines:
            data_list = data[str(number_of_machine)]
            results.append(run_experiments(algorithm=alg, data_list=data_list, experiment_name=experiment_name))
        all_results[alg] = results

    for (alg, results), marker in zip(all_results.items(),['.','v','x','^','<','>']):
        plt.plot(numbers_of_machines, results, marker=marker, label=alg)
    plt.ylabel('Average weighted completion time')
    plt.xlabel('Number of jobs')
    plt.legend()
    plt.savefig(f'tests_figures/{experiment_name}.png')
    with open(file=f'expirements_results/json_results/{experiment_name}.json', mode='w') as f:
        json.dump(fp=f, obj=all_results)


if __name__ == '__main__':
    main()