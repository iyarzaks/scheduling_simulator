import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

from constants import RUN_IN, STARTED_AT


def make_data(real_data=False):
    if real_data:
        jobs_df = pd.read_csv('jobs_data.csv', skiprows=lambda i: i > 1 and random.random() > 0.01)
        jobs_df['job_name'] = [f'job_{i+1}' for i in jobs_df.index]
    else:
        jobs = [{'job_name': f'job_{i}', 'p': random.randint(1, 20),
                 'd': random.randint(1, 100)} for i in range(1, 1001)]
        jobs_df = pd.DataFrame(jobs)
    jobs_df['w'] = [random.randint(1, 20) for i in range(jobs_df.shape[0])]
    jobs_df[RUN_IN] = ''
    jobs_df[STARTED_AT] = ''
    jobs_df['finished_at'] = ''
    jobs_df = shuffle(jobs_df)
    servers = [{'server_name': f'server_{i}', 'D': [(pd.Interval(left=0, right=np.inf), 512)]} for i in range(1, 11)]
    servers_df = pd.DataFrame(servers)
    return jobs_df, servers_df


def build_I_star(jobs_df, machines_size):
    jobs_df['p'] = (jobs_df['p'] * jobs_df['d']) / machines_size
    jobs_df['d'] = machines_size
    return jobs_df


def check_avail_start_time(server, demand, time):
    first_start_time = 0
    remaining_time = time
    for section in server['D']:
        times_interval, capacity = section
        if capacity >= demand:
            remaining_time = remaining_time - times_interval.left
            if time <= 0:
                break
        else:
            first_start_time = times_interval.right
            remaining_time = time
    return first_start_time


def find_first_fit(servers_df, job, specific_server=None):
    servers_df['optional_start_time'] = servers_df.apply(check_avail_start_time, axis=1, demand=job['d'],
                                                         time=job['p'])
    if specific_server is not None:
        return servers_df[servers_df['server_name'] == specific_server].iloc[0]
    return servers_df.sort_values(by='optional_start_time').iloc[0]


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
        elif start_time < interval.left < finish_time < interval.right:
            new_sections.extend([(pd.Interval(interval.left, finish_time), capacity - demand),
                                 (pd.Interval(finish_time, interval.right), capacity)])
        elif start_time < interval.left < interval.right <= finish_time:
            new_sections.extend([(pd.Interval(interval.left, interval.right), capacity - demand)])
        elif finish_time <= interval.left:
            new_sections.append((interval, capacity))
    if new_sections[-1][0].right != np.inf:
        print(new_sections)
    servers_df.at[index_to_change, 'D'] = new_sections
    return servers_df


def simulate_scheduling(jobs_df, servers_df, algorithm='RANDOM', opt=False, prints=True):
    jobs_df['v'] = jobs_df['p'] * jobs_df['d']
    jobs_df['v/w'] = jobs_df['v'] / jobs_df['w']
    jobs_df['p/w'] = jobs_df['p'] / jobs_df['w']
    if algorithm == 'TWO_STEP_SVF':
        I_star_df = build_I_star(jobs_df=jobs_df.copy(), machines_size=servers_df.iloc[0]['D'][0][1])
        simulate_scheduling(jobs_df=I_star_df, servers_df=servers_df.copy(), algorithm='SVF', prints=False)
        jobs_df[RUN_IN] = I_star_df[RUN_IN]
    if algorithm == 'TWO_STEP_WSVF':
        I_star_df = build_I_star(jobs_df=jobs_df.copy(), machines_size=servers_df.iloc[0]['D'][0][1])
        simulate_scheduling(jobs_df=I_star_df, servers_df=servers_df.copy(), algorithm='WSVF', prints=False)
        jobs_df[RUN_IN] = I_star_df[RUN_IN]
    i = 0
    with tqdm(total=jobs_df.shape[0]) as pbar:
        while '' in list(jobs_df[STARTED_AT]):
            unscheduled_jobs = jobs_df[jobs_df[STARTED_AT] == '']['job_name']
            if algorithm == 'RANDOM':
                first_in_line = unscheduled_jobs.iloc[0]
            elif algorithm == 'SPT':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='p').iloc[0]['job_name']
            elif algorithm == 'SVF':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='v').iloc[0]['job_name']
            elif algorithm == 'WSPT':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='p/w').iloc[0]['job_name']
            elif algorithm == 'WSVF':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='v/w').iloc[0]['job_name']
            elif algorithm == 'TWO_STEP_SVF':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='v').iloc[0]['job_name']
                alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0],
                                              specific_server=
                                              jobs_df[jobs_df['job_name'] == first_in_line].iloc[0][RUN_IN])
            elif algorithm == 'TWO_STEP_WSVF':
                first_in_line = jobs_df[jobs_df[STARTED_AT] == ''].sort_values(by='v/w').iloc[0]['job_name']
                alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0],
                                              specific_server=
                                              jobs_df[jobs_df['job_name'] == first_in_line].iloc[0][RUN_IN])

            if algorithm not in ['TWO_STEP_SVF', 'TWO_STEP_WSVF']:
                alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0])


            # updating

            jobs_df.loc[jobs_df['job_name'] == first_in_line, RUN_IN] = alloc_server['server_name']
            jobs_df.loc[jobs_df['job_name'] == first_in_line, STARTED_AT] = alloc_server['optional_start_time']
            finish_time = alloc_server['optional_start_time'] + \
                          jobs_df[jobs_df['job_name'] == first_in_line].iloc[0]['p']
            jobs_df.loc[jobs_df['job_name'] == first_in_line, 'finished_at'] = finish_time

            servers_df = update_servers(server=alloc_server['server_name'], start_time=alloc_server['optional_start_time'],
                                        finish_time=finish_time,
                                        demand=jobs_df[jobs_df['job_name'] == first_in_line].iloc[0]['d'],
                                        servers_df=servers_df)
            i += 1
            if i % 100 == 0:
                pbar.update(n=100)
        pbar.update(n=(i % 100))
    if prints:
        if opt:
            print(f"\nOPT* {algorithm} RESULTS")
        else:
            print(f"\n{algorithm} RESULTS")
        print("-"*20)
        print(f"Average waiting time: {(jobs_df['finished_at']-jobs_df['p']).mean()}\n")
        print(f"Average weighted waiting time: {(jobs_df['w']*(jobs_df['finished_at'] - jobs_df['p'])).mean()}\n")
        print(f"Average finish time: {jobs_df['finished_at'].mean()}\n")
        print(f"Average weighted finish time: {(jobs_df['w']*jobs_df['finished_at']).mean()}\n")



def main():
    jobs_df, servers_df = make_data(real_data=True)
    # I_star_df = build_I_star(jobs_df=jobs_df.copy(), machines_size=servers_df.iloc[0]['D'][0][1])
    # simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='RANDOM')
    # simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='SPT')
    # simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='SVF')
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='WSPT')
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='WSVF')
    # simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='TWO_STEP_SVF')
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='TWO_STEP_WSVF')
    # simulate_scheduling(I_star_df.copy(), servers_df.copy(), algorithm='SPT', opt=True)
    # simulate_scheduling(I_star_df.copy(), servers_df.copy(), algorithm='WSPT', opt=True)
    # print(f"Average p: {(jobs_df['p']).mean()}\n")


if __name__ == '__main__':
    main()