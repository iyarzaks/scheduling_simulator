import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle



def make_data(real_data=False):
    if real_data:
        jobs_df = pd.read_csv('jobs_data.csv', skiprows=lambda i: i > 1 and random.random() > 0.01)
        jobs_df['job_name'] = [f'job_{i+1}' for i in jobs_df.index]
    else:
        jobs = [{'job_name': f'job_{i}', 'p': random.randint(1, 20),
                 'd': random.randint(1, 100)} for i in range(1, 1001)]
        jobs_df = pd.DataFrame(jobs)
    jobs_df['run_in'] = ''
    jobs_df['started_at'] = ''
    jobs_df['finished_at'] = ''
    jobs_df = shuffle(jobs_df)
    servers = [{'server_name': f'server_{i}', 'D': [(pd.Interval(left=0, right=np.inf), 512)]} for i in range(1, 6)]
    servers_df = pd.DataFrame(servers)
    return jobs_df, servers_df


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


def find_first_fit(servers_df, job):
    servers_df['optional_start_time'] = servers_df.apply(check_avail_start_time, axis=1, demand = job['d'],
                                                         time = job['p'])
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
        print (new_sections)
    servers_df.at[index_to_change, 'D'] = new_sections
    return servers_df


def simulate_scheduling(jobs_df, servers_df, algorithm='random'):
    jobs_df['v'] = jobs_df['p'] * jobs_df['d']
    i = 0
    with tqdm(total=jobs_df.shape[0]) as pbar:
        while '' in list(jobs_df['run_in']):
            unscheduled_jobs = jobs_df[jobs_df['run_in'] == '']['job_name']
            if algorithm == 'random':
                first_in_line = unscheduled_jobs.iloc[0]
            elif algorithm == 'spt':
                first_in_line = jobs_df[jobs_df['run_in'] == ''].sort_values(by='p').iloc[0]['job_name']
            elif algorithm == 'svf':
                first_in_line = jobs_df[jobs_df['run_in'] == ''].sort_values(by='v').iloc[0]['job_name']
            alloc_server = find_first_fit(servers_df, jobs_df[jobs_df['job_name'] == first_in_line].iloc[0])

            # updating

            jobs_df.loc[jobs_df['job_name'] == first_in_line, 'run_in'] = alloc_server['server_name']
            jobs_df.loc[jobs_df['job_name'] == first_in_line, 'started_at'] = alloc_server['optional_start_time']
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
    print (f"\n{algorithm} Average waiting time: {(jobs_df['finished_at']-jobs_df['p']).mean()}\n")
    print(f"\n{algorithm} Average finish time: {jobs_df['finished_at'].mean()}\n")



def main():
    jobs_df, servers_df = make_data(real_data=True)
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='random')
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='spt')
    simulate_scheduling(jobs_df.copy(), servers_df.copy(), algorithm='svf')

if __name__ == '__main__':
    main()