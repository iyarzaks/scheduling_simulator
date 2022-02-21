from mip import *
import pandas as pd
from simulator import simulate_scheduling
import ast
import pickle
from integer_problem_creator import create_problem, create_integer_problem
import time
import datetime
import os
from humanfriendly import format_timespan


def solve_to_optimize(problem_name, f, m=None, write=True):
    if m is None:
        m = Model(sense=MINIMIZE, solver_name=GRB)
        m.read(f'{problem_name}/model.lp')
    print('model has {} vars, {} constraints and {} nzs'.format(m.num_cols, m.num_rows, m.num_nz))
    m.verbose = 0
    m.threads = -1
    status = m.optimize()
    if status == OptimizationStatus.OPTIMAL:
        print('optimal solution cost {} found'.format(m.objective_value))
    elif status == OptimizationStatus.FEASIBLE:
        print('sol.cost {} found, best possible: {}'.format(m.objective_value, m.objective_bound))
    elif status == OptimizationStatus.NO_SOLUTION_FOUND:
        print('no feasible solution found, lower bound is: {}'.format(m.objective_bound))
    if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
        if write:
            print(f'solver objective_value: {m.objective_value}', file=f)
            print(f'IS_OPTIMAL: {m.status == OptimizationStatus.OPTIMAL}', file=f)
            print('solver solution:', file=f)
            already_started = []
            for v in m.vars:
                if abs(v.x) > 1e-6: # only printing non-zeros
                    if 's' in v.name:
                        j, i, t = v.name.split('_')[1].split(',')
                        if j not in already_started:
                            # print(f"Start job {j} in machine {i} at time {t}", file=f)
                            already_started.append(j)
                        # print('{} : {}'.format(v.name, v.x))
        return m.objective_value


def solve_with_alg(problem_name, f, write=None, jobs_df=None, servers_df=None, randomize_server_fit=False,
                   algorithm = 'WSVF_batch_dispatch'):
    if jobs_df is None:
        jobs_df = pd.read_csv(f'{problem_name}/jobs.csv').fillna('')
    if servers_df is None:
        with open(f'{problem_name}/servers.pkl', 'rb') as pkl_f:
            servers_df = pickle.load(pkl_f)
    result = simulate_scheduling(jobs_df=jobs_df, servers_df=servers_df, algorithm=algorithm,
                                 randomize_server_fit=randomize_server_fit)
    if write:
        print(f"{algorithm} result: {jobs_df.shape[0] * result}", file=f)
    print(f"{algorithm} result: {jobs_df.shape[0] * result}")
    return result


def solve(problem_name='', new=False, write=True, m=None, jobs_df=None, servers_df=None):
    # if write:
    #     with open(f'{problem_name}/solution{"" if new else "."}.txt', 'w') as f:
    #         wsvf_res = solve_with_alg(problem_name=problem_name, f=f, write=write, jobs_df=jobs_df, servers_df=servers_df,
    #                                   algorithm='WSVF_batch_dispatch')
            # ts = int(time.time())
            # opt_res = solve_to_optimize(problem_name=problem_name, f=f, m=m, write=write)
            # te = int(time.time())
            # print(f'solver works for: {datetime.timedelta(seconds=te-ts)}', file=f)
            # wsvf_res_rand = solve_with_wsvf(problem_name=problem_name, f=f, write=write, jobs_df=jobs_df,
            #                            servers_df=servers_df, randomize_server_fit=True)
    # else:
    wspt_result = solve_with_alg(problem_name=problem_name, f=None, write=write, jobs_df=jobs_df.copy(),
                              servers_df=servers_df.copy(), algorithm='WSPT')
    svf_result = solve_with_alg(problem_name=problem_name, f=None, write=write, jobs_df=jobs_df.copy(),
                                 servers_df=servers_df.copy(), algorithm='SVF')
    spt_result = solve_with_alg(problem_name=problem_name, f=None, write=write, jobs_df=jobs_df.copy(),
                                 servers_df=servers_df.copy(), algorithm='SPT')
    wsvf_dinamic = solve_with_alg(problem_name=problem_name, f=None, write=write, jobs_df=jobs_df.copy(),
                                servers_df=servers_df.copy(), algorithm='WSVF')
    # wsvf_dinamic_w_bypass = solve_with_alg(problem_name=problem_name, f=None, write=write, jobs_df=jobs_df.copy(),
    #                               servers_df=servers_df.copy(), algorithm='WSVF_DINAMIC_with_bypassing')
    # opt_res = solve_to_optimize(problem_name=problem_name, f=None, m=m, write=True)
    opt_res = None
    return wspt_result, svf_result, spt_result, opt_res, wsvf_dinamic



def main():
    # counter = 25
    # new = True
    # success = False
    # problem_root_dir = f'solver_tests/new_test_{counter}'
    # if new:
    #     while success is False:
    #         try:
    #             success = os.makedirs(problem_root_dir)
    #         except:
    #             print(f'{problem_root_dir} is exist')
    #             counter += 1
    #             problem_root_dir = f'solver_tests/new_test_{counter}'
    #     create_problem(problem_name=problem_root_dir, number_of_machines=5, number_of_jobs=40)
    # create_integer_problem(problem_name=problem_root_dir)
    # solve(problem_root_dir, new)
    # df = pd.read_csv('wsvf_opt_compare.csv',index_col=0)
    const_number_of_jobs = 2000
    const_number_of_machines = 7
    # machines_numbers = range(1, 7)
    jobs_numbers = range(1000, 5000, 1000)
    alphas = np.array(range(50, 55, 5))
    alphas = alphas / 100
    prediction_stds = range(0, 6)
    results_list = []
    # for M in machines_numbers:
    #     servers_df, problem_df = create_problem(number_of_machines=M, number_of_jobs=const_number_of_jobs,
    #                                             write=False, problem_name='')
    #     m = create_integer_problem(servers_df=servers_df, jobs_df=problem_df, write=False, problem_name='')
    #     wsvf_res, opt_res = solve(problem_name='', new=False, write=False, m=m, jobs_df=problem_df,
    #                               servers_df=servers_df)
    #     sigma_p = sum(problem_df['w'] * problem_df['p'])
    #     results_list.append({'WSVF_batch_dispatch': wsvf_res, 'OPT': opt_res, 'M': M, 'N': const_number_of_jobs,
    #                          'SIGMA_W_P': sigma_p})
    for a in alphas:
        all_experiment_results = defaultdict(list)

        for prediction_std in prediction_stds:
            for experiment in range(1):
                servers_df, problem_df = create_problem(number_of_machines=const_number_of_machines,
                                                        number_of_jobs=const_number_of_jobs,
                                                        write=False, problem_name='', alpha=a,
                                                        prediction_std=prediction_std)
                # m = create_integer_problem(servers_df=servers_df, jobs_df=problem_df, write=True, problem_name='')
                wspt_result, svf_result, spt_result, opt_res, wsvf_dinamic = solve(problem_name='', new=False, write=False, jobs_df=problem_df,
                                          servers_df=servers_df)
                sigma_p = sum(problem_df['w'] * problem_df['p'])
                all_experiment_results['WSVF'].append(wsvf_dinamic)
                all_experiment_results['SVF'].append(svf_result)
                all_experiment_results['WSPT'].append(wspt_result)
                all_experiment_results['SPT'].append(spt_result)
                if opt_res is None:
                    all_experiment_results['OPT'].append(sigma_p)
                else:
                    all_experiment_results['OPT'].append(opt_res)
                all_experiment_results['SIGMA_W_P'].append(sigma_p)
                results_list.append({'WSVF': np.mean(all_experiment_results['WSVF']),
                                     # 'OPT': np.mean(all_experiment_results['OPT']),
                                     'SVF': np.mean(all_experiment_results['SVF']),
                                     'WSPT': np.mean(all_experiment_results['WSPT']),
                                     'SPT': np.mean(all_experiment_results['SPT']),
                                     'M': const_number_of_machines, 'N': const_number_of_jobs})

        # results_list.append({'WSVF_batch_dispatch': np.mean(all_experiment_results['WSVF_batch_dispatch']),
        #                      # 'OPT': np.mean(all_experiment_results['OPT']),
        #                      'SVF': np.mean(all_experiment_results['SVF']),
        #                      'SPT': np.mean(all_experiment_results['SPT']),
        #                      'WSPT': np.mean(all_experiment_results['WSPT']),
        #                      'WSVF_dynamic_no_bypassing': np.mean(all_experiment_results['SPT']),
        #                      'WSVF_dynamic_with_bypassing': np.mean(all_experiment_results['WSPT']),
        #                      'M': const_number_of_machines, 'N': const_number_of_jobs})
    df = pd.DataFrame(results_list)
    # df.loc[df['OPT'].isna(), 'OPT'] = df.loc[df['OPT'].isna(), 'SIGMA_W_P']
    df.to_csv('online_comparison_stds_normal_2.csv')




if __name__ == '__main__':
    main()
