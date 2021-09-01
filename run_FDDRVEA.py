import time
import copy
import scipy.io
import numpy as np
from MOEA.RVEA import RVEA
from libs.AF import Multi_AF
from datetime import datetime
from libs.Client import Client
from libs.Server import Server
from config import args_parser
from sklearn.cluster import KMeans
from pymoo.factory import get_problem
from libs.Ray_Client import RayClient
from utils.initialize import initialize_pop
from utils.data_utils import non_dominated, dedupe
from pymoo.factory import get_reference_directions

import ray

num_chosen = 5
num_b4_d = 11

args = args_parser()

# -------------Experimental mode---------------------------#
# -------------Modify hyper-parameters here----------------#

problems = ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
Ms = [3, 5, 10, 20]
ds = [30]

xlb, xub = 0, 1

file_path = './results/NSGA/'

Max_IR = args.runs

N = args.num_users

frac = args.frac

gens = args.max_gens

E = args.E

lr = args.lr

pf = args.pf

opt = args.opt

seed = args.seed

num_T = int(frac * N)

if __name__ == '__main__':

    for prob in problems:
        for M in Ms:
            for d in ds:
                n_ref = {'3': 105, '5': 126, '10': 230, '20': 420}
                ref_dirs = get_reference_directions("energy", M, n_ref[str(M)], seed=1)

                kernel_size = int(np.sqrt(M + d)) + 3

                max_samp = 11 * d - 1 + 25
                max_FE = 11 * d - 1 + 120

                # record pop and fit
                record_pop = {}
                record_fit = {}

                # run time
                run_time = np.zeros(Max_IR)
                # train time
                train_time = np.zeros(Max_IR)
                # optimization time
                opt_time = np.zeros(Max_IR)

                lb = xlb * np.ones(d)
                ub = xub * np.ones(d)

                Function = get_problem(prob, n_var=d, n_obj=M)

                params = {'E': E, 'kernel_size': kernel_size, 'lr': lr,
                          'd_out': M, 'opt': opt}

                ray.init(include_dashboard=False)
                ray_clients = [RayClient.remote(params, id_num=i) for i in range(num_T)]
                clients = [Client(params, id_num=i) for i in range(num_T)]

                t0 = time.time()

                for IR in range(Max_IR):
                    # re-LHS training data
                    print('\033[1;35;46m Re-LHS ' + prob + ' d=' + str(d) + ' data, ' + str(IR + 1) + ' run\033[0m')

                    # reset random seed
                    now = datetime.now()
                    clock = 100 * (now.year + now.month + now.day + now.hour + now.minute + now.second)
                    np.random.seed(clock)

                    # generate initial data set
                    init_x = initialize_pop(num_b4_d * d - 1, d, lb, ub)
                    init_f = Function.evaluate(init_x)

                    # initialize dataset 11d - 1
                    D_init, T_init = [init_x for i in range(N)], [init_f for i in range(N)]
                    # data selected by Bayesian optimization
                    D_bo, T_bo = [np.ones((1, 2)) for i in range(N)], [np.ones((1, 1)) for i in range(N)]
                    # create a dataset that store all found solutions
                    D_arch, T_arch = copy.deepcopy(D_init[0]), copy.deepcopy(T_init[0])

                    # count real fitness evaluations
                    Real_FE = num_b4_d * d - 1

                    server = Server(params)

                    init_front = copy.deepcopy(init_f)

                    res = None

                    idx_users = np.random.choice(range(N), num_T, replace=False)
                    search_seed = seed

                    run_start = time.time()
                    run_train_t = 0
                    run_opt_t = 0

                    # start from 11d, end at 11d
                    while Real_FE < max_FE:

                        print('Real FE ', Real_FE)

                        # define a tmp client dataset, using for stack Dl and D
                        D_x, T_f = [[] for i in range(num_T)], [[] for i in range(num_T)]

                        # reset chosen_pop
                        chosen_pop = []

                        t1 = time.time()

                        # 1: get the model of the server
                        server_model = server.broadcast()

                        for i, idx in enumerate(idx_users):
                            # real-evaluate the chosen sample, count the number
                            if D_bo[idx].shape[1] == d:
                                D_x[i] = np.vstack((D_init[idx], D_bo[idx]))
                                T_f[i] = np.vstack((T_init[idx], T_bo[idx]))
                            else:
                                D_x[i] = D_init[idx]
                                T_f[i] = T_init[idx]

                            if len(D_x[i]) > max_samp:
                                # first 1/2 chosen by non-dominated, randomly 1/2 others
                                D_x[i], T_f[i] = non_dominated(D_x[i], T_f[i], rand_half=False, num_select=max_samp)

                            # 2: overwrite the local model
                            ray_clients[i].synchronize.remote(server_model)
                            # 3: local training
                            ray_clients[i].train.remote(D_x[i], T_f[i])

                        all_models = ray.get([ray_clients[i].compute_update.remote() for i in range(num_T)])
                        [clients[i].reload(all_models[i]) for i in range(num_T)]

                        # local models ---> averaging ---> global model
                        server.average(clients)
                        t2 = time.time()
                        run_train_t += t2 - t1
                        print('update, train time: %.2f' % (t2 - t1))

                        multi_LCB = Multi_AF(server=server, clients=clients,
                                             ac_type='G', n_var=d, n_obj=M, xl=lb, xu=ub)

                        while len(chosen_pop) < num_chosen:
                            res_X, res_F = RVEA(multi_LCB, lb, ub, M, ref_dirs, n_pop=n_ref[str(M)], n_gen=gens,
                                                coding='gaussian')

                            tmp_pop = dedupe(res_X, D_arch)

                            if len(tmp_pop) >= num_chosen:
                                # best situation, choose num_chosen distinct pops with k-means
                                k_means = KMeans(n_clusters=num_chosen).fit(tmp_pop)
                                chosen_pop = k_means.cluster_centers_

                            elif 0 < len(tmp_pop) < num_chosen:
                                # if not, adding distinct pops incrementally to save time
                                search_seed += 100
                                if len(chosen_pop) == 0:
                                    chosen_pop = tmp_pop
                                else:
                                    chosen_pop = np.vstack((chosen_pop, tmp_pop))
                                    chosen_pop = dedupe(chosen_pop, dedupe_self=False)

                                    if len(chosen_pop) >= num_chosen:
                                        k_means = KMeans(n_clusters=num_chosen).fit(chosen_pop)
                                        chosen_pop = k_means.cluster_centers_

                        search_seed += 100

                        t3 = time.time()
                        run_opt_t += t3 - t2
                        print('optimization time: %.2f' % (t3 - t2))

                        chosen_fit = Function.evaluate(chosen_pop).reshape(-1, M)

                        D_arch = np.vstack((D_arch, chosen_pop))
                        T_arch = np.vstack((T_arch, chosen_fit))

                        idx_users = np.random.choice(range(N), num_T, replace=False)

                        for idx in idx_users:
                            p = np.random.rand()
                            if p >= pf:
                                if D_bo[idx].shape[1] != d:
                                    D_bo[idx] = chosen_pop
                                    T_bo[idx] = chosen_fit

                                else:
                                    D_bo[idx] = np.vstack((D_bo[idx], chosen_pop))
                                    T_bo[idx] = np.vstack((T_bo[idx], chosen_fit))

                        Real_FE += len(chosen_pop)

                    train_time[IR] = run_train_t
                    opt_time[IR] = run_opt_t

                    record_pop[prob + '_run_' + str(IR + 1)] = D_arch
                    record_fit[prob + '_run_' + str(IR + 1)] = T_arch

                    run_end = time.time()
                    run_time[IR] = run_end - run_start

                t_final = time.time()

                print(prob, '\nM=', M, ', d=', d)
                print('20 runs elapsed time: %.2f' % (t_final - t0), 's\n')

                file_name = file_path + prob + '_M' + str(M) + '_d' + str(d)
                scipy.io.savemat(file_name + '_pop.mat', record_pop)
                scipy.io.savemat(file_name + '_fit.mat', record_fit)

                ray.shutdown()
