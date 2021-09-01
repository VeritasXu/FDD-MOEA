import time
import copy
import scipy.io
import numpy as np
import pandas as pd
from libs.AF import Multi_AF
from datetime import datetime
from libs.Client import Client
from libs.Server import Server
from config import args_parser
from sklearn.cluster import KMeans
from pymoo.optimize import minimize
from pymoo.factory import get_problem
from libs.Ray_Client import RayClient
from pymoo.algorithms.nsga2 import NSGA2
from utils.initialize import initialize_pop
from utils.visual import visual_2d, visual_3d
from utils.data_utils import non_dominated, dedupe
from pymoo.factory import get_reference_directions
from pymoo.factory import get_performance_indicator
from pymoo.factory import get_sampling, get_crossover, get_mutation



import ray
ray.init(include_dashboard=False)

# -------------Experimental mode---------------------------#
# --------Modify hyper-parameters in config----------------#
args = args_parser()

# object number
M = args.M
d = args.d

n_ref = {'3': 105, '5': 126, '10': 230, '20': 420}
ref_dirs = get_reference_directions("energy", M, n_ref[str(M)], seed=1)

if 'dtlz' in args.func:
    lb = np.zeros(d)
    ub = np.ones(d)
    Function = get_problem(args.func, n_var=d, n_obj=M)

    if '5' in args.func or '6' in args.func or '7' in args.func:
        pf = Function.pareto_front()
        pass
    else:
        pf = Function.pareto_front(ref_dirs)
        pass

elif 'wfg' in args.func:
    lb = np.zeros(d)
    ub = 2 * np.arange(1, d + 1)
    k_dic = {'10': 6, '20': 10, '40': 10, '80': 40}
    Function = get_problem(args.func, n_var=d, n_obj=M, k=k_dic[str(d)])

    if '1' in args.func or '2' in args.func or '3' in args.func:
        pf = Function.pareto_front()
    else:
        pf = Function.pareto_front(ref_dirs)


else:
    d, lb, ub, Function, pf = None, None, None, None, None
    print('Error in Test Function')


#total number of clients
N = args.num_users
num_T = int(args.frac * N)
num_b4_d = 11

num_data = num_b4_d * d

kernel_size = int(np.sqrt(M + d)) + 3
# kernel_size = 2 * d + 1

best_pop = []

max_samp = num_b4_d * d - 1 + 25
max_FE = num_b4_d * d - 1 + 120
num_chosen = 5
gens = args.max_gens

Max_IR = args.runs

#record pop and fit
record_pop = {}
record_fit = {}

#record IGD
IGD_records = np.zeros((25, Max_IR))
#run time
run_time = np.zeros(Max_IR)

#train time
train_time = np.zeros(Max_IR)
#optimization time
opt_time = np.zeros(Max_IR)

A = None

if __name__ == '__main__':


    params = {'E': args.E, 'kernel_size': kernel_size, 'lr': args.lr,
              'd_out': M, 'opt': args.opt}

    ray_clients = [RayClient.remote(params, id_num=i) for i in range(num_T)]
    clients = [Client(params, id_num=i) for i in range(num_T)]


    t0 = time.time()

    for IR in range(Max_IR):
        # re-LHS training data
        print('\033[1;35;46m Re-LHS ' + args.func + ' d=' + str(d) + ' data, ' + str(IR + 1) + ' run\033[0m')

        # reset random seed
        now = datetime.now()
        clock = 100 * (now.year + now.month + now.day + now.hour + now.minute + now.second)
        np.random.seed(clock)

        # generate initial data set
        init_x = initialize_pop(num_b4_d * d-1, d, lb, ub)
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
        iter_igd = []

        # initial IGD
        igd_indicator = get_performance_indicator("igd", pf)

        igd = igd_indicator.calc(T_arch)
        iter_igd.append(igd)
        print("Initial IGD %.5f" % igd)


        idx_users = np.random.choice(range(N), num_T, replace=False)
        search_seed = args.seed

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
                    D_x[i], T_f[i] = non_dominated(D_x[i], T_f[i], rand_half=True, num_select=max_samp)


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

            algorithm = NSGA2(pop_size=50,
                              sampling=get_sampling("real_random"),
                              crossover=get_crossover("real_sbx", prob=0.9, eta=20),
                              mutation=get_mutation("real_pm", eta=20),
                              eliminate_duplicates=True,
                              )

            multi_LCB = Multi_AF(server=server, clients=clients,
                                    ac_type='G', n_var=d, n_obj=M, xl=lb, xu=ub)

            # I dont know why seed of pymoo influence performance so significantly
            while len(chosen_pop) < num_chosen:
                res = minimize(multi_LCB,
                               algorithm,
                               ('n_gen', gens),
                               seed=search_seed,
                               save_history=False,
                               verbose=False)

                tmp_pop = dedupe(res.X, D_arch)

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
                if p >= args.pf:
                    if D_bo[idx].shape[1] != d:
                        D_bo[idx] = chosen_pop
                        T_bo[idx] = chosen_fit

                    else:
                        D_bo[idx] = np.vstack((D_bo[idx], chosen_pop))
                        T_bo[idx] = np.vstack((T_bo[idx], chosen_fit))



            Real_FE += len(chosen_pop)

            A = non_dominated(None, T_arch)
            igd = igd_indicator.calc(T_arch)
            iter_igd.append(igd)
            print("IGD %.5f" % igd)

        IGD_records[:, IR] = np.array(iter_igd)[0:25]
        train_time[IR] = run_train_t
        opt_time[IR] = run_opt_t

        record_pop[args.func+'_run_' + str(IR+1)] = D_arch
        record_fit[args.func+'_run_' + str(IR+1)] = T_arch

        run_end = time.time()
        run_time[IR] = run_end - run_start
        # visual_3d(A, pf)

    t_final = time.time()
    igd_mean, igd_std = np.mean(IGD_records, axis=1), np.std(IGD_records, axis=1)

    # visual_2d([i for i in range(25)], igd_mean)

    print(args.func, '\nM=', M, ', d=', d)
    print('IGD Mean ± std: %.5f' % igd_mean[-1], '$\pm$ %.5f, ' % igd_std[-1])
    print('20 runs elapsed time: %.2f' % (t_final - t0), 's\n')

    file_name = './results/NSGA/' + args.func + '_M' + str(M) + '_d' + str(d)
    scipy.io.savemat(file_name + '_pop.mat', record_pop)
    scipy.io.savemat(file_name + '_fit.mat', record_fit)

    run_time = pd.DataFrame(run_time)
    run_time.to_csv(file_name + '_runtime.csv', index=False, header=False)

    train_time = pd.DataFrame(train_time)
    train_time.to_csv(file_name + '_train.csv', index=False, header=False)

    opt_time = pd.DataFrame(opt_time)
    opt_time.to_csv(file_name + '_opt.csv', index=False, header=False)

    mean_profiles = pd.DataFrame(igd_mean)
    mean_profiles.to_csv(file_name + '_IGD.csv', index=False, header=False)

    std_profiles = pd.DataFrame(igd_std)
    std_profiles.to_csv(file_name + '_IGD_std.csv', index=False, header=False)
