import copy
import numpy as np
from functools import partial
from datetime import datetime
from pymoo.interface import mutation
from pymoo.interface import crossover
from pymoo.factory import get_mutation
from pymoo.factory import get_crossover
from scipy.spatial.distance import cdist
from utils.initialize import initialize_pop



np.seterr(divide='ignore', invalid='ignore')


def _obj_wrapper(func, args, kwargs, x):
    out={}
    func._evaluate(x, out, *args, **kwargs)
    return out['F']


def cosine_similarity(dirs, ref_dirs):
    pair_dist = cdist(dirs, ref_dirs, 'cosine')
    cosine = 1 - pair_dist
    return cosine


def uniform_crossover(x1, x2):
    sigma = np.random.rand()
    x1_ = sigma * x1 + (1 - sigma) * x2
    x2_ = (1 - sigma) * x1 + sigma * x2
    return x1_, x2_


def gaussian_mutation(x):
    return x + np.random.randn(len(x))


def adp_selection(X, F, ref_dirs, theta0):
    n_ref, _ = ref_dirs.shape

    z_min = np.min(F, axis=0)
    F_shift = F - z_min
    F_norm = F_shift / (np.linalg.norm(F_shift, axis=1).reshape(-1, 1) + 1e-14)

    angle = np.arccos(cosine_similarity(F_norm, ref_dirs))
    associate = angle.argmin(axis=1)
    associate = associate.T

    cosine = cosine_similarity(ref_dirs, ref_dirs)
    row, col = np.diag_indices_from(cosine)
    cosine[row, col] = 0
    gamma = np.arccos(cosine).min(axis=1)
    gamma = gamma.T

    index = np.unique(associate)
    sub_pops = dict(zip(np.arange(n_ref), [[]] * n_ref))
    selection = []
    for idx in index:
        tmp = list(np.where(associate == idx)[0])
        sub_pops.update({idx: tmp})
        if len(sub_pops[idx]) != 0:
            sub_pop = sub_pops[idx]
            F_sub = F_norm[sub_pop]
            d1 = np.linalg.norm(F_sub, axis=1)
            d = d1 * (1 + theta0 * angle[sub_pop, idx]/gamma[idx])
            min_idx_adp = np.argmin(d)
            selection.append(sub_pop[min_idx_adp])

    return X[selection, :], F[selection]


def RVEA(func, lb, ub, n_obj, ini_refV, n_pop=50, n_gen=50, alpha=2.0, fr=0.1, coding='real', args=(),
         kwargs=None):

    if kwargs is None:
        kwargs = {}

    lb = np.array(lb)
    ub = np.array(ub)
    d = len(lb)
    obj = partial(_obj_wrapper, func, args, kwargs)

    now = datetime.now()
    clock = 100 * (now.year + now.month + now.day + now.hour + now.minute + now.second)
    np.random.seed(clock)

    pop_rand = initialize_pop(n_pop, d, lb, ub)
    pop_fitness = obj(pop_rand)

    Vec = copy.deepcopy(ini_refV)


    c_gens = [i for i in range(n_gen)]
    for c_gen in c_gens:

        if coding == 'real':
            pop_rand = pop_rand[np.random.permutation(pop_rand.shape[0])]
            if pop_rand.shape[0] % 2 != 0:
                pop_rand = np.vstack((pop_rand, pop_rand[0, :]))
            idx1 = [i for i in range(0, pop_rand.shape[0], 2)]
            idx2 = [i + 1 for i in idx1]
            parent1 = pop_rand[idx1]
            parent2 = pop_rand[idx2]
            pop_cross = crossover(get_crossover("real_sbx", prob=0.9, eta=20), parent1, parent2, xl=lb, xu=ub)
            pop_mutation = mutation(get_mutation("real_pm", eta=20, prob=1.0), pop_cross, xl=lb, xu=ub)
            pop_rand = np.vstack((pop_rand, pop_mutation))
            pop_fitness = obj(pop_rand)

        elif coding == 'gaussian':
            X_ = np.full(pop_rand.shape, np.nan)

            # reproduce offspring
            for i in range(pop_rand.shape[0] // 2):
                x1 = pop_rand[np.random.choice(pop_rand.shape[0]), :]
                x2 = pop_rand[np.random.choice(pop_rand.shape[0]), :]
                x1, x2 = uniform_crossover(x1, x2)
                x1, x2 = gaussian_mutation(x1), gaussian_mutation(x2)
                X_[2 * i], X_[2 * i + 1] = np.clip(x1, lb, ub), np.clip(x2, lb, ub)

            if pop_rand.shape[0] % 2 != 0:
                x1 = pop_rand[np.random.choice(pop_rand.shape[0]), :]
                x2 = pop_rand[np.random.choice(pop_rand.shape[0]), :]
                x1, x2 = uniform_crossover(x1, x2)
                x1, x2 = gaussian_mutation(x1), gaussian_mutation(x2)
                X_[-1] = np.clip(x1, lb, ub)

            # evaluate and merge with parents
            F_ = obj(X_)
            pop_rand, pop_fitness = np.vstack([pop_rand, X_]), np.vstack([pop_fitness, F_])

        else:
            print('Please select coding from gaussian and real')

        # reference vector adaption
        if c_gen % (n_gen * fr) == 0:
            z_min = np.min(pop_fitness, axis=0)
            z_max = np.max(pop_fitness, axis=0)
            Vec = ini_refV * (z_max - z_min)
            Vec = Vec / np.linalg.norm(Vec, axis=1).reshape(-1, 1)

        # APD based selection
        theta0 = (c_gen / n_gen) ** alpha * n_obj
        pop_rand, pop_fitness = adp_selection(pop_rand, pop_fitness, Vec, theta0)

    return pop_rand, pop_fitness