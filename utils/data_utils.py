import copy
import math
import random
import numpy as np
from pandas.core.common import flatten
from scipy.spatial.distance import cdist
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def index_bootstrap(num_data, prob):
    """
    :param num_data: the index matrix of input, int
    :param prob: the probability for one index sample to be chose, >0
    return: index of chose samples, bool

    example:
    a=np.array([[1,2,3,4],[0,0,0,0]]).T
    rand_p = np.random.rand(4)
    b=np.greater(rand_p,0.5)
    b is the output, and we can use a[b] to locate data
    """
    rand_p = np.random.rand(num_data)

    out = np.greater(rand_p, 1-prob)

    if True not in out:
        out = index_bootstrap(num_data, prob)

    return out

def mini_batches(input_x, input_y, distance, batch_size=64, seed=0):
    """
    return random batch indexes for a list
    """
    np.random.seed(seed)
    num_samp = input_x.shape[0]
    batches = []
    permutation = list(np.random.permutation(num_samp))
    num_batch = math.floor(num_samp / batch_size)
    iter_list = [i for i in range(num_batch)]

    for k in iter_list:
        batch_index = permutation[k * batch_size:(k + 1) * batch_size]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))
    if num_samp % batch_size != 0:
        batch_index = permutation[batch_size * num_batch:]
        batches.append((input_x[batch_index], input_y[batch_index], distance[batch_index]))

    return batches


def sort_data(pop, pop_obj, num_select):
    """
    sort the data (x, y) according to the descending sequence and pick first num_select points
    for single objective problem
    :param pop: [N, d]
    :param pop_obj: [N, 1]
    :param num_select:
    :return:
    """
    data_index = np.argsort(pop_obj.flatten())
    return pop[data_index[0:num_select], :], pop_obj[data_index[0:num_select], :]

def non_dominated(pop, pop_obj, rand_half=False, num_select=None):
    """
    the first num_select/2 training data are the optimal ones chosen by nondominated sorting and crowding distance,
    and the other half are randomly sampled without replacement

    :param pop:
    :param pop_obj:
    :param rand_half: whether select half randomly
    :param num_select:
    :return:
    """
    nds = NonDominatedSorting()
    fronts = nds.do(pop_obj)

    if num_select is not None:
        if rand_half is False:
            data_index = list(flatten(fronts))[0:num_select]
            # print(fronts)
            # print(list(flatten(fronts))[0:5], "# f")
        else:
            index1 = list(flatten(fronts))[0:int(num_select/2)]
            remaining_list = list(flatten(fronts))[int(num_select/2):-1]
            index2 = random.sample(remaining_list, num_select-int(num_select/2))
            data_index = index1 + index2

        return pop[data_index], pop_obj[data_index]
    else:
        data_index = fronts[0]
        return pop_obj[data_index]

def find_duplicates(X, epsilon=1e-16, dedupe_self=True):
    # calculate the distance matrix from each point to another
    D = cdist(X, X)
    if dedupe_self is True:
        # set the diagonal to infinity
        di = np.diag_indices(X.shape[0])
    else:
        # set upper triangular to inf
        di = np.triu_indices(len(X))

    D[di] = np.inf
    # set as duplicate if a point is really close to this one
    is_duplicate = np.any(D < epsilon, axis=1)

    return is_duplicate

def dedupe(res, origin=None, epsilon=1e-6, dedupe_self=True):
    """
    # Remove the duplicate part of res in origin

    :return:
    """
    num_res = res.shape[0]
    if origin is not None:
        stack_res = np.vstack((res, origin))
    else:
        stack_res = copy.deepcopy(res)
    is_unique = np.where(np.logical_not(find_duplicates(stack_res, epsilon=epsilon, dedupe_self=dedupe_self)))[0]
    top_unique = list(is_unique[0:num_res])
    while len(top_unique) != 0 and top_unique[-1] >= num_res :
        top_unique.pop()
    _res = res[top_unique]

    return _res
