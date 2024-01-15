from math import exp
import numpy as np
import torch

np.random.seed(7)


def rbf_kernel(x1, x2, variance = 0.5):
    return exp(-1 * ((x1-x2) ** 2) / (2*variance))


def gram_matrix(xs):
    return [[rbf_kernel(x1,x2) for x2 in xs] for x1 in xs]


def get_data_slice_1(number_trials=110, number_time=90, positive=True):
    length = 10
    xs = np.linspace(0, length, number_time)
    mean = [0 for x in xs]
    gram = gram_matrix(xs)

    xs_list, ys_list = [], []
    for i in range(0, number_trials):
        ys = np.random.multivariate_normal(mean, gram)
        xs_list.append(xs)
        if positive:
            ys_list.append(np.clip(ys,0,10**6))
        else:
            ys_list.append(ys+0.2)

    return torch.tensor(np.stack(ys_list)).float()


def get_data_vector_1(number_neurons=100):

    return torch.rand(number_neurons)


def get_tensor_1(number_trials=110, number_neurons=100, number_time=90, positive=True):
    slice = get_data_slice_1(number_trials=number_trials, number_time=number_time, positive=positive)
    v = get_data_vector_1(number_neurons)
    t = torch.einsum('ij,k->ijk', [slice, v])
    if positive:
        return (t/t.mean()).permute(0,2,1), slice, v#
    else:
        return (t).permute(0, 2, 1), slice, v  #