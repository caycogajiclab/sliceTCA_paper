import numpy as np
from .helper_functions import *
import sklearn

"""
Throughout:
data (ndarray): time x trial x ... x neuron
movement (ndarray): time x ... x trial x XY 
"""


def decoding(x, y, regularization=1):
    """
    Linearly decodes (with bias) y from x (least square)
    in:
        x (ndarray): z_1, z_2, ..., X
        y (ndarray): z_1, z_2, ..., Y
    out:
        decoder_weight (ndarray): dim(Y) x dim(X) decoder weight
        decoder_bias (ndarray): dim(Y) decoder bias
    """

    x_shape = x.shape

    x_with_ones = np.concatenate([x, np.ones((list(x_shape[:-1])+[1]))], axis=-1)

    x_with_ones_flat = x_with_ones.reshape(-1, x_shape[-1]+1)
    y_flat = y.reshape(-1, y.shape[-1])

    if regularization==0:
        W_b = np.linalg.lstsq(x_with_ones_flat, y_flat, rcond=None)[0].T # dim(Y) x dim(X)+1
    else:
        W_b = sklearn.linear_model.ridge_regression(x_with_ones_flat, y_flat, alpha=regularization)

    W, b = W_b[:,:x_shape[-1]], W_b[:,-1]

    return W, b


def angles_from_vectors(x):
    """
    in:
        x (ndarray): ... x XY
    out:
        angle (ndarray): ... angle in 0 to 2*np.pi
    """

    return np.arctan2(x[...,1], x[...,0])+np.pi


def velocity_decoding(data, movement, regularization=1):
    """
    Linearly decodes (with bias) the velocity from the data and integrates it from true starting point
    in:
        data, movement
    out:
        decoded_movement (ndarray): time x trial x ... x XY decoded movement
        decoding_function (python function): a function which takes data as input and returns movement
    """

    velocity = movement[1:]-movement[:-1]
    data_cut = data[:-1]

    W, b = decoding(data_cut, velocity, regularization)

    def decoder(x, initial_pos):

        velocity_hat = x[:-1] @ W.T + b

        position_zero_centered = np.cumsum(velocity_hat, axis=0)
        position_no_init = position_zero_centered + initial_pos[np.newaxis]
        position = np.concatenate([initial_pos[np.newaxis], position_no_init], axis=0)

        return position

    return decoder(data, movement[0]), decoder


def position_decoding(data, movement, regularization=1):
    """
    Linearly decodes (with bias) the position from the data
    in:
        data, movement
    out:
        decoded_movement (ndarray): time x trial x ... x x XY decoded movement
        decoding_function (python function): a function which takes data as input and returns movement
    """

    W, b = decoding(data, movement, regularization)

    def decoder(x, initial_pos): return x @ W.T + b

    return decoder(data, None), decoder


def trial_wise_r2(y, y_hat, mean=True):
    """
    Returns trial-wise R^2
    in:
        y (ndarray): time x trial x neuron
        y_hat (ndarray): time x trial x neuron
    out:
        r2 (float): mean R^2
        r2_std (float): std R^2
    """
    trial_r2 = 1 - np.sum((y - y_hat) ** 2, axis=(0, 2)) / np.sum((y - y.mean(axis=0)[np.newaxis]) ** 2,
                                                                  axis=(0, 2))
    r2 = trial_r2.mean(axis=0) if mean else trial_r2
    r2_std = trial_r2.std(axis=0)
    return r2, r2_std


def k_fold_cross_validated_r2(data, movement, decoding_function, folds=5, sample_size=1, plot=False, mean=True, seed=None):
    """
    Performs k-fold crossvalidated linear decoding of the data and returns test R^2
    in:
        data, movement
        decoding function (python function): a function which takes data and movement as an argument and returns a decoder
        folds (int): the number of splits of the data
        sample_size (int): how many times the whole procedure is done
    out:
        r2_mean (float): mean R^2 over trials, folds, permutations (see trial_wise_r2)
        r2_std (float): std R^2 over trials
    """

    if seed is not None: np.random.seed(seed)

    flattened_data = flatten_mid_dims(data)
    nb_time, nb_trial, nb_neuron = flattened_data.shape

    flattened_movement = flatten_mid_dims(movement)

    size_of_folds_samples = round((nb_trial-nb_trial % folds)/folds)

    all_r2 = []
    all_r2_std = []
    for sample in range(sample_size):
        permutation = np.random.permutation(nb_trial)
        permuted_data, permuted_movement = flattened_data[:,permutation], flattened_movement[:,permutation]
        n = 0
        r2 = []
        r2_std = []
        for fold in range(folds):
            train_data, test_data = split(permuted_data, n, size_of_folds_samples)
            train_movement, test_movement = split(permuted_movement, n, size_of_folds_samples)

            train_movement_hat, decoder = decoding_function(train_data, train_movement)
            test_movement_hat = decoder(test_data, test_movement[0])

            if plot:
                import matplotlib.pyplot as plt
                plt.plot(train_movement_hat[...,0], train_movement_hat[...,1], color='grey', alpha=0.5)
                plt.plot(train_movement[..., 0], train_movement[..., 1], linestyle='--', color='black')
                plt.plot(test_movement_hat[..., 0], test_movement_hat[..., 1], color='red')
                plt.plot(test_movement[..., 0], test_movement[..., 1], linestyle='--', color='red')

                plt.show()

            current_r2, current_r2_std = trial_wise_r2(test_movement, test_movement_hat, mean)

            r2 += [current_r2]
            r2_std += [current_r2_std]

            n += size_of_folds_samples

        all_r2 += [r2]
        all_r2_std += [r2_std]

    all_r2 = np.array(all_r2)
    all_r2_std = np.array(all_r2_std)

    return all_r2.mean() if mean else all_r2, all_r2_std.mean()


def condition_average(data, movement, angle):
    """
    in:
        data, movement
        angle (ndarray): trial a float condition
    out:
        condition_avg_data (ndarray)
        condition_avg_movement (ndarray)
    """
    unique_angles = np.unique(angle)
    condition_avg_data = np.stack([np.mean(data[:, angle==a], axis=1) for a in unique_angles], axis=1)
    condition_avg_movement = np.stack([np.mean(movement[:, angle==a], axis=1) for a in unique_angles], axis=1)

    return condition_avg_data, condition_avg_movement
