import numpy as np


def mse(x, y): return np.mean((x-y)**2)


def flatten_mid_dims(x):
    """
    in:
        x (ndarray): time x ... x neuron
    out:
        x (ndarray): time x K x neuron
    """

    x_shape = x.shape
    data_flatten = x.reshape(x_shape[0], -1, x_shape[-1])

    return data_flatten


def split(x, n, dn):
    """
    in:
        x (ndarray): trial x time x neuron
        n (int): the index to start the split at
        dn (int): split ends at n+dn
    out:
        train_x (ndarray): trial-dn x time x neuron
        test_x (ndarray): dn x time x neuron
    """

    test_x = x[:, n:n + dn]
    train_x = np.concatenate([x[:, :n], x[:, n + dn:]], axis=1)

    return train_x, test_x