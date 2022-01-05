from numpy.random import normal

import numpy as np
import pandas as pd


def gen_data_x(n: int) -> np.ndarray:
    return normal(0, 1, n)


def gen_data_A(m: int, n: int) -> np.ndarray:
    return normal(0, 1, m * n).reshape([m, n])


def gen_data_eps(sigma: np.float, m: int) -> np.ndarray:
    return normal(0, sigma, m)


def compute_corr(a: np.ndarray, b: np.ndarray) -> np.float:
    """ compute corr between two vectors"""
    data = pd.DataFrame({'a': a, 'b': b})
    corr = data.corr()
    return corr['a']['b']
