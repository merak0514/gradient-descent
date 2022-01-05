from numpy.random import normal
from typing import Tuple
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

def gen_data_all(m, n) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_gt = gen_data_x(n)
    A = gen_data_A(m, n)
    _b_real = np.matmul(A, x_gt)
    var_Ax = np.var(_b_real)
    var_eps = var_Ax / 20
    eps = gen_data_eps(np.sqrt(var_eps), m)
    b = _b_real + eps
    return A, x_gt, b
