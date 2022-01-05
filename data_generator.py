from numpy.random import normal
from numpy.random import seed
import numpy as np
import pandas as pd

seed(777)

n = 50
m = 200  # 200 个数据


def gen_data_x() -> np.ndarray:
    return normal(0, 1, n)


def gen_data_A() -> np.ndarray:
    return normal(0, 1, m * n).reshape([m, n])


def gen_data_eps(sigma: np.float) -> np.ndarray:
    return normal(0, sigma, m)


def compute_corr(a: np.ndarray, b: np.ndarray) -> np.float:
    """ compute corr between two vectors"""
    data = pd.DataFrame({'a': a, 'b': b})
    corr = data.corr()
    return corr['a']['b']

