import numpy as np
from numpy import ndarray as nd


class GD:
    def __init__(self, n, m, A: nd, b: nd, x_gt: nd, stopping_gap: float = 0.1,
                 mode: str = "backtracking", lr: float = 0.001):
        self.n = n
        self.m = m  # 200 个数据
        self.A: nd = A
        self.b: nd = b
        self.x_gt: nd = x_gt
        self.x: nd = self.starting_point()
        self.last_x: nd = self.x
        if mode not in ["backtracking", "exact", "fixed"]:
            raise ValueError('mode should within ["backtracking", "exact", "fixed"]')
        self.lr = lr  # only useful when mode is set to fixed.


        # params
        self.beta = 0.3  # param for backtracking
        self.alpha = 0.1  # param for backtracking
        self.stopping_gap = stopping_gap

        # stats
        self.stat_improves = []
        self.stat_real_gap = []

    def starting_point(self) -> nd:
        return np.zeros(self.n)

    @staticmethod
    def compute_norm(a: nd, b: nd) -> nd:
        return np.linalg.norm(a-b)

    def f(self, x) -> nd:
        """原函数"""
        return np.linalg.norm(np.matmul(self.A, x)-self.b) ** 2

    def f_der(self, x: nd) -> nd:
        """导函数（除以2之后）"""
        return np.matmul(np.matmul(self.A.T, self.A), x) - np.matmul(self.A.T, self.b)

    def backtracking(self, delta_x: nd) -> float:
        """
        backtracking line search
        :param delta_x: 梯度方向
        """
        t: float = 1
        judge = self.f(self.x+t*delta_x) - (
                self.f(self.x) + self.alpha * t * np.dot(self.f_der(self.x), delta_x))
        while judge >= 0:
            t = self.beta * t
            judge = self.f(self.x + t * delta_x) - (
                        self.f(self.x) + self.alpha * t * np.dot(self.f_der(self.x), delta_x))
        return t

    def update(self, t:float, delta_x: nd):
        """
        :param t: step size
        :param delta_x: direction
        """
        self.x += t*delta_x

    @staticmethod
    def normalization(x):
        x = (x-np.mean(x))/(np.max(x)-np.min(x))
        return x

    def gd(self) -> nd:
        self.x = self.starting_point()
        improve = 100
        while improve > self.stopping_gap:
            self.last_x = self.x.copy()
            _direct = -self.f_der(self.x)  # determine the descent direction
            _direct_standard = self.normalization(_direct)
            _t = self.backtracking(_direct_standard)  # compute the step size
            self.update(_t, _direct_standard)
            improve = self.compute_norm(self.x, self.last_x)
            self.stat_improves.append(improve)
            self.stat_real_gap.append(self.compute_norm(self.x, self.x_gt))
        return self.x

