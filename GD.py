import numpy as np
from numpy import ndarray as nd


class GD:

    def __init__(self, n, m, A: nd, b: nd, x_gt: nd, stopping_gap: nd = 0.1):
        self.n = n
        self.m = m  # 200 个数据
        self.A: nd = A
        self.b: nd = b
        self.x_gt: nd = x_gt
        self.x: nd = self.starting_point()
        self.last_x: nd = self.x

        self.beta = 0.8  # param for backtracking
        self.alpha = 0.1  # param for backtracking
        self.stopping_gap = stopping_gap

    def starting_point(self) -> nd:
        return np.random.normal(0, 1, self.n)

    @staticmethod
    def compute_norm(a: nd, b: nd) -> nd:
        return np.linalg.norm(a-b)

    def f(self, x) -> nd:
        """原函数"""
        return np.linalg.norm(np.matmul(self.A, self.x)-self.b) ** 2

    def f_der(self, x: nd) -> nd:
        """导函数（除以2之后）"""
        return np.matmul(self.A.T, self.A, x) - np.matmul(self.A.T, self.b)

    def backtracking(self, delta_x: nd) -> float:
        """
        backtracking line search
        :param delta_x: 梯度方向
        """
        t: float = 1
        while self.f(self.x+t*delta_x) - (self.f(self.x) + self.alpha * t * np.dot(self.f_der(self.x), delta_x)) >= 0:
            t = self.beta * t
        return t

    def update(self, t:float, delta_x: nd):
        """
        :param t: step size
        :param delta_x: direction
        """
        self.x += t*delta_x

    def gd(self) -> nd:
        self.x = self.starting_point()
        self.last_x = self.x
        while self.norm(self.x, self.last_x) > self.stopping_gap:
            self.last_x = x
            _der = self.f_der(self.x)  # determine the descent direction
            _t = self.backtracking(_der)  # compute the step size
            self.update(_t, delta_x)
        return self.x
