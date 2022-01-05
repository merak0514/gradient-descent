import numpy as np
from numpy import ndarray as nd
from matplotlib import pyplot as plt
import time


class GD:
    def __init__(self, n, m, A: nd, b: nd, x_gt: nd, stop_gap: float = 0.1,
                 mode: str = "backtracking", lr: float = 0.001):
        self.n = n
        self.m = m  # 200 个数据
        self.A: nd = A
        self.b: nd = b
        self.x_gt: nd = x_gt
        self.x: nd = self._starting_point()
        self.last_x: nd = self.x
        if mode not in ["backtracking", "exact", "fixed", "sgd"]:
            raise ValueError('mode should within ["backtracking", "exact", "fixed", "sgd"]')
        if mode == "exact":
            raise ValueError('Damn, not implemented yet!')
        self.mode = mode
        self.lr = lr  # only useful when mode is set to fixed.


        # params
        self.beta = 0.3  # param for backtracking
        self.alpha = 0.1  # param for backtracking
        self.stop_gap = stop_gap

        # sgd params
        self.sgd_m = 50
        self.max_round = 100

        # stats
        self.stat_improves = []
        self.stat_real_gap = []
        self.stat_step_count = 0
        self.stat_time = 0

    def _starting_point(self) -> nd:
        return np.zeros(self.n)

    @staticmethod
    def norm(a: nd, b: nd) -> nd:
        return np.linalg.norm(a-b)

    def _f(self, x) -> nd:
        """原函数"""
        return np.linalg.norm(np.matmul(self.A, x)-self.b) ** 2

    def _f_der(self, x: nd) -> nd:
        """导函数（除以2之后）"""
        return np.matmul(np.matmul(self.A.T, self.A), x) - np.matmul(self.A.T, self.b)

    def _backtracking(self, direction: nd) -> float:
        """
        backtracking line search
        :param direction: 梯度方向
        """
        t: float = 1
        judge = self._f(self.x + t * direction) - (
                self._f(self.x) + self.alpha * t * np.dot(self._f_der(self.x), direction))
        while judge >= 0:
            t = self.beta * t
            judge = self._f(self.x + t * direction) - (
                    self._f(self.x) + self.alpha * t * np.matmul(self._f_der(self.x).T, direction))
        return t

    def _exact(self, direction: nd) -> float:
        """exact line search TBD； Edit: don't want to finish this shit."""
        return 1

    def _update(self, t: float, delta_x: nd):
        """
        :param t: step size
        :param delta_x: direction
        """
        self.x += t*delta_x

    @staticmethod
    def _normalization(x):
        x = (x-np.mean(x))/(np.max(x)-np.min(x))
        return x

    def _gd(self) -> nd:
        t1 = time.time()
        self.x = self._starting_point()
        improve = 100
        while improve > self.stop_gap:
            self.stat_step_count += 1
            self.last_x = self.x.copy()
            _direction = -self._f_der(self.x)  # determine the descent direction

            if self.mode == "backtracking":
                _t = self._backtracking(_direction)  # compute the step size
            elif self.mode == "fixed":
                _t = self.lr
            else:
                _t = self._exact(_direction)
            self._update(_t, _direction)
            improve = self.norm(self.x, self.last_x)
            self.stat_improves.append(improve)
            self.stat_real_gap.append(self.norm(self.x, self.x_gt))

        t2 = time.time()
        self.stat_time = t2-t1
        return self.x

    def _sgd(self) -> nd:
        def sgd_f(A, b, x) -> nd:
            """原函数"""
            return np.linalg.norm(np.matmul(A, x) - b) ** 2

        def sgd_f_der(A, b, x) -> nd:
            """导函数（除以2之后）"""
            return np.matmul(np.matmul(A.T, A), x) - np.matmul(A.T, b)

        def sgd_backtracking(self, direction: nd, A, b) -> float:
            """
            backtracking line search
            :param direction: 梯度方向
            """
            t: float = 1
            judge = sgd_f(A, b, self.x + t * direction) - (sgd_f(A, b, self.x) +
                     self.alpha * t * np.dot(sgd_f_der(A, b, self.x), direction))
            while judge >= 0:
                t = self.beta * t
                judge = sgd_f(A, b, self.x + t * direction) - (sgd_f(A, b, self.x) +
                                                               self.alpha * t * np.dot(sgd_f_der(A, b, self.x),
                                                                                       direction))

            return t

        t1 = time.time()
        round_count = 0
        while round_count <= self.max_round:
            chosen_ids = np.random.choice(range(0, self.m), self.sgd_m)
            chosen_A = np.array([self.A[i]for i in sorted(chosen_ids)])
            chosen_b = np.array([self.b[i]for i in sorted(chosen_ids)])
            
            _direction = -sgd_f_der(chosen_A, chosen_b, self.x)  # determine the descent direction

            _t = sgd_backtracking(self, _direction, chosen_A, chosen_b)  # compute the step size
            self._update(_t, _direction)
            self.stat_real_gap.append(self.norm(self.x, self.x_gt))
            round_count += 1
        self.stat_time = time.time() - t1
        return self.x

    def run(self, mode: str = None) -> nd:
        self.stat_improves = []
        self.stat_real_gap = []
        self.stat_step_count = 0
        self.stat_time = 0
        self.x = self._starting_point()
        if mode:
            if mode not in ["backtracking", "exact", "fixed", "sgd"]:
                raise ValueError('mode should within ["backtracking", "exact", "fixed", "sgd"]')
            self.mode = mode
        if self.mode == "sgd":
            return self._sgd()
        else:
            return self._gd()

    def draw_gaps(self):
        plt.title(self.mode)
        plt.xlabel("step")
        plt.ylabel("gap")
        x_ind = list(range(1, len(self.stat_real_gap)+1))
        plt.plot(x_ind, self.stat_real_gap)
        # plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.3))
        plt.show()


class PseudoInverse:
    def __init__(self, n, m, A: nd, b: nd, x_gt: nd):
        self.n = n
        self.m = m
        self.A: nd = A
        self.b: nd = b
        self.x_gt: nd = x_gt

        # stats
        self.stat_time = 0

    def compute(self):
        t1 = time.time()
        x = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(self.A.T, self.A)), self.A.T), self.b)
        self.stat_time = time.time()-t1
        return x
