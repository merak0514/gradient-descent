from numpy.random import seed
from data_generator import *
from algo import GD
seed(777)

n = 50
m = 200  # 200 个数据
stop_gap = 0.1  # 停止条件

if __name__ == '__main__':
    x_gt = gen_data_x(n)
    A = gen_data_A(m, n)
    _b_real = np.matmul(A, x_gt)
    var_Ax = np.var(_b_real)
    var_eps = var_Ax / 20
    eps = gen_data_eps(np.sqrt(var_eps), m)
    b = _b_real + eps
    print(f"相关性为{compute_corr(_b_real, b)}")
    gd = GD(n=n, m=m, A=A, b=b, x_gt=x_gt, stop_gap=stop_gap)
    x_est = gd._gd()

    gd = GD(n=n, m=m, A=A, b=b, x_gt=x_gt, stop_gap=stop_gap)
    x_est = gd._sgd()

