from data_generator import *


if __name__ == '__main__':
    x_gt = gen_data_x()
    A = gen_data_A()
    _b_real = np.matmul(A, x_gt)
    var_Ax = np.var(_b_real)
    var_eps = var_Ax / 20
    eps = gen_data_eps(np.sqrt(var_eps))
    b_got = _b_real + eps

    c = 0