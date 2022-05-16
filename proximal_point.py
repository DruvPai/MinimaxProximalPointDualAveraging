import torch
from tqdm import tqdm
import pathlib
import matplotlib.pyplot as plt

figpath = pathlib.Path("robust_linreg")


def proximal_operator(z_t, q_avg, eta, proj_Z, t_prox_gd=1000, alpha_prox_gd=1e-5):
    # finds min_(z in Z) {eta * <avg_q_k, z> + 1/2 |z - z_t|^2}
    # algorithm is projected gradient descent for T_PROX_GD steps
    # gradient of the update is z - (z_t - eta * avg_q_k)
    z = torch.randn(z_t.shape)
    for k in range(t_prox_gd):
        z = proj_Z(z - alpha_prox_gd * (z_t - eta * q_avg))
    return z


def F(x, y, dfdx, dfdy):
    return torch.cat(tensors=(dfdx(x, y), -dfdy(x, y)))


def mirror_descent(x_t, y_t, x_t_minus_1, y_t_minus_1, dfdx, dfdy):
    return F(x_t, y_t, dfdx, dfdy)


def operator_extrapolation(x_t, y_t, x_t_minus_1, y_t_minus_1, dfdx, dfdy):
    return 2 * F(x_t, y_t, dfdx, dfdy) - F(x_t_minus_1, y_t_minus_1, dfdx, dfdy)


def pdhg(x_t, y_t, x_t_minus_1, y_t_minus_1, dfdx, dfdy):
    return torch.cat(tensors=(2*dfdx(x_t, y_t) - dfdx(x_t_minus_1, y_t_minus_1), -dfdy(x_t, y_t_minus_1)))

A = torch.randn((15, 10))
b = torch.randn((15, ))

def f(x, y):
    return torch.linalg.norm(A @ (x + y) - b) ** 2


def dfdx(x, y):
    return 2 * A.T @ (A @ (x + y) - b)


def dfdy(x, y):
    return 2 * A.T @ (A @ (x + y) - b)


def minmax_optimization(d_x, d_y, f, dfdx, dfdy, eta, T, q_fn, proj_X, proj_Y):
    proj_Z = lambda z: torch.cat(tensors=(proj_X(z[:d_x]), proj_Y(z[d_x:d_x+d_y])))

    z_history = []
    f_history = []

    x_t_minus_1 = proj_X(torch.randn(d_x))
    y_t_minus_1 = proj_Y(torch.randn(d_y))

    x_t = proj_X(torch.randn(d_x))
    y_t = proj_Y(torch.randn(d_y))
    z_t = torch.cat(tensors=(x_t, y_t))

    z_history.append(z_t)
    f_history.append(f(x_t, y_t))

    q_avg = torch.zeros_like(z_t)

    for t in tqdm(range(T)):
        q_t = q_fn(x_t, y_t, x_t_minus_1, y_t_minus_1, dfdx, dfdy)
        q_avg = ((t * q_avg) + q_t) / (t + 1)

        x_t_minus_1 = x_t
        y_t_minus_1 = y_t

        z_t = proximal_operator(q_avg, eta, z_t, proj_Z)
        x_t = z_t[:d_x]
        y_t = z_t[d_x:d_x+d_y]

        z_history.append(z_t)
        f_history.append(f(x_t, y_t))

    return z_history, f_history


algs = {"mirror": mirror_descent, "oe": operator_extrapolation, "pdhg": pdhg}


for alg_name in algs:
    path = figpath / alg_name
    path.mkdir(exist_ok=True, parents=True)

    alg = algs[alg_name]
    z_history, f_history = minmax_optimization(
        10, 10, f, dfdx, dfdy, 0.01, 1000, alg,
        lambda x: x, lambda y: 0.001 * y / torch.linalg.norm(y)
    )
    plt.title("$f(x_{t}, y_{t})$")
    plt.xlabel("$t$")
    plt.plot(f_history)
    plt.savefig(path / "f_history.jpg")
    plt.close()

    # plt.title("$(x_{t}, y_{t})$")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.xlim([-10, 10])
    # plt.ylim([-10, 10])
    # plt.plot([z_history[t][0] for t in range(len(z_history))], [z_history[t][1] for t in range(len(z_history))], '-o')
    # plt.savefig(path / "z_history.jpg")
    # plt.close()
