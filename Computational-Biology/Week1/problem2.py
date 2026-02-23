import numpy as np
import matplotlib.pyplot as plt

# Parameters
K = 1e3
r = 0.1
b = 1


def population_next(N, K, r, b):
    return (r + 1) * N / (1 + (N / K) ** b)


# Fixed points
def fixed_points(K, r, b):
    N1 = 0
    N2 = K * r ** (1 / b)
    return N1, N2


def eigenvalues(K, r, b):
    # At N1 = 0
    lambda1 = r + 1
    # At N2 = K * r^{1/b}
    lambda2 = (1 + (1 - b) * r) / (1 + r)
    return lambda1, lambda2


def simulate(N0, K, r, b, steps=50):
    N = np.zeros(steps)
    N[0] = N0
    for t in range(steps - 1):
        N[t + 1] = population_next(N[t], K, r, b)
    return N


def linearized_solution(N0, N_star, eigenvalue, steps=50):
    delta = N0 - N_star
    return N_star + delta * eigenvalue ** np.arange(steps)


if __name__ == "__main__":
    N0_values = [1, 2, 3, 10]
    steps = 100
    N1, N2 = fixed_points(K, r, b)
    lambda1, lambda2 = eigenvalues(K, r, b)

    # Task d)
    N_star = N1
    eig = lambda1

    # Task f:
    N_star = N2
    eig = lambda2
    delta_N0_values = [-10, -3, -2, -1, 1, 2, 3, 10]

    # for delta_N0 in delta_N0_values:
    #     N0 = N_star + delta_N0
    #     N_exact = simulate(N0, K, r, b, steps)
    #     N_lin = linearized_solution(N0, N_star, eig, steps)
    #     plt.plot(np.arange(steps), N_exact, label=f"Exact $\\delta N_0$={delta_N0}")
    #     plt.plot(np.arange(steps), N_lin, "--", label=f"Linearized $\\delta N_0$={delta_N0}")

    plt.figure(figsize=(12, 7))
    for N0 in N0_values:
        N_exact = simulate(N0, K, r, b, steps)
        N_lin = linearized_solution(N0, N_star, eig, steps)
        plt.plot(np.arange(steps), N_exact, label=f"Exact N0={N0}")
        plt.plot(np.arange(steps), N_lin, "--", label=f"Linearized N0={N0}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time step")
    plt.ylabel("Population size (log-log scale)")
    plt.title("Population Dynamics: Exact vs Linearized (log-log scale)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("task2")
    # plt.savefig("task2f")
