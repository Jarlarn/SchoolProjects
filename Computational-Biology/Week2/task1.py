import numpy as np
import matplotlib.pyplot as plt

# Parameters
r, K, A, B, D = 0.5, 8, 1, 1, 1
rho, q = r / A, K / B
L, dx, dt, T = 100, 1, 0.01, 10
steps = int(T / dt)

u1_steady = (q - 1) / 2 + np.sqrt(((q - 1) ** 2) / 4 + (q - (q / rho)))
u2_steady = (q - 1) / 2 - np.sqrt(((q - 1) ** 2) / 4 + (q - (q / rho)))

print(u1_steady, u2_steady)


def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return lap


def initialize_ramp(L, u0, xi0, dx):
    xi = np.arange(1, L + 1) * dx
    u = u0 / (1 + np.exp(xi - xi0))
    return xi, u


def simulate_population(u, steps, dx, dt, rho, q, D):
    for t in range(steps):
        lap_u = laplacian(u, dx)
        du_dt = rho * u * (1 - u / q) - (u / (1 + u)) + D * lap_u
        u += dt * du_dt
        u[[0, -1]] = u[[1, -2]]  # Zero-flux boundary conditions
    return u


def compute_derivative(u, dx):
    du_dxi = np.zeros_like(u)
    du_dxi[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    return du_dxi


parameter_sets = [
    {"xi0": 20, "u0": u1_steady, "label": r"$\xi_0=20, u_0=u_1^*$"},  # u0 = u1*
    {"xi0": 50, "u0": u2_steady, "label": r"$\xi_0=50, u_0=u_2^*$"},  # u0 = u2*
    {
        "xi0": 50,
        "u0": 1.1 * u2_steady,
        "label": r"$\xi_0=50, u_0=1.1u_2^*$",
    },  # u0 = 1.1 * u2*
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for params in parameter_sets:
    xi0, u0, label = params["xi0"], params["u0"], params["label"]
    xi, u = initialize_ramp(L, u0, xi0, dx)
    u = simulate_population(u, steps, dx, dt, rho, q, D)
    du_dxi = compute_derivative(u, dx)

    axes[0].plot(xi, u, label=label)

    axes[1].plot(u, du_dxi, label=label)

axes[0].set_xlabel("Position (ξ)")
axes[0].set_ylabel("Population Density (u)")
axes[0].set_title("Wave Profiles")
axes[0].legend()

axes[1].set_xlabel("u")
axes[1].set_ylabel("du/dξ")
axes[1].set_title("Phase Plane")
axes[1].legend()

plt.tight_layout()
plt.savefig("1b")
plt.show()
