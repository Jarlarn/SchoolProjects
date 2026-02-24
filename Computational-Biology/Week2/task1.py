import numpy as np
import matplotlib.pyplot as plt

# Parameters
r, K, A, B, D = 0.5, 8, 1, 1, 1
rho, q = r / A, K / B
L, dx, dt, T = 100, 1, 0.01, 40
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


# Function to estimate wave velocity c
def estimate_wave_velocity(u_init, steps, dx, dt, rho, q, D, xi, u_front):
    # Choose two time points: t1 (halfway), t2 (end)
    t1 = steps // 2
    t2 = steps
    u = u_init.copy()
    front_pos_t1 = None
    front_pos_t2 = None
    for t in range(steps + 1):
        if t == t1:
            # Find position where u crosses u_front
            idx = np.argmin(np.abs(u - u_front))
            front_pos_t1 = xi[idx]
        if t == t2:
            idx = np.argmin(np.abs(u - u_front))
            front_pos_t2 = xi[idx]
        if t < steps:
            lap_u = laplacian(u, dx)
            du_dt = rho * u * (1 - u / q) - (u / (1 + u)) + D * lap_u
            u += dt * du_dt
            u[[0, -1]] = u[[1, -2]]
    if front_pos_t1 is not None and front_pos_t2 is not None:
        c = (front_pos_t2 - front_pos_t1) / ((t2 - t1) * dt)
        return c
    else:
        return None


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

    # Estimate wave velocity c
    # Use u_front = (u0) / 2 as the threshold for the wave front
    c = estimate_wave_velocity(
        initialize_ramp(L, u0, xi0, dx)[1], steps, dx, dt, rho, q, D, xi, u0 / 2
    )
    print(f"Estimated wave velocity c for {label}: {c:.4f}")

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
