import numpy as np
import matplotlib.pyplot as plt

# Parameters
r, K, A, B, D = 0.5, 8, 1, 1, 1
rho, q = r / A, K / B
L, dx, dt, T = 100, 1, 0.01, 500  # Increase T for a longer simulation
steps = int(T / dt)

u1_steady = (1 - q) / 2 + np.sqrt(((q - 1) ** 2) / 4 - q * (rho - 1) / rho)
u2_steady = (1 - q) / 2 - np.sqrt(((q - 1) ** 2) / 4 - q * (rho - 1) / rho)


# Function to compute the Laplacian using the second-order symmetric derivative
def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return lap


# Function to initialize the smoothed ramp function
def initialize_ramp(L, u0, xi0, dx):
    xi = np.arange(1, L + 1) * dx
    u = u0 / (1 + np.exp(xi - xi0))
    return xi, u


# Function to simulate population dynamics
def simulate_population(u, steps, dx, dt, rho, q, D):
    for t in range(steps):
        lap_u = laplacian(u, dx)
        du_dt = rho * u * (1 - u / q) - (u / (1 + u / q)) + D * lap_u
        u += dt * du_dt
        u[[0, -1]] = u[[1, -2]]  # Zero-flux boundary conditions
    return u


# Function to compute the derivative of u with respect to xi
def compute_derivative(u, dx):
    du_dxi = np.zeros_like(u)
    du_dxi[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    return du_dxi


# Simulate for each parameter set
parameter_sets = [
    {"xi0": 20, "u0": u1_steady, "label": r"$\xi_0=20, u_0=u_1^*$"},  # u0 = u1*
    {"xi0": 50, "u0": u2_steady, "label": r"$\xi_0=50, u_0=u_2^*$"},  # u0 = u2*
    {
        "xi0": 50,
        "u0": 1.1 * u2_steady,
        "label": r"$\xi_0=50, u_0=1.1u_2^*$",
    },  # u0 = 1.1 * u2*
]

# Initialize the figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loop through parameter sets and plot results
for params in parameter_sets:
    xi0, u0, label = params["xi0"], params["u0"], params["label"]
    xi, u = initialize_ramp(L, u0, xi0, dx)
    u = simulate_population(u, steps, dx, dt, rho, q, D)
    du_dxi = compute_derivative(u, dx)

    # Plot wave profile on the left panel
    axes[0].plot(xi, u, label=label)

    # Plot phase plane trajectory on the right panel
    axes[1].plot(u, du_dxi, label=label)

# Customize left panel
axes[0].set_xlabel("Position (ξ)")
axes[0].set_ylabel("Population Density (u)")
axes[0].set_title("Wave Profiles")
axes[0].legend()

# Customize right panel
axes[1].set_xlabel("u")
axes[1].set_ylabel("du/dξ")
axes[1].set_title("Phase Plane")
axes[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
