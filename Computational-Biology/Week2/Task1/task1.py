import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
r, K, A, B, D = 0.5, 8, 1, 1, 1
rho, q = r / A, K / B
L, dx, dt, T = 100, 1, 0.01, 200
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


def initialize_peak(L, u0, xi0, dx):
    xi = np.arange(1, L + 1) * dx
    u = u0 * np.exp(-((xi - xi0) ** 2))
    return xi, u


def simulate_population(u, steps, dx, dt, rho, q, D):
    for t in range(steps):
        lap_u = laplacian(u, dx)
        du_dt = rho * u * (1 - u / q) - (u / (1 + u)) + D * lap_u
        u += dt * du_dt
        u[[0, -1]] = u[[1, -2]]  # Zero-flux boundary conditions
    return u


def simulate_population_history(u, steps, dx, dt, rho, q, D, record_every=10):
    """Simulate and record snapshots of the population at regular intervals."""
    history = [u.copy()]
    for t in range(steps):
        lap_u = laplacian(u, dx)
        du_dt = rho * u * (1 - u / q) - (u / (1 + u)) + D * lap_u
        u += dt * du_dt
        u[[0, -1]] = u[[1, -2]]
        if (t + 1) % record_every == 0:
            history.append(u.copy())
    return history


def animate_simulation(
    xi,
    u_init,
    steps,
    dx,
    dt,
    rho,
    q,
    D,
    record_every=10,
    title="Population Wave Animation",
    interval=50,
    save_path=None,
):
    u = u_init.copy()
    history = simulate_population_history(u, steps, dx, dt, rho, q, D, record_every)
    times = [0] + [(i + 1) * record_every * dt for i in range(len(history) - 1)]

    fig, ax = plt.subplots(figsize=(10, 5))
    (line,) = ax.plot(xi, history[0], color="blue", lw=2)
    ax.set_xlim(xi[0], xi[-1])
    ax.set_ylim(0, max(np.max(h) for h in history) * 1.1)
    ax.set_xlabel("Position (ξ)")
    ax.set_ylabel("Population Density (u)")
    ax.set_title(title)
    time_text = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment="top"
    )

    def update(frame):
        line.set_ydata(history[frame])
        time_text.set_text(f"t = {times[frame]:.2f}")
        return line, time_text

    anim = FuncAnimation(fig, update, frames=len(history), interval=interval, blit=True)

    if save_path is not None:
        anim.save(
            save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg"
        )
        print(f"Animation saved to {save_path}")
    else:
        plt.show()

    return anim


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


# Helper to sanitize filenames
import re


def sanitize_filename(label):
    # Remove or replace problematic characters for Windows filenames
    return re.sub(r"[^a-zA-Z0-9]", "", label)


parameter_sets = [
    {"xi0": 20, "u0": u1_steady, "label": r"$\xi_0=20, u_0=u_1^*$"},  # u0 = u1*
    {"xi0": 50, "u0": u2_steady, "label": r"$\xi_0=50, u_0=u_2^*$"},  # u0 = u2*
    {
        "xi0": 50,
        "u0": 1.1 * u2_steady,
        "label": r"$\xi_0=50, u_0=1.1u_2^*$",
    },  # u0 = 1.1 * u2*
]


for params in parameter_sets:
    xi0, u0, label = params["xi0"], params["u0"], params["label"]
    xi, u_init = initialize_ramp(L, u0, xi0, dx)

    # Animate the simulation

    safe_label = sanitize_filename(params["label"])
    animate_simulation(
        xi,
        u_init,
        steps,
        dx,
        dt,
        rho,
        q,
        D,
        record_every=100,
        title=f"Wave Animation for {params['label']} (Ramp IC)",
        interval=50,
        save_path=f"animation_{safe_label}.gif",
    )

    u = simulate_population(u_init.copy(), steps, dx, dt, rho, q, D)
    du_dxi = compute_derivative(u, dx)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Wave profile
    axes[0].plot(xi, u, label=label)
    axes[0].set_xlabel("Position (ξ)")
    axes[0].set_ylabel("Population Density (u)")
    axes[0].set_title("Wave Profile")
    axes[0].legend()
    # Phase plane
    axes[1].plot(u, du_dxi, label=label)
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("du/dξ")
    axes[1].set_title("Phase Plane")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        f"wave_phase_{label.replace('$', '').replace('\\', '').replace('^*', '').replace('=', '').replace(',', '').replace(' ', '').replace('_', '')}.png"
    )
    plt.close(fig)

    # Estimate wave velocity c
    c = estimate_wave_velocity(
        initialize_ramp(L, u0, xi0, dx)[1], steps, dx, dt, rho, q, D, xi, u0 / 2
    )
    print(f"Estimated wave velocity c for {label}: {c:.4f}")


# Smoothed peak initial condition cases
peak_cases = [
    {"u0": u1_steady, "label": r"$u_0=u_1^*$"},
    {"u0": 3 * u1_steady, "label": r"$u_0=3u_1^*$"},
]
xi0_peak = 50

for case in peak_cases:
    xi, u_init = initialize_peak(L, case["u0"], xi0_peak, dx)

    # Animate the simulation
    safe_label = sanitize_filename(case["label"])
    animate_simulation(
        xi,
        u_init,
        steps,
        dx,
        dt,
        rho,
        q,
        D,
        record_every=10,
        title=f"Wave Animation for {case['label']} (Peak IC)",
        interval=50,
        save_path=f"animation_{safe_label}.gif",
    )

    # Static final plot
    sol = simulate_population(u_init.copy(), steps, dx, dt, rho, q, D)
    du_dxi = compute_derivative(sol, dx)

    plt.figure(figsize=(8, 4))
    plt.plot(xi, sol, label=case["label"])
    plt.xlabel("Position (ξ)")
    plt.ylabel("Population Density (u)")
    plt.title(f"Wave Profile for {case['label']} (Peak Initial Condition)")
    plt.legend()
    plt.tight_layout()
    plt.show()
