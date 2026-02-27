import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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


def phase_plane_streamplot(ax, c, rho, q, D, u_range=None, v_range=None, density=1.5):
    if u_range is None:
        u_range = (-0.5, q * 1.1)
    if v_range is None:
        v_range = (-2.0, 2.0)

    U_grid = np.linspace(u_range[0], u_range[1], 200)
    V_grid = np.linspace(v_range[0], v_range[1], 200)
    U, V = np.meshgrid(U_grid, V_grid)

    dU = V
    f_U = rho * U * (1 - U / q) - U / (1 + U)
    dV = -(c / D) * V - (1 / D) * f_U

    speed = np.sqrt(dU**2 + dV**2)
    speed[speed == 0] = 1

    ax.streamplot(
        U_grid,
        V_grid,
        dU,
        dV,
        color=speed,
        cmap="coolwarm",
        density=density,
        linewidth=0.6,
        arrowsize=0.8,
    )

    u_fps = [0.0]
    u2_fp = (q - 1) / 2 - np.sqrt(((q - 1) ** 2) / 4 + (q - (q / rho)))
    u1_fp = (q - 1) / 2 + np.sqrt(((q - 1) ** 2) / 4 + (q - (q / rho)))
    u_fps += [u2_fp, u1_fp]
    for u_fp in u_fps:
        if u_range[0] <= u_fp <= u_range[1]:
            ax.plot(u_fp, 0, "ko", markersize=5, zorder=5)


def estimate_wave_velocity(u_init, steps, dx, dt, rho, q, D, xi, u_front):
    t1 = steps // 2
    t2 = steps
    u = u_init.copy()
    front_pos_t1 = None
    front_pos_t2 = None
    for t in range(steps + 1):
        if t == t1:
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
    return re.sub(r"[^a-zA-Z0-9]", "", label)


def classify_fixed_points(c, rho, q, D):
    """Compute Jacobian eigenvalues at each fixed point and classify."""
    fixed_points = {"U=0": 0.0, "U=u2*": u2_steady, "U=u1*": u1_steady}

    # f'(U) = ρ(1 - 2U/q) - 1/(1+U)^2
    def f_prime(U):
        return rho * (1 - 2 * U / q) - 1 / (1 + U) ** 2

    print("\n" + "=" * 60)
    print(f"Fixed-point classification  (c = {c:.4f})")
    print("=" * 60)
    for name, U_star in fixed_points.items():
        fp = f_prime(U_star)
        # Jacobian entries
        trace = -c / D
        det = fp / D
        disc = trace**2 - 4 * det
        lam1 = (trace + np.sqrt(complex(disc))) / 2
        lam2 = (trace - np.sqrt(complex(disc))) / 2

        # Classification
        if det < 0:
            kind = "Saddle point"
        elif disc > 0:
            if trace < 0:
                kind = "Stable node"
            elif trace > 0:
                kind = "Unstable node"
            else:
                kind = "Center (degenerate)"
        else:  # disc <= 0  (complex eigenvalues)
            if trace < 0:
                kind = "Stable spiral"
            elif trace > 0:
                kind = "Unstable spiral"
            else:
                kind = "Center"

        print(f"\n  {name:8s}  (U* = {U_star:+.4f})")
        print(f"    f'(U*) = {fp:+.4f}")
        print(f"    tr(J)  = {trace:+.4f},  det(J) = {det:+.4f},  Δ = {disc:+.4f}")
        print(f"    λ₁ = {lam1:+.4f},  λ₂ = {lam2:+.4f}")
        print(f"    → {kind}")
    print("=" * 60 + "\n")


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

    c = estimate_wave_velocity(
        initialize_ramp(L, u0, xi0, dx)[1], steps, dx, dt, rho, q, D, xi, u0 / 2
    )
    print(f"Estimated wave velocity c for {label}: {c:.4f}")

    classify_fixed_points(c, rho, q, D)

    u = simulate_population(u_init.copy(), steps, dx, dt, rho, q, D)
    du_dxi = compute_derivative(u, dx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(xi, u, label=label)
    axes[0].set_xlabel("Position (ξ)")
    axes[0].set_ylabel("Population Density (u)")
    axes[0].set_title("Wave Profile")
    axes[0].legend()
    u_min, u_max = min(np.min(u) - 0.5, -0.5), max(np.max(u) + 0.5, q * 1.1)
    v_min, v_max = np.min(du_dxi) - 0.5, np.max(du_dxi) + 0.5
    v_span = max(abs(v_min), abs(v_max), 1.0)
    if c is not None:
        phase_plane_streamplot(
            axes[1],
            c,
            rho,
            q,
            D,
            u_range=(u_min, u_max),
            v_range=(-v_span, v_span),
        )
    axes[1].plot(u, du_dxi, "k-", lw=2, label=label, zorder=4)
    n_arrows = 8
    indices = np.linspace(0, len(u) - 2, n_arrows, dtype=int)
    for idx in indices:
        axes[1].annotate(
            "",
            xytext=(u[idx], du_dxi[idx]),
            xy=(u[idx + 1], du_dxi[idx + 1]),
            arrowprops=dict(arrowstyle="->", color="k", lw=2),
            zorder=5,
        )
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("du/dξ")
    axes[1].set_title("Phase Plane")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        f"wave_phase_{label.replace('$', '').replace('\\', '').replace('^*', '').replace('=', '').replace(',', '').replace(' ', '').replace('_', '')}.png"
    )
    plt.close(fig)


# Smoothed peak initial condition cases
peak_cases = [
    {"u0": u1_steady, "label": r"$u_0=u_1^*$"},
    {"u0": 3 * u1_steady, "label": r"$u_0=3u_1^*$"},
]
xi0_peak = 50

for case in peak_cases:
    xi, u_init = initialize_peak(L, case["u0"], xi0_peak, dx)

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
