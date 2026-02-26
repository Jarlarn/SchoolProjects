"""
Problem Set 2, Task 2: Diffusion-Driven Instability
Belousov-Zhabotinsky reaction simulation.

System:
  du/dt = a - (b+1)u + u^2 v + D_u * laplacian(u)
  dv/dt = b*u - u^2 v + D_v * laplacian(v)

Parameters: a=3, b=8, D_u=1, D_v > 1
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ─── Parameters ───────────────────────────────────────────────────────
a = 3.0
b = 8.0
D_u = 1.0
L = 128  # grid size L x L
dx = 1.0  # spatial step
dt = 0.01  # time step

# Steady state
u_star = a  # u* = a = 3
v_star = b / a  # v* = b/a = 8/3

# D_v values to test
Dv_values = [2.3, 3.0, 5.0, 9.0]

# Iteration counts
transient_iters = 1000
steady_iters = 100000  # max iterations to reach steady state
convergence_tol = 1e-6  # convergence criterion (max change per step)


def laplacian_2d(field, dx):
    """
    Compute the discrete 2D Laplacian with periodic boundary conditions
    using the 5-point stencil:
        nabla^2 f_{i,j} = (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4*f_{i,j}) / dx^2
    """
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx**2)


def reaction_u(u, v):
    """Reaction term for u: a - (b+1)*u + u^2*v"""
    return a - (b + 1.0) * u + u**2 * v


def reaction_v(u, v):
    """Reaction term for v: b*u - u^2*v"""
    return b * u - u**2 * v


def simulate(D_v, n_iters, u_init, v_init, check_convergence=False):
    """
    Forward-Euler integration of the reaction-diffusion system.

    Parameters
    ----------
    D_v : float
        Diffusion coefficient for v.
    n_iters : int
        Maximum number of iterations.
    u_init, v_init : ndarray
        Initial concentration fields.
    check_convergence : bool
        If True, stop early when the system reaches steady state.

    Returns
    -------
    u, v : ndarray
        Final concentration fields.
    actual_iters : int
        Number of iterations actually performed.
    """
    u = u_init.copy()
    v = v_init.copy()
    actual_iters = 0

    for i in range(n_iters):
        lap_u = laplacian_2d(u, dx)
        lap_v = laplacian_2d(v, dx)

        du = (reaction_u(u, v) + D_u * lap_u) * dt
        dv = (reaction_v(u, v) + D_v * lap_v) * dt

        u += du
        v += dv
        actual_iters += 1

        if check_convergence and (i + 1) % 500 == 0:
            max_change = max(np.max(np.abs(du)), np.max(np.abs(dv)))
            if max_change < convergence_tol:
                print(
                    f"    Converged at iteration {actual_iters} "
                    f"(max change = {max_change:.2e})"
                )
                break

    return u, v, actual_iters


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Steady state: (u*, v*) = ({u_star}, {v_star:.4f})")
    print(f"Grid: {L}x{L}, dt = {dt}, dx = {dx}\n")

    # ─── Determine global colour range ────────────────────────────────
    # Run all simulations first to find a consistent color range
    results = {}
    global_umin = float("inf")
    global_umax = float("-inf")

    for D_v in Dv_values:
        print(f"--- D_v = {D_v} ---")

        # Initial conditions: steady state + 10 % random perturbation
        rng = np.random.default_rng(seed=42)
        u0 = u_star + 0.1 * u_star * (2.0 * rng.random((L, L)) - 1.0)
        v0 = v_star + 0.1 * v_star * (2.0 * rng.random((L, L)) - 1.0)

        # Phase 1: transient snapshot (1000 iterations)
        u_trans, v_trans, _ = simulate(D_v, transient_iters, u0, v0)
        print(
            f"  Transient  ({transient_iters} iters): "
            f"u in [{u_trans.min():.4f}, {u_trans.max():.4f}]"
        )

        # Phase 2: continue to (approximate) steady state
        u_ss, v_ss, ss_iters = simulate(
            D_v, steady_iters, u_trans, v_trans, check_convergence=True
        )
        total_iters = transient_iters + ss_iters
        print(
            f"  Steady state ({total_iters} total iters): "
            f"u in [{u_ss.min():.4f}, {u_ss.max():.4f}]\n"
        )

        results[D_v] = (u_trans, u_ss)

        # Track global colour range
        global_umin = min(global_umin, u_trans.min(), u_ss.min())
        global_umax = max(global_umax, u_trans.max(), u_ss.max())

    # ─── Plot heat maps ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Spatial distribution of $u$ — Belousov-Zhabotinsky reaction",
        fontsize=15,
        y=0.98,
    )

    for col, D_v in enumerate(Dv_values):
        u_trans, u_ss = results[D_v]

        # Top row: transient (1000 iterations)
        ax_t = axes[0, col]
        im_t = ax_t.imshow(
            u_trans, cmap="hot", origin="lower", vmin=global_umin, vmax=global_umax
        )
        ax_t.set_title(
            f"$D_v = {D_v}$\nTransient (t = {transient_iters * dt:.0f})", fontsize=11
        )
        ax_t.set_xlabel("x")
        ax_t.set_ylabel("y")

        # Bottom row: steady state
        ax_s = axes[1, col]
        im_s = ax_s.imshow(
            u_ss, cmap="hot", origin="lower", vmin=global_umin, vmax=global_umax
        )
        ax_s.set_title(f"$D_v = {D_v}$\nSteady state", fontsize=11)
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")

    # Single shared colour bar
    fig.subplots_adjust(right=0.91, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im_s, cax=cbar_ax, label="Concentration $u$")

    plt.savefig(os.path.join(output_dir, "heatmaps.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "heatmaps.pdf"), bbox_inches="tight")
    print(f"Figures saved to {output_dir}/")
    plt.show()


if __name__ == "__main__":
    main()
