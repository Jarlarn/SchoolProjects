import numpy as np
import matplotlib.pyplot as plt
import os

a = 3.0
b = 8.0
D_u = 1.0
L = 128
dx = 1.0
dt = 0.01

u_star = a
v_star = b / a

Dv_values = [2.3, 3.0, 5.0, 9.0]

transient_iters = 1000
steady_iters = 100000
convergence_tol = 1e-6


def laplacian_2d(field, dx):
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (dx**2)


def simulate(D_v, n_iters, u_init, v_init, check_convergence=False):
    u = u_init.copy()
    v = v_init.copy()
    actual_iters = 0

    for i in range(n_iters):
        lap_u = laplacian_2d(u, dx)
        lap_v = laplacian_2d(v, dx)

        du = (a - (b + 1) * u + u**2 * v + D_u * lap_u) * dt
        dv = (b * u - u**2 * v + D_v * lap_v) * dt

        u += du
        v += dv
        actual_iters += 1

        if check_convergence and (i + 1) % 500 == 0:
            max_change = max(np.max(np.abs(du)), np.max(np.abs(dv)))
            if max_change < convergence_tol:
                print(f"    Converged at iteration {actual_iters}")
                break

    return u, v, actual_iters


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Steady state: (u*, v*) = ({u_star}, {v_star:.4f})")
    print(f"Grid: {L}x{L}, dt = {dt}, dx = {dx}\n")

    results = {}
    global_umin = float("inf")
    global_umax = float("-inf")

    for D_v in Dv_values:
        print(f"--- D_v = {D_v} ---")

        rng = np.random.default_rng(seed=42)
        u0 = u_star + 0.1 * u_star * (2.0 * rng.random((L, L)) - 1.0)
        v0 = v_star + 0.1 * v_star * (2.0 * rng.random((L, L)) - 1.0)

        u_trans, v_trans, _ = simulate(D_v, transient_iters, u0, v0)
        print(f"  Transient: u in [{u_trans.min():.4f}, {u_trans.max():.4f}]")

        u_ss, v_ss, ss_iters = simulate(
            D_v, steady_iters, u_trans, v_trans, check_convergence=True
        )
        print(f"  Steady state: u in [{u_ss.min():.4f}, {u_ss.max():.4f}]\n")

        results[D_v] = (u_trans, u_ss)
        global_umin = min(global_umin, u_trans.min(), u_ss.min())
        global_umax = max(global_umax, u_trans.max(), u_ss.max())

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        "Spatial distribution of $u$ — Belousov-Zhabotinsky reaction",
        fontsize=15,
        y=0.98,
    )

    for col, D_v in enumerate(Dv_values):
        u_trans, u_ss = results[D_v]

        ax_t = axes[0, col]
        ax_t.imshow(
            u_trans, cmap="hot", origin="lower", vmin=global_umin, vmax=global_umax
        )
        ax_t.set_title(
            f"$D_v = {D_v}$\nTransient (t = {transient_iters * dt:.0f})", fontsize=11
        )
        ax_t.set_xlabel("x")
        ax_t.set_ylabel("y")

        ax_s = axes[1, col]
        im_s = ax_s.imshow(
            u_ss, cmap="hot", origin="lower", vmin=global_umin, vmax=global_umax
        )
        ax_s.set_title(f"$D_v = {D_v}$\nSteady state", fontsize=11)
        ax_s.set_xlabel("x")
        ax_s.set_ylabel("y")

    fig.subplots_adjust(right=0.91, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im_s, cax=cbar_ax, label="Concentration $u$")

    plt.savefig(os.path.join(output_dir, "heatmaps.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "heatmaps.pdf"), bbox_inches="tight")
    print(f"Figures saved to {output_dir}/")
    plt.show()


if __name__ == "__main__":
    main()
