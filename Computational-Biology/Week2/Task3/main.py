import numpy as np
import matplotlib.pyplot as plt

gamma = 1.0
Kc = 2 * gamma
dt = 0.01
T = 50
steps = int(T / dt)

K_values = [1.5, 2.1, 4.0]
N_values = [20, 100, 300]

fig, axes = plt.subplots(
    len(N_values), len(K_values), figsize=(15, 10), sharex=True, sharey=True
)

for row, N in enumerate(N_values):
    omega = np.random.standard_cauchy(N) * gamma

    for col, K in enumerate(K_values):
        theta = np.random.uniform(-np.pi / 2, np.pi / 2, N)
        r_history = []

        for t in range(steps):
            r_complex = np.mean(np.exp(1j * theta))
            r = np.abs(r_complex)
            psi = np.angle(r_complex)
            r_history.append(r)

            dtheta = omega + K * r * np.sin(psi - theta)
            theta += dt * dtheta

        time = np.arange(steps) * dt
        mu = (K - Kc) / Kc
        if mu > 0:
            r_theory = np.sqrt(mu)
        else:
            r_theory = 0

        ax = axes[row, col]
        ax.plot(time, r_history, "b-", lw=0.8, label="Simulation")
        ax.axhline(
            r_theory, color="r", ls="--", lw=1.5, label=f"Theory $r = {r_theory:.2f}$"
        )
        ax.set_ylim(0, 1.1)
        ax.set_title(f"$K = {K}$, $N = {N}$", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        if row == len(N_values) - 1:
            ax.set_xlabel("Time")
        if col == 0:
            ax.set_ylabel("Order parameter $r$")

fig.suptitle(f"Kuramoto model — $K_c = {Kc}$, $\\gamma = {gamma}$", fontsize=14)
fig.tight_layout()
plt.savefig("kuramoto.png", dpi=200)
plt.show()
