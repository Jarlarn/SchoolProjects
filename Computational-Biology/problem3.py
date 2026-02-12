import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01
eta0 = 900
generations = 300
last = 100


def ricker_map(R, alpha, eta0, generations):
    eta = np.zeros(generations)
    eta[0] = eta0
    for t in range(generations - 1):
        eta[t + 1] = R * eta[t] * np.exp(-alpha * eta[t])
    return eta


# --- (a) Bifurcation diagram ---
R_values = np.arange(1, 30.1, 0.1)
etas = []
for R in R_values:
    eta = ricker_map(R, alpha, eta0, generations)
    etas.append(eta[-last:])

R_plot = np.repeat(R_values, last)
eta_plot = np.concatenate(etas)

plt.figure(figsize=(10, 6))
plt.scatter(R_plot, eta_plot, s=2, color="k", alpha=0.5)
plt.xlabel("R")
plt.ylabel(r"$\eta$ (adults)")
plt.title("Bifurcation diagram of the Ricker map")
plt.tight_layout()
plt.savefig("ricker_bifurcation.png")
plt.show()

# --- (b) Population dynamics for representative R values ---
# Choose R values for: fixed point, 2-cycle, 3-cycle, 4-cycle
R_reps = [2, 18.5, 20.5, 22.5]
labels = ["Stable fixed point", "2-point cycle", "3-point cycle", "4-point cycle"]
plt.figure(figsize=(12, 8))
for R, label in zip(R_reps, labels):
    eta = ricker_map(R, alpha, eta0, 41)
    plt.plot(np.arange(41), eta, marker="o", label=f"R={R} ({label})")
plt.xlabel(r"$\tau$ (generation)")
plt.ylabel(r"$\eta_\tau$ (adults)")
plt.title("Population dynamics for representative R values")
plt.legend()
plt.tight_layout()
plt.savefig("ricker_representative_dynamics.png")
plt.show()

# --- (d) Zoom in to estimate R_infty ---
R_zoom = np.arange(22, 23, 0.001)
etas_zoom = []
for R in R_zoom:
    eta = ricker_map(R, alpha, eta0, generations)
    etas_zoom.append(eta[-last:])

R_plot_zoom = np.repeat(R_zoom, last)
eta_plot_zoom = np.concatenate(etas_zoom)

plt.figure(figsize=(10, 6))
plt.scatter(R_plot_zoom, eta_plot_zoom, s=2, color="k", alpha=0.5)
plt.xlabel("R")
plt.ylabel(r"$\eta$ (adults)")
plt.title("Zoomed bifurcation diagram near $R_\\infty$")
plt.tight_layout()
plt.savefig("ricker_bifurcation_zoom.png")
plt.show()

# Estimate R_infty visually from the plot
R_infty_est = 22.6
print(f"Estimated R_infty (onset of chaos): {R_infty_est}")
