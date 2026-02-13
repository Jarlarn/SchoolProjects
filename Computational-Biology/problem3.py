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

plt.figure(figsize=(16, 6))
plt.scatter(R_plot, eta_plot, s=2, color="k", alpha=0.5)
plt.xlabel("R")
plt.ylabel(r"$\eta$ (adults)")
plt.title("Bifurcation diagram of the Ricker map")
plt.tight_layout()
plt.savefig("ricker_bifurcation.png")
plt.show()

# --- (b) Population dynamics for representative R values ---
R_reps = [2, 10.0, 22.5, 13.5]
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


def detect_cycle_length(eta, tol=1e-6):
    unique_vals = np.unique(np.round(eta, decimals=6))
    return len(unique_vals)


for R in R_reps:
    eta = ricker_map(R, alpha, eta0, generations)
    cycle_length = detect_cycle_length(eta[-last:])
    print(f"Cycle length for R={R}: {cycle_length}")


# --- (d) Zoom in to estimate R_infty ---
R_zoom = np.arange(14.6, 15, 0.001)
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

cycle_lengths = []

for R in R_zoom:
    eta = ricker_map(R, alpha, eta0, generations)
    cycle_len = detect_cycle_length(eta[-last:])
    cycle_lengths.append(cycle_len)

# Plot cycle length vs R
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(R_zoom, cycle_lengths, ".-")
plt.xlabel("R")
plt.ylabel("Number of unique points (cycle length)")
plt.title("Numerical check for period-doubling and chaos")
plt.tight_layout()
plt.savefig("ricker_cycle_length_check.png")
plt.show()

# Print R values where cycle length jumps (indicative of bifurcations)
for R, cl in zip(R_zoom, cycle_lengths):
    if cl > 16:  # Arbitrary threshold for chaos
        print(f"Possible onset of chaos at R ≈ {R:.3f} (cycle length: {cl})")
        break
