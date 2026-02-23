import numpy as np
import matplotlib.pyplot as plt

# ─── Parameters ──────────────────────────────────────────────────────────────
A = 20.0
K = 100.0
r = 0.1
N0 = 50.0


def euler_dde(T, tEnd=500, dt=0.01):
    n_steps = int(tEnd / dt)
    tt = np.linspace(0, tEnd, n_steps)
    xx = np.zeros(n_steps)
    delay_steps = int(T / dt)
    xx[: delay_steps + 1] = N0  # history

    for i in range(delay_steps + 1, n_steps):
        x_delay = xx[i - delay_steps] if (i - delay_steps) >= 0 else N0
        x_now = xx[i - 1]
        dxdt = r * x_now * (1 - x_delay / K) * (x_now / A - 1)
        xx[i] = x_now + dt * dxdt

    return tt, xx


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Show different dynamics
# ═══════════════════════════════════════════════════════════════════════════════
print("=== PART A ===")

Tvals_A = [0.9, 1.0, 1.1, 5.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, Tv in enumerate(Tvals_A):
    tt, sol = euler_dde(Tv, tEnd=300)
    axes[i].plot(tt, sol)
    axes[i].axhline(K, color="gray", ls="--", lw=0.8, label="K")
    axes[i].axhline(A, color="orange", ls="--", lw=0.8, label="A")
    axes[i].set_title(f"T = {Tv}")
    axes[i].set_xlabel("t")
    axes[i].set_ylabel("x(t)")
    axes[i].set_ylim(99.99999, 100.00001)
    axes[i].legend(fontsize=8)

plt.suptitle("Part A — Different Dynamics", fontsize=14)
plt.tight_layout()
plt.savefig("partA.png", dpi=150)
plt.show()

print("T = 0.1 : no oscillations (monotone)")
print("T = 1.5 : damped oscillations")
print("T = 3.0 : damped / near-Hopf")
print("T = 5.0 : sustained oscillations (limit cycle)")


# ═══════════════════════════════════════════════════════════════════════════════
# PART B
# ═══════════════════════════════════════════════════════════════════════════════


T_values = np.round(np.arange(0.1, 5.01, 0.1), 1)
threshold = 1e-4
for T in T_values:
    tt, sol = euler_dde(T, tEnd=300)
    dxdt = np.gradient(sol, tt)
    dxdt[np.abs(dxdt) < threshold] = 0
    sign_changes = np.sum(np.diff(np.sign(dxdt)) != 0)
    if sign_changes > 2:
        T_damped = np.round(T, 1)
        break
print(f"T_damped is {T_damped}")


# # ═══════════════════════════════════════════════════════════════════════════════
# # PART C:
# # ═══════════════════════════════════════════════════════════════════════════════


def is_sustained(T, tEnd=500, dt=0.01):
    tt, sol = euler_dde(T, tEnd=tEnd, dt=dt)
    mask = tt >= tEnd - 80
    tail = sol[mask]
    amp = (tail.max() - tail.min()) / 2
    return amp > 10


TH_lo, TH_hi = 2.0, 5.0
while TH_hi - TH_lo > 0.1:
    TH_mid = (TH_lo + TH_hi) / 2
    if is_sustained(TH_mid):
        TH_hi = TH_mid
    else:
        TH_lo = TH_mid

TH_numerical = round((TH_lo + TH_hi) / 2, 1)
print(f"T_H (numerical) ≈ {TH_numerical}")
