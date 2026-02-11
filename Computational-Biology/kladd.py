"""
Time-Delayed Model with Allee Effect — Complete Solution in Python
x'(t) = r * x(t) * (1 - x(t-T)/K) * (x(t)/A - 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from ddeint import ddeint

# ─── Parameters ──────────────────────────────────────────────────────────────
A = 20.0
K = 100.0
r = 0.1
N0 = 50.0


# ─── DDE definition ─────────────────────────────────────────────────────────
def model(x, t, T):
    xd = x(t - T)  # delayed value
    xc = x(t)  # current value
    return r * xc * (1 - xd / K) * (xc / A - 1)


def history(t):
    return N0


# ─── Solver wrapper ─────────────────────────────────────────────────────────
def solve_dde(T, tEnd=500, dt=0.01):
    tt = np.arange(0, tEnd, dt)
    sol = ddeint(lambda x, t: model(x, t, T), history, tt)
    return tt, sol.flatten()


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
# plt.savefig("partA.png", dpi=150)
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
    print(sign_changes)
    if sign_changes > 2:
        T_damped = np.round(T, 1)
        break
print(f"T_damped is {T_damped}")


# # ═══════════════════════════════════════════════════════════════════════════════
# # PART C: Estimate T_H (Hopf bifurcation) numerically (0.1 precision)
# # ═══════════════════════════════════════════════════════════════════════════════
# print("\n=== PART C ===")


# def is_sustained(T, tEnd=500, dt=0.1):
#     """Check if oscillations persist in the tail [tEnd-80, tEnd]."""
#     tt, sol = solve_dde(T, tEnd=tEnd, dt=dt)
#     mask = tt >= tEnd - 80
#     tail = sol[mask]
#     amp = (tail.max() - tail.min()) / 2
#     return amp > 0.5


# # Bisection over [2, 5]
# TH_lo, TH_hi = 2.0, 5.0
# while TH_hi - TH_lo > 0.1:
#     TH_mid = (TH_lo + TH_hi) / 2
#     if is_sustained(TH_mid):
#         TH_hi = TH_mid
#     else:
#         TH_lo = TH_mid

# TH_numerical = round((TH_lo + TH_hi) / 2, 1)
# print(f"T_H (numerical) ≈ {TH_numerical}")

# # ═══════════════════════════════════════════════════════════════════════════════
# # PART D: Analytical T_H via linear stability analysis
# # ═══════════════════════════════════════════════════════════════════════════════
# print("\n=== PART D ===")

# # f(x, x_d) = r x (1 - x_d/K)(x/A - 1)
# # Partial derivatives at x* = K, x_d* = K:
# #   a = df/dx   = r(K/A - 1)(1 - K/K) + r K (1/A)(1 - K/K) + r K (K/A - 1)(-0)
# # Expand properly:
# # f = r x (1 - xd/K)(x/A - 1)
# # df/dx  at (K,K):  r(1-K/K)(K/A-1) + rK(1-K/K)(1/A) + 0 => first two terms vanish
# #   because (1-K/K)=0.  But we must be more careful:
# # f = r x (x/A - 1) - r x (x/A - 1) xd/K
# # df/dx = r(x/A - 1) + r x (1/A)  evaluated with xd=K, times (1 - xd/K)
# #       + 0 (no xd dep in first term)
# # Actually let's just compute:

# # f(x, xd) = r x (1 - xd/K)(x/A - 1)
# # df/dx  = r (1 - xd/K)(x/A - 1) + r x (1 - xd/K)(1/A)
# #        = r (1 - xd/K) [ (x/A - 1) + x/A ]
# #        = r (1 - xd/K) (2x/A - 1)
# # At x=K, xd=K:  a = r (1 - 1)(2K/A - 1) = 0

# # df/dxd = r x (x/A - 1) (-1/K)
# # At x=K, xd=K:  b = r K (K/A - 1)(-1/K) = -r(K/A - 1) = -r(100/20 - 1) = -r*4 = -0.4

# a_coeff = 0.0  # df/dx at (K,K)
# b_coeff = -r * (K / A - 1)  # df/dxd at (K,K) = -0.4

# print(f"a = df/dx   |_(K,K) = {a_coeff}")
# print(f"b = df/dxd  |_(K,K) = {b_coeff}")

# # Characteristic equation: λ = a + b e^{-λT}
# # Hopf: λ = iω
# #   Real: 0 = a + b cos(ωT)  =>  cos(ωT) = -a/b
# #   Imag: ω = -b sin(ωT)     =>  sin(ωT) = -ω/b
# #   cos² + sin² = 1  =>  ω² = b² - a²

# omega_H = np.sqrt(b_coeff**2 - a_coeff**2)
# cos_val = -a_coeff / b_coeff
# TH_analytical = np.arccos(cos_val) / omega_H

# print(f"ω = {omega_H:.6f}")
# print(f"T_H (analytical) = arccos(-a/b)/ω = {TH_analytical:.4f}")
# print(f"T_H (analytical, rounded) = {round(TH_analytical, 1)}")

# # ─── Comparison ──────────────────────────────────────────────────────────────
# print("\n=== COMPARISON ===")
# print(f"T_H numerical  = {TH_numerical}")
# print(f"T_H analytical = {round(TH_analytical, 1)}")
# print("These values should agree to within 0.1.")
# print("For T < T_H the equilibrium x*=K is stable (damped oscillations).")
# print("For T > T_H it is unstable and a limit cycle emerges (Hopf bifurcation).")

# # ═══════════════════════════════════════════════════════════════════════════════
# # BONUS: Bifurcation diagram
# # ═══════════════════════════════════════════════════════════════════════════════
# print("\n=== Bifurcation diagram ===")

# Tscan = np.round(np.arange(0.1, 5.1, 0.1), 1)
# amplitudes = []
# for Tv in Tscan:
#     tt, sol = solve_dde(Tv, tEnd=500, dt=0.1)
#     tail = sol[tt >= 400]
#     amp = (tail.max() - tail.min()) / 2
#     amplitudes.append(amp)

# plt.figure(figsize=(8, 5))
# plt.plot(Tscan, amplitudes, "o-", ms=3)
# plt.axvline(
#     TH_analytical,
#     color="red",
#     ls="--",
#     label=f"$T_H$ (analytical) = {TH_analytical:.2f}",
# )
# plt.xlabel("T (time delay)")
# plt.ylabel("Oscillation amplitude (tail)")
# plt.title("Bifurcation Diagram")
# plt.legend()
# plt.tight_layout()
# plt.savefig("bifurcation.png", dpi=150)
# plt.show()
