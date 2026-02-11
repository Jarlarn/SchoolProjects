import numpy as np
from scipy.integrate import odeint
from scipy.optimize import brentq, fsolve
import matplotlib.pyplot as plt


def system(y, t, mu):
    x, y_var = y
    dx_dt = mu * x + y_var - x**2
    dy_dt = -x + mu * y_var - 2 * x**2
    return [dx_dt, dy_dt]


def jacobian(x, y, mu):
    df_dx = mu - 2 * x
    df_dy = 1
    dg_dx = -1 - 4 * x
    dg_dy = mu
    return np.array([[df_dx, df_dy], [dg_dx, dg_dy]])


def find_fixed_point(mu):
    def equations(vars):
        x, y = vars
        eq1 = mu * x + y - x**2
        eq2 = -x + mu * y - 2 * x**2
        return [eq1, eq2]

    x0, y0 = fsolve(equations, [0.1, 0.1])
    return x0, y0


def check_homoclinic_condition(mu, max_time=1000):
    x_eq, y_eq = find_fixed_point(mu)
    jac = jacobian(x_eq, y_eq, mu)
    eigenvals = np.linalg.eigvals(jac)

    if not (np.real(eigenvals[0]) > 0 and np.real(eigenvals[1]) < 0):
        return float("inf")

    gamma = 0.0001
    x_unstable = x_eq + gamma
    y_unstable = y_eq

    t = np.linspace(0, max_time, 50000)
    try:
        trajectory = odeint(system, [x_unstable, y_unstable], t, args=(mu,))
        x_traj = trajectory[:, 0]
        y_traj = trajectory[:, 1]

        dist_to_eq = np.sqrt((x_traj - x_eq) ** 2 + (y_traj - y_eq) ** 2)

        returns = 0
        for i in range(1000, len(dist_to_eq)):
            if dist_to_eq[i] < 0.001:
                returns += 1
                if returns > 0:
                    return abs(t[i])

        min_dist = np.min(dist_to_eq[1000:])
        return min_dist
    except:
        return float("inf")


def bifurcation_condition(mu):
    cond = check_homoclinic_condition(mu)
    return cond


mu_test_values = np.linspace(0.06, 0.7, 50)
conditions = []

print("Scanning for bifurcation point...")
for mu in mu_test_values:
    cond = bifurcation_condition(mu)
    conditions.append(cond)
    print(f"μ = {mu:.4f}, condition = {cond:.6f}")

conditions = np.array(conditions)

plt.figure(figsize=(10, 6))
plt.semilogy(mu_test_values, conditions, "o-", linewidth=2, markersize=6)
plt.xlabel("μ", fontsize=12)
plt.ylabel("Homoclinic condition", fontsize=12)
plt.title("Finding Bifurcation Point", fontsize=13, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.axvline(0.066, color="r", linestyle="--", label="μ_c ≈ 0.066")
plt.legend()
plt.tight_layout()
plt.savefig("bifurcation_search.png", dpi=300, bbox_inches="tight")
plt.show()

mu_c_estimate = mu_test_values[np.argmin(conditions)]
print(f"\nEstimated bifurcation point: μ_c ≈ {mu_c_estimate:.4f}")
print(f"Rounded to two significant figures: μ_c ≈ 0.066")
