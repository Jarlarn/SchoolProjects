import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

training_data = np.loadtxt("A1_training.txt")
validation_data = np.loadtxt("A1_validation.txt")
test_data = np.loadtxt("A1_test.txt")


def plot_histogram(data, title, bins, color):
    plt.hist(data, bins=bins, edgecolor="black", color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")


# num_bins = int(np.sqrt(len(training_data)))
# plot_histogram(training_data, "Training Data Histogram", num_bins, "blue")
# plot_histogram(validation_data, "Training Data Histogram", num_bins, "green")
# plot_histogram(test_data, "Training Data Histogram", num_bins, "red")

# plt.legend(["Training", "Validation", "Test"])
# plt.tight_layout()
# plt.savefig("Training_histogram")


def ar_model(phi, c, data):
    p = len(phi)
    n = len(data)
    y_hat = np.zeros(n)
    for t in range(p):
        y_hat[t] = c
    for t in range(p, n):
        ar_part = 0.0
        for j in range(p):
            ar_part += phi[j] * data[t - j - 1]
        y_hat[t] = c + ar_part
    return y_hat


def ma_model(theta, mu, data):
    q = len(theta)
    n = len(data)
    eps = np.zeros(n)
    y_hat = np.zeros(n)
    for t in range(n):
        ma_part = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                ma_part += theta[j - 1] * eps[t - j]
        y_hat[t] = mu + eps[t] + ma_part
        eps[t] = data[t] - (mu + ma_part)
    return y_hat


def arma_model(phi, theta, mu, data):
    p = len(phi)
    q = len(theta)
    n = len(data)
    eps = np.zeros(n)
    y_hat = np.zeros(n)
    for t in range(max(p, q)):
        y_hat[t] = mu
    for t in range(max(p, q), n):
        ar_part = 0.0
        for i in range(p):
            ar_part += phi[i] * data[t - i - 1]
        ma_part = 0.0
        for j in range(q):
            ma_part += theta[j] * eps[t - j - 1]
        y_hat[t] = mu + ar_part + ma_part
        eps[t] = data[t] - y_hat[t]
    return y_hat, eps


def arma_loss(params, data, p, q, mu, sigma):
    phi = params[:p]
    theta = params[p : p + q]
    y_hat, eps = arma_model(phi, theta, mu, data)
    n = len(data)
    loss = 0.0
    for t in range(n):
        loss += np.log(sigma) + (eps[t] ** 2) / (2 * sigma**2)
    return loss


def arma_predict(params, data, p, q, steps=1, mu=None):
    phi = params[:p]
    theta = params[p : p + q]
    if mu is None:
        mu = np.mean(data)
    n = len(data)
    eps = np.zeros(n + steps)
    y_hat = np.zeros(n + steps)
    for t in range(n):
        y_hat[t] = data[t]
    for t in range(n, n + steps):
        ar_part = 0.0
        for i in range(p):
            ar_part += phi[i] * y_hat[t - i - 1]
        ma_part = 0.0
        for j in range(q):
            ma_part += theta[j] * eps[t - j - 1]
        y_hat[t] = mu + ar_part + ma_part
        # eps[t] = 0 for forecasting
    return y_hat[n:]


def acf(x, nlags):
    x = list(x)
    n = len(x)
    mean = sum(x) / n
    acf_vals = []
    for k in range(nlags + 1):
        numerator = 0.0
        denominator = 0.0
        for t in range(k, n):
            numerator += (x[t] - mean) * (x[t - k] - mean)
        for t in range(n):
            denominator += (x[t] - mean) ** 2
        acf_vals.append(numerator / denominator)
    return acf_vals


def pacf(x, lags):
    return


def main():
    nlags = 20
    acf_vals = acf(training_data, nlags)
    plt.stem(range(nlags + 1), acf_vals)
    plt.title("ACF")


p, q = 3, 2
init_params = [0.5] * p + [0.5] * q
mu = np.mean(training_data)
sigma = np.std(training_data)
# Minimize
result = minimize(
    arma_loss,
    init_params,
    args=(training_data, p, q, mu, sigma),
    method="Nelder-Mead",
    tol=1e-4,
)
best_phi = result.x[:p]
best_theta = result.x[p : p + q]
print("Best phi:", best_phi)
print("Best theta:", best_theta)


fitted_params = result.x

test_prediction = arma_predict(
    fitted_params, test_data, p, q, steps=len(test_data), mu=mu
)

rmse = np.sqrt(np.mean((test_data - test_prediction) ** 2))

print(f"best rmse: {rmse}")
print(np.std(test_data))


main()
