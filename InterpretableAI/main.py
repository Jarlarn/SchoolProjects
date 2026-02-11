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


num_bins = int(np.sqrt(len(training_data)))
plot_histogram(training_data, "Training Data Histogram", num_bins, "blue")
plot_histogram(validation_data, "Training Data Histogram", num_bins, "green")
plot_histogram(test_data, "Training Data Histogram", num_bins, "red")

plt.legend(["Training", "Validation", "Test"])
plt.tight_layout()
plt.savefig("Training_histogram")


def arma_loss(params, data):
    # Negative log-likelihood loss function
    phi, theta, mu, sigma = params
    n = len(data)
    eps = np.zeros(n)
    y_hat = np.zeros(n)
    y_hat[0] = mu
    for t in range(1, n):
        y_hat[t] = mu + phi * data[t - 1] + theta * eps[t - 1]
        eps[t] = data[t] - y_hat[t]

    return np.sum(np.log(sigma) + (eps**2) / (2 * sigma**2))


def arma_predict(params, data, steps=1):
    phi, theta, mu, sigma = params
    n = len(data)
    eps = np.zeros(n + steps)
    y_hat = np.zeros(n + steps)
    y_hat[:n] = data
    for t in range(n, n + steps):
        y_hat[t] = mu + phi * y_hat[t - 1] + theta * eps[t - 1]
        # eps[t] can be set to 0 for forecasting
    return y_hat[n:]


# Initial parameter guess: [phi, theta, mu, sigma]
init_params = [0.5, 0.5, np.mean(training_data), np.std(training_data)]
result = minimize(
    arma_loss, init_params, args=(training_data,), method="Nelder-Mead", tol=1e-4
)
fitted_params = result.x

test_prediction = arma_predict(fitted_params, test_data, steps=len(test_data))

rmse = np.sqrt(np.mean((test_data - test_prediction) ** 2))

print(rmse)
print(np.std(test_data))
