import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter1d

DATA_FILE = "overnight_20240621.csv"


def poisson(mu, r):
    """Return Poisson distribution with mean mu evaluated at r."""
    return 1 / math.factorial(r) * mu**r * np.exp(-mu)


def gaussian(mu, std, x):
    """Return Gaussian distribution with mean mu and standard deviation std
    evaluated at x."""
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std) ** 2)


df = pd.read_csv(DATA_FILE, parse_dates=["Timestamp"])
cpms = df["CPM"].to_numpy()

# Print some stats of dataset
mean = cpms.mean()
var = cpms.var()
std = cpms.std()
print("Mean:", mean)
print("Variance:", var)
print("Stdev:", std)

# Histogram data (normalized)
min_cpm = int(cpms.min())
max_cpm = int(cpms.max())
bins = np.arange(min_cpm, max_cpm + 1) - 0.5
values, _, _ = plt.hist(
    cpms, bins=bins, histtype="step", density=True, label="recorded data"
)

# Plot Poisson distribution based on mean
probs_poisson = [poisson(mean, int(x)) for x in bins[:-1] + 0.5]
plt.stairs(probs_poisson, edges=bins, label="theoretical Poisson")

# Plot Gaussian distribution based on mean and standard deviation
probs_gaussian = gaussian(mean, std, bins[:-1])
plt.stairs(probs_gaussian, edges=bins, label="theoretical Gaussian")

plt.title("Counts Per Minute (CPM) Sampled at 60 Second Intervals")
plt.xlabel("CPM")
plt.ylabel("Probability")
plt.legend()

# Plot time series CPMs
plt.figure()
plt.plot(df["Timestamp"], cpms, label="raw data")

# Apply Gaussian filter
smoothed_cpms = gaussian_filter1d(cpms, sigma=8)
plt.plot(df["Timestamp"], smoothed_cpms, label="smoothed with $\\sigma = 8$")

plt.title("CPM over Time")
plt.xlabel("Timestamp")
plt.ylabel("CPM")
plt.legend()

plt.show()