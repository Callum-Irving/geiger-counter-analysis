import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scienceplots

plt.style.use(["science", "notebook", "no-latex", "grid"])
# plt.style.use("plotstyle.mplstyle")

# Uncertainty in distance from source to tube in meters
DISTANCE_UNCERTAINTY = 1e-3

# Read in data
df_files = [
    "./data/alpha_source_5mm.csv",
    "./data/alpha_source_10mm.csv",
    "./data/alpha_source_15mm.csv",
    "./data/alpha_source_20mm.csv",
    "./data/alpha_source_25mm.csv",
    "./data/alpha_source_30mm.csv",
    "./data/alpha_source_35mm.csv",
    "./data/alpha_source_40mm.csv",
    "./data/alpha_source_45mm.csv",
]
dfs = [pl.read_csv(file) for file in df_files]

# Generate datapoints
xs = np.arange(5e-3, 50e-3, 5e-3, dtype=float)
xs_uncs = np.ones(len(xs)) * DISTANCE_UNCERTAINTY
ys = np.array([df["CPM"].mean() for df in dfs])
ys_uncs = np.array([df["CPM"].std(ddof=1) for df in dfs])


# Function to fit
def exp_decay(x, yoff, amp, coeff):
    """Exponential decay function yoff + amp e^(-coeff x)."""
    return yoff + amp * np.exp(-coeff * x)


def exp_decay_der(x, _yoff, amp, coeff):
    """Derivative of exponential decay function."""
    return -coeff * amp * np.exp(-coeff * x)


fit_params, cov = curve_fit(exp_decay, xs, ys)
errs = np.sqrt(np.diag(cov))

N_POINTS = 200
xs_fit = np.linspace(0.1e-3, 50e-3, N_POINTS)
ys_fit = exp_decay(xs_fit, *fit_params)
yerrs_fit = exp_decay_der(xs_fit, *fit_params) * DISTANCE_UNCERTAINTY

# Plot background line
df_background = pl.read_csv("./data/no_source_20240621.csv")
background_cpm = df_background["CPM"].to_numpy().mean()  # to_numpy eliminate lsp warning
background_stdev = df_background["CPM"].std(ddof=1)

plt.errorbar(
    xs, ys, xerr=xs_uncs, yerr=ys_uncs, ms=4, capsize=3, fmt="o", c="k", label="Data"
)

plt.plot(xs_fit, ys_fit, c="r", label="Exponential decay fit")
plt.fill_between(
    xs_fit, ys_fit - yerrs_fit, ys_fit + yerrs_fit, color="r", alpha=0.2, edgecolor=None
)

plt.plot(
    xs_fit, np.ones(N_POINTS) * background_cpm, "--", c="b", label="Measured background"
)
plt.plot(xs_fit, np.ones(N_POINTS) * fit_params[0], "--", c="g", label="Fit background")

plt.xlabel("Distance from Source to Geiger-Muller Tube [m]")
plt.ylabel("Radioactivity [counts per minute]")
plt.title("Observed Radioactivity from Alpha Source at Various Distances")
plt.legend()
plt.savefig("alpha_source_distance.png", dpi=400)
plt.show()

print("Measured background:", background_cpm, "+/-", background_stdev)
print()
print("Fit parameters:")
print("yoff (background):", fit_params[0], "+/-", errs[0])
print("I0:", fit_params[1], "+/-", errs[1])
print("alpha:", fit_params[2], "+/-", errs[2])
print()
print("Penetration depth:", 1 / fit_params[2])

# NOTE: Should distances be offset so that they are measured from the center of the tube?

# Find speed of alpha particles emitted from source
# Use speed and distances to find times
# Exponential coefficient should be half-life?
