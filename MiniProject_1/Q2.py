#ACCELERATION_TREND QUESTION
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, t

filename = "data/berkeley_data.csv"

# ------------------------------------------------------------
# Read the file and separate its two datasets.
# The second dataset is the global land/ocean series.
# (Same parsing as linear_trend.py, kept identical on purpose
# so both scripts stay consistent if the data file changes.)
# ------------------------------------------------------------
with open(filename, "r") as f:
    lines = f.readlines()

segments = []
current = []

for line in lines:
    stripped = line.strip()
    if stripped.startswith("%"):
        if current:
            segments.append(current)
            current = []
    elif stripped:
        current.append(line)

if current:
    segments.append(current)

raw = pd.read_csv(
    io.StringIO("".join(segments[1])),
    sep=r"\s+",
    header=None
)

monthly = raw.iloc[:, :4].copy()
monthly.columns = ["year", "month", "anomaly", "ci95"]

# ------------------------------------------------------------
# Annual means (complete years only) -- same reasoning as Q1:
# monthly values are autocorrelated, so fit trends on annual
# means rather than raw monthly data.
# ------------------------------------------------------------
annual = (
    monthly
    .groupby("year")
    .agg(anomaly=("anomaly", "mean"), n_months=("anomaly", "count"))
    .reset_index()
)
annual = annual[annual["n_months"] == 12].copy()

print(f"Available annual data: {int(annual['year'].min())}"
      f"-{int(annual['year'].max())}")


def fit_linear(annual_data, start_year, end_year):
    """OLS linear trend over [start_year, end_year], in °C/decade."""
    data = annual_data[
        (annual_data["year"] >= start_year) &
        (annual_data["year"] <= end_year)
    ].copy()

    x = data["year"].to_numpy(dtype=float)
    y = data["anomaly"].to_numpy(dtype=float)

    result = linregress(x, y)

    return {
        "period": f"{int(x.min())}-{int(x.max())}",
        "n_years": len(x),
        "trend_C_per_decade": result.slope * 10.0,
        "se_C_per_decade": result.stderr * 10.0,
        "p_value": result.pvalue,
        # Kept for plotting the fitted line later.
        "slope_C_per_year": result.slope,
        "intercept": result.intercept,
        "x_min": x.min(),
        "x_max": x.max(),
    }


# ------------------------------------------------------------
# 1) Split-window comparison: early vs. late linear trend.
#    Default split follows the assignment: 1970-1997 vs 1998-2025.
# ------------------------------------------------------------
early = fit_linear(annual, 1970, 1997)
late = fit_linear(annual, 1998, 2025)

# Difference in slopes, with combined (independent-sample) uncertainty.
slope_diff = late["trend_C_per_decade"] - early["trend_C_per_decade"]
slope_diff_se = np.sqrt(early["se_C_per_decade"]**2 + late["se_C_per_decade"]**2)

# Approximate z-score / significance of the difference.
z = slope_diff / slope_diff_se

print("\n--- Split-window linear trends ---")
for r in (early, late):
    print(
        f"{r['period']} ({r['n_years']} yrs): "
        f"{r['trend_C_per_decade']:.3f} +/- {r['se_C_per_decade']:.3f} "
        f"C/decade (p={r['p_value']:.2e})"
    )
print(
    f"\nDifference (late - early): {slope_diff:.3f} "
    f"+/- {slope_diff_se:.3f} C/decade  (z = {z:.2f})"
)

# ------------------------------------------------------------
# 2) Quadratic fit over the full record: anomaly = a + b*t + c*t^2
#    c is the acceleration term. Report it in C/decade^2 with its
#    standard error, using np.polyfit's covariance output.
# ------------------------------------------------------------
def fit_quadratic(annual_data, start_year=None, end_year=None):
    data = annual_data.copy()
    if start_year is not None:
        data = data[data["year"] >= start_year]
    if end_year is not None:
        data = data[data["year"] <= end_year]

    # Centre the time axis on its midpoint for numerical stability
    # and easier interpretation of the coefficients.
    x_raw = data["year"].to_numpy(dtype=float)
    x0 = x_raw.mean()
    x = x_raw - x0
    y = data["anomaly"].to_numpy(dtype=float)

    coeffs, cov = np.polyfit(x, y, deg=2, cov=True)
    c, b, a = coeffs  # np.polyfit returns highest power first
    c_se = np.sqrt(cov[0, 0])

    # Convert from C/year^2 to C/decade^2.
    accel = c * 100.0
    accel_se = c_se * 100.0

    n = len(x)
    dof = n - 3
    t_crit = t.ppf(0.975, dof)
    ci95 = t_crit * accel_se

    return {
        "period": f"{int(x_raw.min())}-{int(x_raw.max())}",
        "n_years": n,
        "accel_C_per_decade2": accel,
        "accel_se": accel_se,
        "accel_ci95": ci95,
        "midpoint_year": x0,
        # Kept for plotting the fitted curve later.
        "a": a, "b": b, "c": c,
        "x_min": x_raw.min(),
        "x_max": x_raw.max(),
    }


quad_full = fit_quadratic(annual)

print("\n--- Quadratic fit (full record) ---")
print(f"Period: {quad_full['period']} ({quad_full['n_years']} yrs)")
print(
    f"Acceleration term: {quad_full['accel_C_per_decade2']:.4f} "
    f"+/- {quad_full['accel_se']:.4f} C/decade^2 "
    f"(95% CI: +/- {quad_full['accel_ci95']:.4f})"
)
if quad_full["accel_C_per_decade2"] > 0:
    print("-> positive curvature: warming rate itself is increasing over time")
else:
    print("-> negative curvature: warming rate itself is decreasing over time")

# ------------------------------------------------------------
# Plot 1: split-window trend comparison, zoomed into 1970-2024
# so the two fitted lines and their divergence (or lack of it)
# are actually visible.
# ------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(11, 6))

recent = annual[annual["year"] >= 1970]
ax1.scatter(
    recent["year"], recent["anomaly"],
    s=16, color="grey", alpha=0.6, label="Annual mean anomaly"
)

for r, color in ((early, "tab:blue"), (late, "tab:red")):
    x_line = np.array([r["x_min"], r["x_max"]])
    y_line = r["intercept"] + r["slope_C_per_year"] * x_line
    ax1.plot(
        x_line, y_line, color=color, linewidth=2.5,
        label=f"{r['period']}: {r['trend_C_per_decade']:.3f}"
              f" ± {r['se_C_per_decade']:.3f} °C/decade"
    )

ax1.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax1.set_title("Split-Window Linear Trends: 1970-1997 vs. 1998-2024",
              fontsize=14, pad=12)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Temperature anomaly (°C)", fontsize=12)
ax1.grid(True, alpha=0.25)
ax1.legend(frameon=False, loc="upper left", fontsize=10)

caption1 = (
    f"Annual mean global land/ocean temperature anomalies from 1970-2024 "
    f"(grey points), with linear trends fitted separately to 1970-1997 (blue) "
    f"and 1998-2024 (red). The two slopes differ by "
    f"{slope_diff:.3f} ± {slope_diff_se:.3f} °C/decade (z = {z:.2f})."
)
fig1.text(0.5, 0.01, caption1, ha="center", va="bottom", fontsize=9, wrap=True)
fig1.tight_layout(rect=[0, 0.1, 1, 1])
fig1.savefig("q2_split_window_trends.png", dpi=300, bbox_inches="tight")

# ------------------------------------------------------------
# Plot 2: quadratic fit over the full 1850-2024 record, to show
# long-term curvature in the warming rate.
# ------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(11, 6))

ax2.scatter(
    annual["year"], annual["anomaly"],
    s=14, color="grey", alpha=0.5, label="Annual mean anomaly"
)

x_full = np.linspace(quad_full["x_min"], quad_full["x_max"], 300)
x_centered = x_full - quad_full["midpoint_year"]
y_quad = (
    quad_full["a"]
    + quad_full["b"] * x_centered
    + quad_full["c"] * x_centered**2
)
ax2.plot(
    x_full, y_quad, color="black", linewidth=2.5,
    label=f"Quadratic fit: accel = {quad_full['accel_C_per_decade2']:.4f}"
          f" ± {quad_full['accel_se']:.4f} °C/decade²"
)

ax2.axhline(0, color="black", linewidth=0.6, alpha=0.5)
ax2.set_title("Quadratic Fit to the Full Temperature Record (1850-2024)",
              fontsize=14, pad=12)
ax2.set_xlabel("Year", fontsize=12)
ax2.set_ylabel("Temperature anomaly (°C)", fontsize=12)
ax2.grid(True, alpha=0.25)
ax2.legend(frameon=False, loc="upper left", fontsize=10)

caption2 = (
    "Annual mean global land/ocean temperature anomalies, 1850-2024 (grey "
    "points), with a quadratic curve fitted to the full record (black). "
    "The positive acceleration term indicates the warming rate itself has "
    "increased over the course of the record."
)
fig2.text(0.5, 0.01, caption2, ha="center", va="bottom", fontsize=9, wrap=True)
fig2.tight_layout(rect=[0, 0.1, 1, 1])
fig2.savefig("q2_quadratic_fit.png", dpi=300, bbox_inches="tight")

plt.show()