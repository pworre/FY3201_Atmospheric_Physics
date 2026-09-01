#ACCELERATION_TREND QUESTION

import io
import numpy as np
import pandas as pd
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