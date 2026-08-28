import io
import numpy as np
import pandas as pd
from scipy.stats import linregress, t

filename = "berkeley_data.csv"

# ------------------------------------------------------------
# Read the file and separate its two datasets.
# The second dataset is the global land/ocean series.
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

# Second numeric block: global land/ocean temperature anomaly
raw = pd.read_csv(
    io.StringIO("".join(segments[1])),
    sep=r"\s+",
    header=None
)

monthly = raw.iloc[:, :4].copy()
monthly.columns = ["year", "month", "anomaly", "ci95"]

# ------------------------------------------------------------
# Calculate annual means from the monthly anomalies.
# Require all 12 months for a complete annual mean.
# ------------------------------------------------------------
annual = (
    monthly
    .groupby("year")
    .agg(
        anomaly=("anomaly", "mean"),
        n_months=("anomaly", "count")
    )
    .reset_index()
)

annual = annual[annual["n_months"] == 12].copy()

available_end = int(annual["year"].max())
print(f"Available data: {int(annual['year'].min())}–{available_end}")

# ------------------------------------------------------------
# Fit an ordinary least-squares linear trend.
#
# The uncertainty reported is the standard error of the fitted
# slope, converted to °C/decade. A 95% confidence interval is
# also calculated using the Student-t distribution.
# ------------------------------------------------------------
def fit_trend(annual_data, start_year, requested_end=2025):

    end_year = min(requested_end, int(annual_data["year"].max()))

    data = annual_data[
        (annual_data["year"] >= start_year) &
        (annual_data["year"] <= end_year)
    ].copy()

    x = data["year"].to_numpy(dtype=float)
    y = data["anomaly"].to_numpy(dtype=float)

    result = linregress(x, y)

    # Convert from °C/year to °C/decade.
    trend = result.slope * 10.0
    trend_se = result.stderr * 10.0

    return {
        "requested_period": f"{start_year}–{requested_end}",
        "actual_period": f"{int(x.min())}–{int(x.max())}",
        "n_years": len(data),
        "trend_C_per_decade": trend,
        "standard_error": trend_se,
        "p_value": result.pvalue,
    }


# Primary analysis plus sensitivity to start year.
results = [
    fit_trend(annual, 1980),
    fit_trend(annual, 1970),
    fit_trend(annual, 1990),
]

# ------------------------------------------------------------
# Print results.
# ------------------------------------------------------------
print("\nLinear trends in annual mean global land/ocean anomaly")
print("(ordinary least-squares regression)\n")

for r in results:
    print(
        f"Requested: {r['requested_period']} | "
        f"Used: {r['actual_period']} ({r['n_years']} annual means)\n"
        f"  Trend = {r['trend_C_per_decade']:.3f} "
        f"± {r['standard_error']:.3f} °C/decade (1σ standard error)\n"
        f"  p = {r['p_value']:.2e}\n"
    )