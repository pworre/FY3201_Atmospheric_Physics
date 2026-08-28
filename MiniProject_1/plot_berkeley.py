import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ------------------------------------------------------------
# Berkeley Earth global temperature anomaly
#
# The supplied file contains whitespace-separated numeric data.
# It consists of two consecutive monthly datasets; the second
# dataset is the global land/ocean temperature anomaly series.
#
# Columns used:
#   0 = year
#   1 = month
#   2 = temperature anomaly (°C)
#   3 = quoted uncertainty (95% confidence interval half-width)
# ------------------------------------------------------------

filename = "berkeley_data.csv"

# Read whitespace-separated data (the file has no usable CSV header).
raw = pd.read_csv(
    filename,
    sep=r"\s+",
    header=None,
    comment="%",
    engine="python"
)

# The year sequence restarts at the boundary between the two datasets.
# Find that restart and retain the second dataset (land/ocean).
restart = np.where(np.diff(raw.iloc[:, 0].to_numpy()) < 0)[0]

if len(restart) == 0:
    raise ValueError("Could not identify the start of the land/ocean dataset.")

start = restart[0] + 1
df = raw.iloc[start:, :4].copy()

df.columns = ["year", "month", "anomaly", "ci95"]

# Create a monthly datetime axis.
df["date"] = pd.to_datetime(
    dict(year=df["year"], month=df["month"], day=1)
)

# Remove any rows without the required measurements.
df = df.dropna(subset=["anomaly", "ci95"]).sort_values("date")

# A centred 12-month moving average.
df["moving_average"] = (
    df["anomaly"]
    .rolling(window=12, center=True, min_periods=12)
    .mean()
)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

# Monthly anomalies.
ax.plot(
    df["date"],
    df["anomaly"],
    linewidth=0.8,
    alpha=0.65,
    label="Monthly anomaly"
)

# The file quotes 95% confidence intervals, so the shaded band
# is labelled explicitly as a 95% confidence interval.
ax.fill_between(
    df["date"],
    df["anomaly"] - df["ci95"],
    df["anomaly"] + df["ci95"],
    alpha=0.20,
    label="95% confidence interval"
)

# 12-month moving average.
ax.plot(
    df["date"],
    df["moving_average"],
    linewidth=2.2,
    label="12-month moving average"
)

# Reference line for zero anomaly.
ax.axhline(0, color="black", linewidth=0.8, alpha=0.7)

# Labels and formatting.
ax.set_title(
    "Berkeley Earth Monthly Global Land/Ocean Temperature Anomaly",
    fontsize=15,
    pad=12
)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Temperature anomaly (°C)", fontsize=12)

# Readable date ticks.
ax.xaxis.set_major_locator(mdates.YearLocator(20))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_minor_locator(mdates.YearLocator(5))

ax.grid(True, which="major", alpha=0.25)
ax.legend(frameon=False, loc="upper left")

# Caption that allows the figure to stand on its own.
caption = (
    "Monthly global land/ocean surface temperature anomalies from the "
    "Berkeley Earth dataset. The anomaly is expressed in degrees Celsius "
    "relative to the dataset's reference climatology. The shaded region "
    "shows the quoted 95% confidence interval, and the bold line is a "
    "centred 12-month moving average that highlights the longer-term trend."
)

fig.text(
    0.5, 0.01,
    caption,
    ha="center",
    va="bottom",
    fontsize=9,
    wrap=True
)

fig.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()

# Optional: save a publication-quality image.
# fig.savefig("berkeley_global_land_ocean_anomaly.png",
#             dpi=300, bbox_inches="tight")