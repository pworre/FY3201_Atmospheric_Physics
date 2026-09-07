import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Q4: The chocolate test.
# Correlate global temperature with the number of Norwegian
# newborns given the first name "Live" -- a steadily growing
# quantity with no physical link to climate.
# Source: Statistics Norway (SSB), table 10467.
# URL: https://www.ssb.no/statbank/table/10467
# ------------------------------------------------------------

# ---- 1) Read Berkeley Earth temperature data (same parsing as before) ----
filename = "data/berkeley_data.csv"

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

raw = pd.read_csv(io.StringIO("".join(segments[1])), sep=r"\s+", header=None)
monthly = raw.iloc[:, :4].copy()
monthly.columns = ["year", "month", "anomaly", "ci95"]

annual_temp = (
    monthly.groupby("year")
    .agg(anomaly=("anomaly", "mean"), n_months=("anomaly", "count"))
    .reset_index()
)
annual_temp = annual_temp[annual_temp["n_months"] == 12][["year", "anomaly"]]

# ---- 2) Read the SSB name data ----
# Columns: "girls' or boys' name", "year", "Born persons"
# ".." = no data available, "." = suppressed (fewer than 4 people, privacy rule)
name_file = "data/livenavn.csv"

names = pd.read_csv(name_file, sep="\t", quotechar='"')
names.columns = ["name", "year", "count"]

# Convert year and count to numeric, turning ".." and "." into NaN.
names["year"] = pd.to_numeric(names["year"], errors="coerce")
names["count"] = pd.to_numeric(names["count"], errors="coerce")
names = names.dropna(subset=["year", "count"])
names["year"] = names["year"].astype(int)

# ---- 3) Harmonize: merge on the common years ----
merged = pd.merge(annual_temp, names[["year", "count"]], on="year", how="inner").dropna()

print(f"Common period: {int(merged['year'].min())}-{int(merged['year'].max())} "
      f"({len(merged)} years)")

# ---- 4) Correlation ----
corr = merged["anomaly"].corr(merged["count"])
print(f"\nPearson correlation (temperature vs. number of newborns named Live): "
      f"r = {corr:.4f}")

# ---- 5) Plot: dual-axis time series + scatter ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1b = ax1.twinx()
l1, = ax1.plot(merged["year"], merged["anomaly"], color="tab:red",
               linewidth=2, label="Temperature anomaly (°C)")
l2, = ax1b.plot(merged["year"], merged["count"], color="tab:purple",
                linewidth=2, label='Newborns named "Live"')
ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature anomaly (°C)", color="tab:red")
ax1b.set_ylabel('Number of newborns named "Live"', color="tab:purple")
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1b.tick_params(axis="y", labelcolor="tab:purple")
ax1.set_title("Two Unrelated Rising Series Over Time")
ax1.legend(handles=[l1, l2], loc="upper left", frameon=False, fontsize=9)

ax2.scatter(merged["count"], merged["anomaly"], color="tab:purple", alpha=0.6)
ax2.set_xlabel('Number of newborns named "Live"')
ax2.set_ylabel("Temperature anomaly (°C)")
ax2.set_title(f"r = {corr:.3f}")

caption = (
    f"Global temperature anomaly vs. number of Norwegian newborns named "
    f'"Live", {int(merged["year"].min())}-{int(merged["year"].max())} '
    f"(Statistics Norway, table 10467). Despite no physical mechanism "
    f"linking Norwegian naming trends to global climate, the two series "
    f"are strongly correlated (r = {corr:.3f}) simply because both have "
    f"risen over the same period."
)
fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9, wrap=True)
fig.tight_layout(rect=[0, 0.09, 1, 1])
fig.savefig("q4_chocolate_test.png", dpi=300, bbox_inches="tight")
plt.show()