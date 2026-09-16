import requests
import pandas as pd
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
client_id = 'fdd19ec9-0bd3-4c3f-a42a-d808b1597901'
STATION = "SN50540"  # Florida, Bergen
url = "https://frost.met.no/observations/v0.jsonld"

START_YEAR = 1991
END_YEAR = 2025

# ==============================================================
# PART A: fetch daily precipitation, 1991-2025
# ==============================================================
all_data = []
for year in range(START_YEAR, END_YEAR + 1):
    print(f"{year}: ", end="")
    params = {
        "sources": STATION,
        "elements": "sum(precipitation_amount P1D)",
        "timeresolutions": "P1D",
        "referencetime": f"{year}-01-01/{year+1}-01-01",
        "timeoffsets": "default",
        "levels": "default",
        "qualities": "0,1,2,3,4"
    }
    r = requests.get(url, params=params, auth=(client_id, ""))

    if r.status_code == 404:
        print("no data")
        continue
    if r.status_code != 200:
        print("error", r.status_code)
        print(r.text[:300])
        continue

    data = r.json()
    n = 0
    for item in data.get("data", []):
        for obs in item["observations"]:
            all_data.append({
                "time": item["referenceTime"],
                "precip": obs["value"]
            })
            n += 1

    print(f"{n:,} observations")
    sleep(0.2)

df = pd.DataFrame(all_data)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").drop_duplicates(subset="time").set_index("time")
df.to_csv(f"{STATION}_daily_precipitation.csv")

# ------------------------------------------------------------
# Diagnostic: check for rows with a bad/missing timestamp
# (these show up as NaT in the index and would otherwise silently
# turn the year/month columns into floats, breaking later indexing)
# ------------------------------------------------------------
print("Rows with missing/invalid time:", df.index.isna().sum())
if df.index.isna().sum() > 0:
    print(df[df.index.isna()])
df = df[~df.index.isna()]

df["year"] = df.index.year
df["month"] = df.index.month

# ==============================================================
# PART B: monthly climate profile - which months are wettest/driest
# ==============================================================
monthly_totals = df.groupby(["year", "month"])["precip"].sum().reset_index()
monthly_totals["year"] = monthly_totals["year"].astype(int)
monthly_totals["month"] = monthly_totals["month"].astype(int)

month_mean = np.zeros(12)
month_std = np.zeros(12)

for m in range(1, 13):
    vals = monthly_totals[monthly_totals["month"] == m]["precip"]
    month_mean[m - 1] = np.mean(vals)
    month_std[m - 1] = np.std(vals)

months = np.arange(1, 13)

# ------------------------------------------------------------
# Find individual year-month values that lie more than 2 standard
# deviations from that month's mean (i.e. unusually far from normal)
# ------------------------------------------------------------
outlier_months = []
outlier_values = []
outlier_years = []

for idx, row in monthly_totals.iterrows():
    m = int(row["month"])
    val = row["precip"]
    mean = month_mean[m - 1]
    std = month_std[m - 1]
    if abs(val - mean) > 2 * std:
        outlier_months.append(m)
        outlier_values.append(val)
        outlier_years.append(int(row["year"]))

for m, v, y in zip(outlier_months, outlier_values, outlier_years):
    print(f"{y}-{m:02d}: {v:.0f} mm (mean {month_mean[m-1]:.0f} \u00b1 {month_std[m-1]:.0f})")

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(months, month_mean, yerr=month_std, capsize=4, color="steelblue", label="Mean \u00b1 std")
ax.scatter(outlier_months, outlier_values, color="firebrick", s=15, zorder=3, label="> 2\u03c3 from normal")

ax.set_xticks(months)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly precipitation [mm]")
ax.set_title(f"({STATION}) Mean monthly precipitation with outliers, {START_YEAR}\u2013{END_YEAR}")
ax.legend()
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
fig.savefig("task2_monthly_climatology.png", dpi=300)
plt.show()

# ==============================================================
# PART C: has precipitation increased over time?
# ==============================================================
annual_totals = df.groupby("year")["precip"].sum().reset_index()

result = linregress(annual_totals["year"], annual_totals["precip"])
print(f"Trend: {result.slope:.2f} mm/year, p = {result.pvalue:.3g}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(annual_totals["year"], annual_totals["precip"], color="steelblue")
ax.plot(
    annual_totals["year"],
    result.intercept + result.slope * annual_totals["year"],
    color="firebrick",
    label=f"trend: {result.slope:.2f} mm/yr"
)
ax.set_xlabel("Year")
ax.set_ylabel("Annual total precipitation [mm]")
ax.set_title(f"({STATION}) Annual precipitation total, {START_YEAR}\u2013{END_YEAR}")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("task2_annual_trend.png", dpi=300)
plt.show()