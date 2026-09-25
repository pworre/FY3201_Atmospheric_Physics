import requests
import pandas as pd
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
client_id = 'fdd19ec9-0bd3-4c3f-a42a-d808b1597901'
STATION = "SN50540"  # Florida, Bergen
START_YEAR = 1991
END_YEAR = 2025
url = "https://frost.met.no/observations/v0.jsonld"

all_data = []
for year in range(START_YEAR, END_YEAR + 1):
    print(f"{year}: ", end="")
    params = {
        "sources": STATION,
        "elements": "mean(air_pressure_at_sea_level P1D)",
        "referencetime": f"{year}-01-01/{year+1}-01-01",
        "timeresolutions": "P1D",
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
                "pressure": obs["value"]
            })
            n += 1
    print(f"{n:,} observations")
    sleep(0.2)

# ------------------------------------------------
# Combine all observations
# ------------------------------------------------
df = pd.DataFrame(all_data)
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df[~df["time"].isna()]
df = (
    df
    .sort_values("time")
    .drop_duplicates(subset="time")
    .set_index("time")
)

df.to_csv(f"{STATION}_daily_pressure.csv")

df["daynumber"] = df.index.dayofyear
df["year"] = df.index.year

# ------------------------------------------------
# Day-of-year climatology (same loop style as Appendix A)
# ------------------------------------------------
pressure_mean = np.zeros(365)
pressure_std = np.zeros(365)
daynumber = np.zeros(365)

for d in range(365):
    vals = df[(df["daynumber"] >= d + 1) & (df["daynumber"] < d + 2)]["pressure"]
    pressure_mean[d] = np.mean(vals)
    pressure_std[d] = np.std(vals)
    daynumber[d] = d + 1

# ------------------------------------------------
# Figure 1: variation in pressure through the year
# ------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(daynumber, pressure_mean, s=8, color="gray", label="Daily mean pressure")
ax.errorbar(daynumber, pressure_mean, yerr=pressure_std, fmt='none', ecolor="steelblue", alpha=0.3, capsize=0)
ax.set_xlabel("Day number")
ax.set_ylabel("Mean sea level pressure [hPa]")
ax.set_xlim(1, 366)
ax.set_title(f"({STATION}) daily mean pressure with day-to-day variation, {START_YEAR}\u2013{END_YEAR}")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("task3a_pressure_climatology.png", dpi=300)
plt.show()

# ------------------------------------------------
# Figure 2: has pressure changed over time?
# ------------------------------------------------
annual_mean = df.groupby("year")["pressure"].mean()
years = annual_mean.index.to_numpy()
values = annual_mean.to_numpy()

# simple linear trend (no scipy needed)
slope, intercept = np.polyfit(years, values, 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(years, values, color="steelblue")
ax.plot(years, intercept + slope * years, color="firebrick", label=f"trend: {slope:.3f} hPa/yr")
ax.set_xlabel("Year")
ax.set_ylabel("Annual mean pressure [hPa]")
ax.set_title(f"({STATION}) annual mean sea level pressure, {START_YEAR}\u2013{END_YEAR}")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("task3a_pressure_trend.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# Figure 3: daily pressure through January, 2025 vs. the
# 1991-2024 normal
# ------------------------------------------------------------
df["month"] = df.index.month
df["day"] = df.index.day

jan_normal = df[(df["month"] == 1) & (df["year"] < 2025)]
jan_2025 = df[(df["month"] == 1) & (df["year"] == 2025)]

jan_mean = np.zeros(31)
jan_min = np.zeros(31)
jan_max = np.zeros(31)
jan_days = np.zeros(31)

for d in range(1, 32):
    vals = jan_normal[jan_normal["day"] == d]["pressure"]
    jan_mean[d - 1] = np.mean(vals)
    jan_min[d - 1] = np.min(vals)
    jan_max[d - 1] = np.max(vals)
    jan_days[d - 1] = d

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(jan_days, jan_min, jan_max, color="steelblue", alpha=0.15, label="Range 1991\u20132024")
ax.plot(jan_days, jan_mean, color="steelblue", linewidth=1.5, label="Mean 1991\u20132024")

ax.plot(jan_2025["day"], jan_2025["pressure"], color="firebrick", marker="o", markersize=4, linewidth=1.5, label="2025")

ax.set_xlabel("Day of January")
ax.set_ylabel("Mean sea level pressure [hPa]")
ax.set_xlim(1, 31)
ax.set_title(f"({STATION}) daily mean pressure in January, 2025 vs. 1991\u20132024 normal")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("task3a_january_2025.png", dpi=300)
plt.show()