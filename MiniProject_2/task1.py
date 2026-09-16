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
url = "https://frost.met.no/observations/v0.jsonld"

# ==============================================================
# PART A: fetch the 1991-2020 climate normal period
# ==============================================================
NORMAL_START = 1991
NORMAL_END = 2020

all_data = []
for year in range(NORMAL_START, NORMAL_END + 1):
    print(f"{year}: ", end="")
    params = {
        "sources": STATION,
        "elements": "max(air_temperature P1D),min(air_temperature P1D)",
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
        row = {"time": item["referenceTime"]}
        for obs in item["observations"]:
            if obs["elementId"] == "max(air_temperature P1D)":
                row["tmax"] = obs["value"]
            elif obs["elementId"] == "min(air_temperature P1D)":
                row["tmin"] = obs["value"]
        all_data.append(row)
        n += 1

    print(f"{n:,} observations")
    sleep(0.2)

normal_df = pd.DataFrame(all_data)
normal_df["time"] = pd.to_datetime(normal_df["time"], utc=True)
normal_df = (
    normal_df
    .sort_values("time")
    .drop_duplicates(subset="time")
    .set_index("time")
)
normal_df["daynumber"] = normal_df.index.dayofyear
normal_df.to_csv(f"{STATION}_normal_1991_2020.csv")

# ==============================================================
# PART B: fetch 2025 (the year we compare against the normal)
# ==============================================================
YEAR_START = 2025
YEAR_END = 2025

all_data = []
for year in range(YEAR_START, YEAR_END + 1):
    print(f"{year}: ", end="")
    params = {
        "sources": STATION,
        "elements": "max(air_temperature P1D),min(air_temperature P1D)",
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
        row = {"time": item["referenceTime"]}
        for obs in item["observations"]:
            if obs["elementId"] == "max(air_temperature P1D)":
                row["tmax"] = obs["value"]
            elif obs["elementId"] == "min(air_temperature P1D)":
                row["tmin"] = obs["value"]
        all_data.append(row)
        n += 1

    print(f"{n:,} observations")
    sleep(0.2)

year_df = pd.DataFrame(all_data)
year_df["time"] = pd.to_datetime(year_df["time"], utc=True)
year_df = (
    year_df
    .sort_values("time")
    .drop_duplicates(subset="time")
    .set_index("time")
)
year_df["daynumber"] = year_df.index.dayofyear
year_df.to_csv(f"{STATION}_2025_daily_maxmin.csv")

# ==============================================================
# PART C: day-of-year climatology from the 1991-2020 normal
# (same style as your Appendix A stats loop, just run twice:
#  once for tmax, once for tmin)
# ==============================================================
tmax_mean = np.zeros(365)
tmax_min = np.zeros(365)
tmax_max = np.zeros(365)

tmin_mean = np.zeros(365)
tmin_min = np.zeros(365)
tmin_max = np.zeros(365)

daynumber = np.zeros(365)

for d in range(365):
    mask = (normal_df["daynumber"] >= d + 1) & (normal_df["daynumber"] < d + 2)

    tmax_mean[d] = np.mean(normal_df[mask]["tmax"])
    tmax_min[d] = np.min(normal_df[mask]["tmax"])
    tmax_max[d] = np.max(normal_df[mask]["tmax"])

    tmin_mean[d] = np.mean(normal_df[mask]["tmin"])
    tmin_min[d] = np.min(normal_df[mask]["tmin"])
    tmin_max[d] = np.max(normal_df[mask]["tmin"])

    daynumber[d] = d + 1

# ==============================================================
# PART D: plot 2025 against the 1991-2020 normal
# ==============================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Normal period: shaded range + mean line
ax.fill_between(daynumber, tmax_min, tmax_max, color="firebrick", alpha=0.12, label="Max range 1991–2020")
ax.plot(daynumber, tmax_mean, color="firebrick", linewidth=1, alpha=0.6, label="Mean max 1991–2020")

ax.fill_between(daynumber, tmin_min, tmin_max, color="steelblue", alpha=0.12, label="Min range 1991–2020")
ax.plot(daynumber, tmin_mean, color="steelblue", linewidth=1, alpha=0.6, label="Mean min 1991–2020")

# 2025: scatter on top
ax.scatter(year_df["daynumber"], year_df["tmax"], s=10, color="firebrick", label="2025 max")
ax.scatter(year_df["daynumber"], year_df["tmin"], s=10, color="steelblue", label="2025 min")

ax.set_xlabel("Day number")
ax.set_ylabel("Temperature [°C]")
ax.set_xlim(1, 366)
ax.set_title(f"({STATION}) 2025 daily max/min vs. 1991–2020 climate normal")
ax.legend(ncol=2, fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
plt.show()