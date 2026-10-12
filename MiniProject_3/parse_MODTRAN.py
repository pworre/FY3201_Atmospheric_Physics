#parse_MODTRAN.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from solar_spectrum import (
    planck_irradiance_at_earth,
    SOLAR_RADIUS_M,
    ASTRONOMICAL_UNIT_M,
)

def is_header_line(line: str) -> bool:
    return "RADIANCE(WATTS/CM2-STER" in line

def is_block_terminator(line: str) -> bool:
    return line.strip().startswith("iv")

float_re = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")

def try_parse_data_line(line: str):
    nums = float_re.findall(line)
    if len(nums) == 12:
        return list(map(lambda x: float(x.replace("D","E").replace("d","e")), nums))
    return None


def read_file(spath):
    text = spath.read_text(errors="ignore").splitlines()
    rows = []
    in_block = False
    skip_next = 0
    for ln in text:
        if is_header_line(ln):
            in_block = True
            skip_next = 2  # the two column header lines immediately after the title
            continue
        if in_block and skip_next > 0:
            skip_next -= 1
            continue
        if in_block:
            if is_block_terminator(ln):
                in_block = False
                continue
            parsed = try_parse_data_line(ln)
            if parsed is not None:
                rows.append(parsed)
    if not rows:
        raise RuntimeError("No radiance rows were detected.")
    cols = [
        "wn_cm1", "wav_um",
        "path_therm_cm1","path_therm_um",
        "surf_emis_cm1","surf_emis_um",
        "surf_refl_cm1","surf_refl_um",
        "total_cm1","total_um",
        "integral_cm1","trans"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.sort_values("wn_cm1", inplace=True)
    df.drop_duplicates(subset=["wn_cm1"], keep="last", inplace=True)
    df.sort_values("wav_um", inplace=True)
    return df      


# Change this to your path
src_path = Path(__file__).parent / "part2" / "raw_out_0_km.txt"
df_0=read_file(src_path)

src_path = Path(__file__).parent / "part2" / "raw_out_10_km.txt"
df_10=read_file(src_path)

src_path = Path(__file__).parent / "part2" / "raw_out_25_km.txt"
df_25=read_file(src_path)

src_path = Path(__file__).parent / "part2" / "raw_out_50_km.txt"
df_50=read_file(src_path)

src_path = Path(__file__).parent / "part2" / "raw_out_100_km.txt"
df_100=read_file(src_path)

# Adding blackbody curves
def planck_radiance_per_nm(wavelength_nm, temperature_k):
    # Undo the solar-disk scaling in planck_irradiance_at_earth,
    # then convert from W m^-2 to W cm^-2.
    solar_dilution = (SOLAR_RADIUS_M / ASTRONOMICAL_UNIT_M) ** 2
    irradiance = planck_irradiance_at_earth(wavelength_nm, temperature_k)
    return irradiance / (np.pi * solar_dilution) * 1e-4


#plt.figure(figsize=(6.3, 4.0))  # ~16 cm x 10 cm
fig, ax = plt.subplots(figsize=(6.3, 4.0))

for df, label in [
    (df_0, "0 km"),
    (df_10, "10 km"),
    (df_25, "25 km"),
    (df_50, "50 km"),
    (df_100, "100 km"),
]:
    wavelength_nm = df["wav_um"].to_numpy() * 1000
    radiance_per_nm = df["total_um"].to_numpy() / 1000
    ax.plot(wavelength_nm, radiance_per_nm, label=label)

wavelength_nm = df_0["wav_um"].to_numpy() * 1000
for temperature_k in [220, 240, 260, 280, 300]:
    ax.plot(
        wavelength_nm,
        planck_radiance_per_nm(wavelength_nm, temperature_k),
        "--",
        label=f"{temperature_k} K",
    )

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral radiance (W cm$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
ax.set_title("MODTRAN total radiance vs wavelength")
ax.set_xlim(5000, 30000)  # Same range as 5–30 µm
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
