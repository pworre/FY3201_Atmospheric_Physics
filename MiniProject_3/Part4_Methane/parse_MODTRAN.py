#parse_MODTRAN.py
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path



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
src_path = Path("C:\\...your location...\\raw_out_10_km.txt")
df_10=read_file(src_path)

src_path = Path("C:\\...your location...\\raw_out_50_km.txt")
df_50=read_file(src_path)

src_path = Path("C:\\...your location...\\raw_out_100_km.txt")
df_100=read_file(src_path)


plt.figure(figsize=(6.3, 4.0))  # ~16 cm x 10 cm
plt.plot(df_10["wav_um"].values, df_10["total_um"].values,label="10km")
plt.plot(df_50["wav_um"].values, df_50["total_um"].values,label="50km")
plt.plot(df_100["wav_um"].values, df_100["total_um"].values,label="100km")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral radiance (W cm$^{-2}$ sr$^{-1}$ µm$^{-1}$)")
plt.title("MODTRAN total radiance vs wavelength")
plt.tight_layout()
plt.xlim([5,30])
plt.legend()
plt.show()
