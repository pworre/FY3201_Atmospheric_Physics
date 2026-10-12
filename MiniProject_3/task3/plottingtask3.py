"""FY3201 Mini-project 3, Part 3 - CO2 sweep with MODTRAN (100 km, looking down).

Folder layout (this script lives in the task3 folder):
    task3/
    ├── part3_co2_sweep.py
    ├── co2_flux.csv        columns: co2_ppm, flux_Wm2  ('Upward IR Heat Flux' from the web page)
    └── task3co2data/
        ├── co2_0.txt       raw MODTRAN output ("Show Raw Model Output"), one per run
        └── ...

Outputs (written to task3/):
    fig_part3_spectra.png   selected spectra with blackbody curves
    fig_part3_flux.png      flux vs CO2 and flux vs ln(CO2)
    table_part3_flux.tex    flux table for the report
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "task3co2data"
FLUX_CSV = HERE / "co2_flux.csv"

CO2_SPECTRA = [0, 1, 10, 100, 400, 800]   # runs shown in the spectral figure
LOG_FIT_RANGE = (10, 800)                  # ppm range for the F = a + b ln(C) fit
BB_TEMPS = [220, 240, 260, 280, 300]       # K, background blackbody curves


# ------------------------------------------------------------------ parsing
# The four functions below are copied unchanged from the course-provided
# parse_MODTRAN.py (Blackboard).

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


# ------------------------------------------------------------------ helpers
def to_nm_units(df):
    """Add wavelength in nm and radiance in W m^-2 sr^-1 nm^-1.

    total_um is in W cm^-2 sr^-1 um^-1:
      cm^-2 -> m^-2 : x 1e4
      um^-1 -> nm^-1: x 1e-3      => total factor x 10
    """
    df = df[df["wn_cm1"] > 0].copy()     # 0 cm^-1 would give infinite wavelength
    df["wav_nm"] = df["wav_um"] * 1e3
    df["L_nm"] = df["total_um"] * 10.0
    return df


def planck_nm(wav_nm, T):
    """Blackbody spectral radiance B_lambda(T) in W m^-2 sr^-1 nm^-1."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    w = np.asarray(wav_nm) * 1e-9                       # nm -> m
    return 2 * h * c**2 / w**5 / np.expm1(h * c / (w * k * T)) * 1e-9


# ------------------------------------------------------------------ figures
def plot_spectra():
    fig, ax = plt.subplots(figsize=(8, 4.8))

    # Blackbody curves in the background, labelled at their Wien peak
    wl_bb = np.linspace(5000, 30000, 800)
    for T in BB_TEMPS:
        ax.plot(wl_bb, planck_nm(wl_bb, T), color="0.7", ls="--", lw=0.9)
        x_lab = 2.898e6 / T                              # Wien peak in nm
        ax.text(x_lab, planck_nm(x_lab, T) + 1.5e-4, f"{T} K", color="0.45",
                fontsize=7, ha="center", va="bottom")

    cmap = plt.get_cmap("viridis")
    for i, ppm in enumerate(CO2_SPECTRA):
        df = to_nm_units(read_file(DATA_DIR / f"co2_{ppm}.txt"))
        ax.plot(df["wav_nm"], df["L_nm"], lw=0.7,
                color=cmap(i / (len(CO2_SPECTRA) - 1)), label=f"{ppm} ppm")

    ax.set_xlim(5000, 30000)
    ax.set_ylim(0, 0.0108)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)")
    ax.set_title(r"MODTRAN outgoing spectrum at 100 km for different CO$_2$")
    ax.legend(title=r"CO$_2$", fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig_part3_spectra.png", dpi=200)
    plt.close(fig)


def plot_flux(tab):
    c = tab["co2_ppm"].values.astype(float)
    F = tab["flux_Wm2"].values

    # Least-squares fit F = a + b ln(C) over LOG_FIT_RANGE
    lo, hi = LOG_FIT_RANGE
    m = (c >= lo) & (c <= hi)
    b, a = np.polyfit(np.log(c[m]), F[m], 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) linear concentration axis
    ax1.plot(c, F, "o-", ms=4)
    ax1.set_xlabel(r"CO$_2$ (ppm)")
    ax1.set_ylabel(r"Upward IR heat flux (W m$^{-2}$)")
    ax1.set_title(r"(a) Outgoing flux vs CO$_2$")

    # (b) ln(CO2) axis; 0 ppm cannot be shown since ln(0) = -inf
    pos = c > 0
    ax2.plot(np.log(c[pos]), F[pos], "o", ms=5, label="MODTRAN")
    x = np.linspace(np.log(lo), np.log(hi), 50)
    ax2.plot(x, a + b * x, "k--", lw=1,
             label=rf"fit {lo}–{hi} ppm: $F = {a:.1f} {b:+.2f}\,\ln C$")
    ax2.set_xlabel(r"ln(CO$_2$ / 1 ppm)")
    ax2.set_ylabel(r"Upward IR heat flux (W m$^{-2}$)")
    ax2.set_title(r"(b) Outgoing flux vs ln(CO$_2$)")
    ax2.legend(fontsize=8)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "fig_part3_flux.png", dpi=200)
    plt.close(fig)
    return a, b


# ------------------------------------------------------------------ main
def main():
    tab = pd.read_csv(FLUX_CSV, skipinitialspace=True).sort_values("co2_ppm")

    plot_spectra()
    a, b = plot_flux(tab)
    tab.to_latex(HERE / "table_part3_flux.tex", index=False, float_format="%.2f")

    print(tab.to_string(index=False))
    print(f"\nLog fit {LOG_FIT_RANGE[0]}-{LOG_FIT_RANGE[1]} ppm: "
          f"F = {a:.2f} + ({b:.3f}) ln C")
    F = tab.set_index("co2_ppm")["flux_Wm2"]
    print(f"Delta F = F(800) - F(400) = {F[800] - F[400]:.2f} W m^-2")


if __name__ == "__main__":
    main()