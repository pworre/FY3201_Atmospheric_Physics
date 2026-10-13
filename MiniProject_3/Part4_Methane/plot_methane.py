"""
Atmosfærefysikk – Mini-project 3, Part 4: Methane sweep
1. Save each MODTRAN raw output in raw_output/ using the filenames below.
2. Enter the corresponding MODTRAN "IR out" value (W m^-2) in methane_flux.csv.
3. Run: python plot_methane.py
The script does NOT calculate IR out from spectral radiance. IR out is an
integrated flux reported by MODTRAN and must be copied from the MODTRAN run.
"""
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_output"
RESULTS_DIR = ROOT / "results"
FLUX_CSV = ROOT / "methane_flux.csv"
CO2_CSV = ROOT / "co2_flux.csv"  # Optional: Part 3 data for comparison
# Use these exact filenames for the nine CH4 runs.
CH4_VALUES = [0.0, 0.5, 1.0, 1.5, 1.9, 2.5, 3.0, 4.0, 5.0]
RAW_FILES = {
       0.0: "methane_0ppm.txt",
       0.5: "methane_0p5ppm.txt",
       1.0: "methane_1ppm.txt",
       1.5: "methane_1p5ppm.txt",
       1.9: "methane_1p9ppm.txt",
       2.5: "methane_2p5ppm.txt",
       3.0: "methane_3ppm.txt",
       4.0: "methane_4ppm.txt",
       5.0: "methane_5ppm.txt",
}
# Select spectra to highlight in the report. All available spectra are also
# plotted in a separate figure.
SELECTED_SPECTRA = [0.0, 1.0, 3.0, 4.0, 5.0]
# MODTRAN text output parsing settings based on the supplied parse_MODTRAN.py.
RADIANCE_HEADER = "RADIANCE(WATTS/CM2-STER"
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")
def create_flux_template_if_missing():
       """Create the methane flux input file without overwriting existing data."""
       if not FLUX_CSV.exists():
           pd.DataFrame({
               "ch4_ppm": CH4_VALUES,
               "ir_out_w_m2": [np.nan] * len(CH4_VALUES),
           }).to_csv(FLUX_CSV, index=False)
           print(f"Created {FLUX_CSV}")
           print("Fill in ir_out_w_m2 using the MODTRAN 'IR out' value for each run.")
def parse_numeric_data_line(line):
       """Parse a radiance data row containing the expected 12 numeric columns."""
       values = FLOAT_RE.findall(line)
       if len(values) != 12:
           return None
       values = [v.replace("D", "E").replace("d", "e") for v in values]
       try:
           return [float(v) for v in values]
       except ValueError:
           return None
def read_modtran_spectrum(path):
       """
       Read the radiance table from one MODTRAN raw output.
       Columns follow the supplied parser:
       wn_cm1, wav_um, path_therm_cm1, path_therm_um, surf_emis_cm1,
       surf_emis_um, surf_refl_cm1, surf_refl_um, total_cm1, total_um,
       integral_cm1, trans.
       """
       if not path.exists():
           raise FileNotFoundError(
               f"Missing MODTRAN output: {path}\n"
               "Save the raw output in raw_output/ with the expected filename."
           )
       rows = []
       in_block = False
       skip_next = 0
       with path.open("r", errors="ignore") as file:
           for line in file:
               if RADIANCE_HEADER in line:
                   in_block = True
                   skip_next = 2
                   continue
               if not in_block:
                   continue
               if skip_next > 0:
                   skip_next -= 1
                   continue
               # The supplied parser identifies the end of this table by "iv".
               if line.strip().startswith("iv"):
                   in_block = False
                   continue
               row = parse_numeric_data_line(line)
               if row is not None:
                   rows.append(row)
       if not rows:
           raise RuntimeError(
               f"No radiance rows found in {path.name}. "
               "Check that this is the raw MODTRAN output and that its table "
               "format matches the supplied parser."
           )
       columns = [
           "wn_cm1", "wav_um",
           "path_therm_cm1", "path_therm_um",
           "surf_emis_cm1", "surf_emis_um",
           "surf_refl_cm1", "surf_refl_um",
           "total_cm1", "total_um",
           "integral_cm1", "trans",
       ]
       df = pd.DataFrame(rows, columns=columns)
       df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["wav_um", "total_um"])
       df = df.drop_duplicates(subset=["wav_um"], keep="last")
       df = df.sort_values("wav_um").reset_index(drop=True)
       if df.empty:
           raise RuntimeError(f"The radiance table in {path.name} contains no usable rows.")
       return df
def load_methane_flux():
       """Read measured MODTRAN IR-out values, preserving missing values."""
       if not FLUX_CSV.exists():
           create_flux_template_if_missing()
       flux = pd.read_csv(FLUX_CSV)
       required = {"ch4_ppm", "ir_out_w_m2"}
       if not required.issubset(flux.columns):
           raise ValueError(
               f"{FLUX_CSV.name} must contain columns: ch4_ppm, ir_out_w_m2"
           )
       flux["ch4_ppm"] = pd.to_numeric(flux["ch4_ppm"], errors="coerce")
       flux["ir_out_w_m2"] = pd.to_numeric(flux["ir_out_w_m2"], errors="coerce")
       flux = flux.dropna(subset=["ch4_ppm"])
       return flux.sort_values("ch4_ppm")
def plot_all_spectra(spectra):
       RESULTS_DIR.mkdir(parents=True, exist_ok=True)
       fig, ax = plt.subplots(figsize=(10, 6))
       plotted = 0
       for ch4 in CH4_VALUES:
           if ch4 not in spectra:
               continue
           df = spectra[ch4]
           ax.plot(df["wav_um"] * 1000, df["total_um"], label=f"CH₄ = {ch4:g} ppm")
           plotted += 1
       if plotted == 0:
           print("No spectra available to plot.")
           plt.close(fig)
           return
       ax.set_xlim(5000, 30000)
       ax.set_xlabel("Wavelength (nm)")
       ax.set_ylabel(
           r"Spectral radiance (W cm$^{-2}$ sr$^{-1}$ µm$^{-1}$)"
       )
       ax.set_title("MODTRAN outgoing infrared spectrum for different CH₄ levels")
       ax.grid(True, alpha=0.3)
       ax.legend(ncol=2, fontsize=9)
       fig.tight_layout()
       fig.savefig(RESULTS_DIR / "methane_spectra_all.png", dpi=300)
       plt.close(fig)
       print(f"Saved {RESULTS_DIR / 'methane_spectra_all.png'}")
def plot_selected_spectra(spectra):
       RESULTS_DIR.mkdir(parents=True, exist_ok=True)
       fig, ax = plt.subplots(figsize=(9, 5.5))
       plotted = 0
       for ch4 in SELECTED_SPECTRA:
           if ch4 not in spectra:
               warnings.warn(f"Selected spectrum missing for CH₄ = {ch4:g} ppm.")
               continue
           df = spectra[ch4]
           ax.plot(df["wav_um"] * 1000, df["total_um"], label=f"CH₄ = {ch4:g} ppm")
           # Export the spectrum values for reuse in the report or other software.
           df[["wav_um", "total_um"]].to_csv(
               RESULTS_DIR / f"spectrum_ch4_{str(ch4).replace('.', 'p')}ppm.csv",
               index=False,
           )
           plotted += 1
       if plotted == 0:
           print("No selected spectra available to plot.")
           plt.close(fig)
           return
       ax.set_xlim(5000, 30000)
       ax.set_xlabel("Wavelength (nm)")
       ax.set_ylabel(
           r"Spectral radiance (W cm$^{-2}$ sr$^{-1}$ µm$^{-1}$)"
       )
       ax.set_title("Selected MODTRAN spectra at different CH₄ levels")
       ax.grid(True, alpha=0.3)
       ax.legend()
       fig.tight_layout()
       fig.savefig(RESULTS_DIR / "methane_spectra_selected.png", dpi=300)
       plt.close(fig)
       print(f"Saved {RESULTS_DIR / 'methane_spectra_selected.png'}")
def plot_zoomed_methane_band(spectra):
    """Plot the 7–9 µm region to highlight methane-related spectral changes."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for ch4 in SELECTED_SPECTRA:
        if ch4 not in spectra:
            continue
        df = spectra[ch4]
        ax.plot(
            df["wav_um"] * 1000,
            df["total_um"],
            label=f"CH₄ = {ch4:g} ppm",
        )
        plotted += 1
    if plotted == 0:
        print("No spectra available for the methane-band zoom.")
        plt.close(fig)
        return
    ax.set_xlim(7000, 9000)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(
        r"Spectral radiance (W cm$^{-2}$ sr$^{-1}$ µm$^{-1}$)"
    )
    ax.set_title("Zoomed methane-band region (7–9 µm)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output = RESULTS_DIR / "methane_spectra_zoom.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved {output}")


def plot_methane_flux(flux):
       valid = flux.dropna(subset=["ir_out_w_m2"]).sort_values("ch4_ppm")
       if valid.empty:
           print(
               "No IR-out values entered yet. Fill in methane_flux.csv and run again."
           )
           return
       fig, ax = plt.subplots(figsize=(8, 5))
       ax.plot(valid["ch4_ppm"], valid["ir_out_w_m2"], "o-", linewidth=1.5)
       ax.set_xlabel("Methane concentration (ppm)")
       ax.set_ylabel(r"MODTRAN IR out (W m$^{-2}$)")
       ax.set_title("Upward infrared flux versus methane concentration")
       ax.grid(True, alpha=0.3)
       fig.tight_layout()
       fig.savefig(RESULTS_DIR / "methane_flux.png", dpi=300)
       plt.close(fig)
       print(f"Saved {RESULTS_DIR / 'methane_flux.png'}")
       # Also export the values actually used in the plot.
       valid.to_csv(RESULTS_DIR / "methane_flux_used.csv", index=False)
def plot_optional_co2_comparison(methane_flux):
       """
       Optional Part 3 comparison. If co2_flux.csv exists, it must have columns
       co2_ppm, ir_out_w_m2. The plot shows flux change relative to the baseline
       (CH4=1.9 ppm and CO2=427 ppm), not the absolute fluxes on one shared scale.
       """
       if not CO2_CSV.exists():
           print(
               "Optional comparison skipped: co2_flux.csv not found. "
               "To compare with Part 3, create it with columns "
               "'co2_ppm,ir_out_w_m2'."
           )
           return
       co2 = pd.read_csv(CO2_CSV)
       required = {"co2_ppm", "ir_out_w_m2"}
       if not required.issubset(co2.columns):
           warnings.warn(
               "Skipping CO₂ comparison: co2_flux.csv needs columns "
               "'co2_ppm' and 'ir_out_w_m2'."
           )
           return
       co2["co2_ppm"] = pd.to_numeric(co2["co2_ppm"], errors="coerce")
       co2["ir_out_w_m2"] = pd.to_numeric(co2["ir_out_w_m2"], errors="coerce")
       co2 = co2.dropna(subset=["co2_ppm", "ir_out_w_m2"])
       methane = methane_flux.dropna(subset=["ch4_ppm", "ir_out_w_m2"]).copy()
       if methane.empty or co2.empty:
           print("Optional CO₂ comparison skipped: no complete flux values.")
           return
       # Use the exact baseline if present; otherwise skip rather than guessing.
       ch4_base = methane[np.isclose(methane["ch4_ppm"], 1.9)]
       co2_base = co2[np.isclose(co2["co2_ppm"], 427.0)]
       if ch4_base.empty or co2_base.empty:
           print(
               "Optional CO₂ comparison skipped: include CH₄=1.9 ppm and "
               "CO₂=427 ppm baseline fluxes."
           )
           return
       ch4_base_flux = float(ch4_base.iloc[0]["ir_out_w_m2"])
       co2_base_flux = float(co2_base.iloc[0]["ir_out_w_m2"])
       methane["delta_ir_out_w_m2"] = methane["ir_out_w_m2"] - ch4_base_flux
       co2["delta_ir_out_w_m2"] = co2["ir_out_w_m2"] - co2_base_flux
       fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
       axes[0].plot(
           methane["ch4_ppm"], methane["delta_ir_out_w_m2"], "o-"
       )
       axes[0].axhline(0, linewidth=0.8)
       axes[0].set_xlabel("CH₄ concentration (ppm)")
       axes[0].set_ylabel(r"Change in IR out (W m$^{-2}$)")
       axes[0].set_title("Methane (relative to 1.9 ppm)")
       axes[0].grid(True, alpha=0.3)
       axes[1].plot(co2["co2_ppm"], co2["delta_ir_out_w_m2"], "o-")
       axes[1].axhline(0, linewidth=0.8)
       axes[1].set_xlabel("CO₂ concentration (ppm)")
       axes[1].set_ylabel(r"Change in IR out (W m$^{-2}$)")
       axes[1].set_title("CO₂ (relative to 427 ppm)")
       axes[1].grid(True, alpha=0.3)
       fig.suptitle("Comparison of methane and CO₂ flux responses")
       fig.tight_layout()
       fig.savefig(RESULTS_DIR / "methane_vs_co2_flux_change.png", dpi=300)
       plt.close(fig)
       print(f"Saved {RESULTS_DIR / 'methane_vs_co2_flux_change.png'}")
def main():
       RESULTS_DIR.mkdir(parents=True, exist_ok=True)
       create_flux_template_if_missing()
       spectra = {}
       missing = []
       for ch4 in CH4_VALUES:
           path = RAW_DIR / RAW_FILES[ch4]
           if not path.exists():
               missing.append(path.name)
               continue
           try:
               spectra[ch4] = read_modtran_spectrum(path)
               print(f"Read CH₄={ch4:g} ppm: {path.name} ({len(spectra[ch4])} rows)")
           except (RuntimeError, ValueError) as exc:
               warnings.warn(str(exc))
       if missing:
           print("\nRaw output files not found (those spectra will be skipped):")
           for name in missing:
               print(f"  - {name}")
       if spectra:
           plot_all_spectra(spectra)
           plot_selected_spectra(spectra)
           plot_zoomed_methane_band(spectra)
       else:
           print(
               "\nNo spectra loaded. Put the MODTRAN raw output text files in "
               "raw_output/ using the filenames listed in this script."
           )
       flux = load_methane_flux()
       plot_methane_flux(flux)
       plot_optional_co2_comparison(flux)
       print("\nDone. Results are saved in the results/ folder.")
if __name__ == "__main__":
       main()
