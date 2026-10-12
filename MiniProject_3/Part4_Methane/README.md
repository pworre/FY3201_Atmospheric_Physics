# Part 4 — Methane sweep

## Setup

Use the same MODTRAN settings for every run except methane concentration:
- CO2 = 427 ppm
- CH4 = 0, 0.5, 1.0, 1.5, 1.9, 2.5, 3.0, 4.0, 5.0 ppm
- Keep the atmospheric profile, viewing geometry/altitude, clouds, rain, and all other settings unchanged.
- Use the assignment's specified geometry (100 km, looking down, if that is the configuration used in the earlier parts).

## Files

- `raw_output/`: put each raw MODTRAN output here, using the exact filenames in `plot_methane.py`.
- `methane_flux.csv`: enter the `IR out` value reported by MODTRAN for each methane run. Do not calculate this from spectral radiance.
- `co2_flux.csv` (optional): create this using your Part 3 data, with columns `co2_ppm,ir_out_w_m2`. Include the 427 ppm baseline to enable the comparison plot.
- `results/`: generated plots and selected spectrum CSV files.

## Run

Install dependencies if needed:
```bash
python -m pip install numpy pandas matplotlib
```

Then run from this folder:
```bash
python plot_methane.py
```

The script can be run before all nine spectra or flux values are available; it plots available data and reports missing files/values. The spectrum parser is based on the supplied `parse_MODTRAN.py`; if the raw MODTRAN table format differs, parsing may need to be adjusted to the actual output.
