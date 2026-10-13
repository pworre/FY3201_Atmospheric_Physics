
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def is_header_line(line: str) -> bool:
    return "RADIANCE(WATTS/CM2-STER" in line


def is_block_terminator(line: str) -> bool:
    return line.strip().startswith("iv")


float_re = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?"
)


def try_parse_data_line(line: str):
    nums = float_re.findall(line)

    if len(nums) == 12:
        return [
            float(x.replace("D", "E").replace("d", "e"))
            for x in nums
        ]

    return None


def read_file(spath):
    text = spath.read_text(errors="ignore").splitlines()

    rows = []
    in_block = False
    skip_next = 0

    for line in text:
        if is_header_line(line):
            in_block = True
            skip_next = 2
            continue

        if in_block and skip_next > 0:
            skip_next -= 1
            continue

        if in_block:
            if is_block_terminator(line):
                in_block = False
                continue

            parsed = try_parse_data_line(line)

            if parsed is not None:
                rows.append(parsed)

    if not rows:
        raise RuntimeError(
            f"No radiance rows were detected in {spath}"
        )

    cols = [
        "wn_cm1",
        "wav_um",
        "path_therm_cm1",
        "path_therm_um",
        "surf_emis_cm1",
        "surf_emis_um",
        "surf_refl_cm1",
        "surf_refl_um",
        "total_cm1",
        "total_um",
        "integral_cm1",
        "trans",
    ]

    df = pd.DataFrame(rows, columns=cols)

    df.sort_values("wn_cm1", inplace=True)
    df.drop_duplicates(subset=["wn_cm1"], keep="last", inplace=True)
    df.sort_values("wav_um", inplace=True)

    return df


# ------------------------------------------------------------
# File paths: change these to match your files
# ------------------------------------------------------------

data_folder = Path("/your/path/to/MODTRAN")

file_paths = {
    "10 km": data_folder / "raw_out_10_km.txt",
    "50 km": data_folder / "raw_out_50_km.txt",
    "100 km": data_folder / "raw_out_100_km.txt",
}


# ------------------------------------------------------------
# Read and plot spectra
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

for label, file_path in file_paths.items():
    df = read_file(file_path)

    # Convert wavelength from micrometres to nanometres
    wavelength_nm = df["wav_um"] * 1000

    plt.plot(
        wavelength_nm,
        df["total_um"],
        label=label,
    )

plt.xlabel("Wavelength (nm)")
plt.ylabel(
    "Spectral radiance "
    r"(W cm$^{-2}$ sr$^{-1}$ µm$^{-1}$)"
)
plt.title("MODTRAN Total Radiance vs Wavelength")

# Original range: 5–30 µm = 5000–30000 nm
plt.xlim(5000, 30000)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()