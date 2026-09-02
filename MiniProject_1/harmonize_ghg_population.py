#!/usr/bin/env python3
"""
Harmonize NOAA GML CH4 / CO2 global annual-mean series with UN World
Population Prospects (WPP) population data, and compute correlation
coefficients over the period common to all series.

Why "harmonize" is needed
--------------------------
- co2_annmean_gl.csv / ch4_annmean_gl.csv: NOAA GML global annual means.
  Already one row per year, but with DIFFERENT COVERAGE (CO2 starts in
  1979, CH4 starts in 1984; both currently run through 2025).
- The UN WPP population file is age-disaggregated: one row per
  (Location, Variant, Time, AgeGrp), covering 1950-2023. To get an
  annual TOTAL population series it must be aggregated (summed over all
  AgeGrp) for a single Location + Variant, and it has different
  DATE COVERAGE than the gas series too.

This script:
  1. Loads the CO2 and CH4 series (already annual).
  2. Streams the (potentially huge) population file in chunks, filters
     to one Location/Variant, and sums PopTotal per year -> one row/year.
  3. Restricts all series to the period covered by ALL of them (the
     "common period") and joins them into a single harmonized table.
  4. Computes Pearson and Spearman correlation coefficients (with
     p-values) between every pair of harmonized series.

Usage
-----
    python harmonize_ghg_population.py \
        --co2 co2_annmean_gl.csv \
        --ch4 ch4_annmean_gl.csv \
        --population WPP_population_by_age.csv \
        --location World --variant Medium

The --population argument is optional. If it is omitted, or the file
can't be found, the script simply harmonizes and correlates CO2 vs CH4.
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------

def load_noaa_annual_mean(path: str, value_name: str) -> pd.DataFrame:
    """Load a NOAA GML '*_annmean_gl.csv' file.

    These files have a variable-length '#'-commented header followed by a
    'year,mean,unc' header row and one data row per year. They are
    already at annual resolution, so no resampling is needed here -
    just parsing, renaming, and indexing by year.
    """
    df = pd.read_csv(path, comment="#")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[["year", "mean"]].rename(columns={"mean": value_name})
    return df.set_index("year").sort_index()


def load_population_annual_total(
    path: str,
    location: str = "World",
    variant: str = "Medium",
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Aggregate a UN WPP-style population-by-age CSV into an annual total
    population series (in thousands), for one Location and Variant.

    Expected (relevant) columns:
        Location, Variant, Time, AgeGrp, PopTotal

    The file is read in chunks so that only rows matching the requested
    Location/Variant are ever fully materialized, and PopTotal is summed
    across all age groups (AgeGrp) for each Time (year) to get the total
    population - this collapses the age dimension the raw file carries.
    """
    usecols = ["Location", "Variant", "Time", "PopTotal"]
    totals: dict[int, float] = {}

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)
    for chunk in reader:
        sub = chunk[(chunk["Location"] == location) & (chunk["Variant"] == variant)]
        if sub.empty:
            continue
        grouped = sub.groupby("Time")["PopTotal"].sum()
        for year, val in grouped.items():
            totals[int(year)] = totals.get(int(year), 0.0) + float(val)

    if not totals:
        raise ValueError(
            f"No rows matched Location={location!r} and Variant={variant!r}. "
            "Double-check the exact spelling used in the population file "
            "(e.g. 'World', 'WORLD', or a specific country/region name, "
            "and 'Medium', 'High', 'Low', etc. for Variant)."
        )

    pop = pd.Series(totals, name="population_thousands").sort_index()
    pop.index.name = "year"
    return pop.to_frame()


# ---------------------------------------------------------------------
# Harmonization
# ---------------------------------------------------------------------

def harmonize(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Join year-indexed DataFrames, restricted to the period common to all.

    Coverage differs across inputs (e.g. CO2 1979-2025, CH4 1984-2025,
    population 1950-2023), so the harmonized period is the intersection:
    max(all start years) to min(all end years).
    """
    start = max(df.index.min() for df in frames)
    end = min(df.index.max() for df in frames)
    if start > end:
        raise ValueError("The input series do not share any common period.")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.join(df, how="inner")
    return merged.loc[start:end]


# ---------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------

def correlation_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pearson = df.corr(method="pearson")
    spearman = df.corr(method="spearman")

    print("\nPearson correlation matrix:")
    print(pearson.round(4).to_string())

    print("\nSpearman correlation matrix:")
    print(spearman.round(4).to_string())

    print("\nPairwise Pearson correlations (with p-values):")
    cols = list(df.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r, p = stats.pearsonr(df[cols[i]], df[cols[j]])
            print(f"  {cols[i]:>22s} vs {cols[j]:<22s}: r = {r:7.4f}   p = {p:.2e}   n = {len(df)}")

    return pearson, spearman


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--co2", default="data/co2_annmean_gl.csv", help="Path to co2_annmean_gl.csv")
    parser.add_argument("--ch4", default="data/ch4_annmean_gl.csv", help="Path to ch4_annmean_gl.csv")
    parser.add_argument(
        "--population",
        default="data/WPP2024_PopulationExposureBySingleAgeSex_Medium_1950-2023.csv",
        help="Path to the UN WPP population-by-age CSV (optional; large file, read in chunks).",
    )
    parser.add_argument("--location", default="World", help="Location value to filter population data on.")
    parser.add_argument("--variant", default="Medium", help="Variant value to filter population data on.")
    parser.add_argument("--out", default="harmonized_annual.csv", help="Where to save the harmonized table.")
    args = parser.parse_args()

    frames = [
        load_noaa_annual_mean(args.co2, "co2_ppm"),
        load_noaa_annual_mean(args.ch4, "ch4_ppb"),
    ]

    if args.population:
        pop_path = Path(args.population)
        if pop_path.exists():
            print(f"Aggregating population data from {pop_path} "
                  f"(Location={args.location!r}, Variant={args.variant!r}) ...")
            frames.append(load_population_annual_total(pop_path, args.location, args.variant))
        else:
            print(f"Note: population file '{pop_path}' not found - continuing with CO2/CH4 only.")

    merged = harmonize(frames)
    print(f"\nHarmonized common period: {merged.index.min()}-{merged.index.max()} "
          f"({len(merged)} years)\n")
    print(merged.to_string())

    merged.to_csv(args.out)
    print(f"\nSaved harmonized annual dataset to: {args.out}")

    correlation_report(merged)


if __name__ == "__main__":
    main()
