from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_PATH = Path(__file__).with_name("kurudz_0.1nm.dat")
PLOT_PATH = Path(__file__).with_name("kurucz_spectrum_blackbody.png")
SOLAR_TEMPERATURE_K = 5778.0
SOLAR_RADIUS_M = 6.957e8
ASTRONOMICAL_UNIT_M = 1.495978707e11


def planck_irradiance_at_earth(wavelength_nm: np.ndarray) -> np.ndarray:
    """Return the solar blackbody spectrum at 1 AU in W m^-2 nm^-1."""
    planck_constant = 6.62607015e-34
    speed_of_light = 299792458.0
    boltzmann_constant = 1.380649e-23

    wavelength_m = wavelength_nm * 1e-9
    exponent = (
        planck_constant
        * speed_of_light
        / (wavelength_m * boltzmann_constant * SOLAR_TEMPERATURE_K)
    )
    spectral_radiance_per_m = (
        2
        * planck_constant
        * speed_of_light**2
        / wavelength_m**5
        / np.expm1(exponent)
    )
    dilution = (SOLAR_RADIUS_M / ASTRONOMICAL_UNIT_M) ** 2
    return np.pi * spectral_radiance_per_m * dilution * 1e-9


def main() -> None:
    # The data-file header specifies mW / (m^2 nm); convert to W / (m^2 nm).
    wavelength_nm, irradiance_mw_m2_nm = np.loadtxt(
        DATA_PATH, comments="#", unpack=True
    )
    irradiance_w_m2_nm = irradiance_mw_m2_nm * 1e-3
    blackbody_w_m2_nm = planck_irradiance_at_earth(wavelength_nm)

    measured_integral = np.trapezoid(irradiance_w_m2_nm, wavelength_nm)
    blackbody_integral = np.trapezoid(blackbody_w_m2_nm, wavelength_nm)
    difference_percent = (
        (measured_integral - blackbody_integral) / blackbody_integral * 100
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        wavelength_nm,
        irradiance_w_m2_nm,
        color="tab:blue",
        linewidth=0.8,
        label="Kurucz solar spectrum",
    )
    ax.plot(
        wavelength_nm,
        blackbody_w_m2_nm,
        color="black",
        linewidth=2,
        label=f"5778 K blackbody scaled to 1 AU "
        f"({blackbody_integral:.1f} W m$^{{-2}}$ over range)",
    )
    ax.set_xscale("log")
    ax.set_xlim(wavelength_nm.min(), wavelength_nm.max())
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral irradiance (W m$^{-2}$ nm$^{-1}$)")
    ax.set_title("Solar spectral irradiance and 5778 K blackbody")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)

    print(f"Input units: mW m^-2 nm^-1; converted by multiplying by 1e-3.")
    print(
        f"Wavelength coverage: {wavelength_nm.min():.1f}-"
        f"{wavelength_nm.max():.1f} nm ({wavelength_nm.size} samples)"
    )
    print(
        f"Integrated measured irradiance over this range: "
        f"{measured_integral:.2f} W m^-2"
    )
    print(
        f"Integrated blackbody irradiance over this range: "
        f"{blackbody_integral:.2f} W m^-2"
    )
    print(
        f"Measured integral is {difference_percent:.2f}% above the "
        f"blackbody integral."
    )
    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
