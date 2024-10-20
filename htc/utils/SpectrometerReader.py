# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import numpy as np


class SpectrometerReader:
    def __init__(self, data_dir: Path):
        """
        Reads spectrometer measurements from txt files with a `>>>>>Begin Spectral Data<<<<<` section as produced by the Ocean Insight HR2000+ spectrometer.

        Example directory with colorchecker measurements
        >>> from htc.settings import settings
        >>> spectra_dir = settings.data_dirs.studies / "colorchecker_spectrometer/2022_09_19_MIC1"
        >>> len(list(spectra_dir.iterdir()))  # 100 measurements for all 24 color chips + white and dark calibration
        2600
        >>> reader = SpectrometerReader(spectra_dir)
        >>> reader.label_names()  # doctest: +ELLIPSIS
        ['black', 'blue', 'blue_flower', ...]
        >>> blue_sky = reader.read_spectrum("blue_sky", calibration=True, normalization=1)
        >>> list(blue_sky.keys())
        ['wavelengths', 'spectra', 'median_spectrum', 'std_spectrum']
        >>> blue_sky["wavelengths"][:3]  # First three wavelengths (in nm)
        array([187.255, 187.731, 188.206])
        >>> blue_sky["median_spectrum"][:3]  # First three reflectance values from the median spectrum (calibrated, L1-normalized)
        array([0.00010494, 0.00010494, 0.00010494])

        Args:
            data_dir: Path to the directory which contains the spectrometer files (*.txt files).
        """
        self.data_dir = data_dir
        self.white = self.read_spectrum("white_calibration", calibration=False, normalization=None)
        self.dark = self.read_spectrum("dark_calibration", calibration=False, normalization=None)

    def read_spectrum(
        self,
        label_name: str,
        calibration: bool = False,
        normalization: int = None,
        adapt_to_tivita: bool = False,
        transform_to_tivita: bool = False,
    ) -> dict[str, np.ndarray] | None:
        """
        Reads the spectrometer measurements for a specific label.

        Args:
            label_name: If label_name is a prefix, function will iterate over all files with that prefix in the data directory. To read a single spectrometer file, the path name should be passed.
            calibration: Whether white and dark correction should be performed. Only possible if dark and white measurements exist in the data directory.
            normalization: Which normalization to perform. None: no normalization, 1: L1, 2: L2, ...
            adapt_to_tivita: If True, limit the spectral range to the range of the Tivita camera (from 500 to 1000 nm). If normalization is applied, the resulting spectra is rescaled so that the reflectance values are in the same range as the Tivita reflectance values.
            transform_to_tivita: If True, transform the spectrometer measurements to Tivita measurements by averaging the spectrometer measurements in non-overlapping wavelength ranges corresponding to the Tivita measurements.

        Returns: Dictionary with the following keys (a = number of files, b = dimensionality of the spectrum) or None if no files could be found:
        - `wavelengths`: Array of wavelengths (shape b).
        - `spectra`: Array with the measured reflectance values for each wavelength (shape a x b).
        - `median_spectrum`: Median spectrum across all measurements (shape b).
        - `std_spectrum`: Standard deviation of the spectra across all measurements (shape b).
        - `median_spectrum_mapped`: Median spectrum across all measurements mapped to the Tivita measurements (shape b). Only if transform_to_tivita is True.
        - `std_spectrum_mapped`: Standard deviation of the spectra across all measurements mapped to the Tivita measurements (shape b). Only if transform_to_tivita is True.
        """
        if label_name.endswith(".txt"):
            paths = [self.data_dir / label_name]
        else:
            paths = sorted(self.data_dir.glob(f"{label_name}_HRC*.txt"))

        if len(paths) == 0:
            return None
        else:
            spectra = []
            for path in paths:
                with path.open() as f:
                    lines = f.read().splitlines()

                spectra_start = False
                spectrum_file = []
                for line in lines:
                    if spectra_start:
                        match = re.search(r"(\S+)\s+(\S+)", line)
                        if match is not None:
                            wavelength, value = float(match.group(1)), float(match.group(2))
                            if len(spectrum_file) > 0 and wavelength <= spectrum_file[-1][0]:
                                print(
                                    f"Current wavelength {wavelength} is not larger than previous wavelength"
                                    f" {spectrum_file[-1][0]}"
                                )
                                continue
                            spectrum_file.append([wavelength, value])
                        else:
                            continue

                    if ">>>>>Begin" in line:
                        spectra_start = True

                spectra.append(spectrum_file)

            spectra = np.asarray(spectra)

            if calibration:
                assert (
                    self.dark is not None and self.white is not None
                ), "white and/or dark calibration measurements are missing!"
                spectra[:, :, 1] = (spectra[:, :, 1] - self.dark["median_spectrum"]) / (
                    self.white["median_spectrum"] - self.dark["median_spectrum"]
                )

            if adapt_to_tivita:
                x = spectra[0, :, 0]
                mask = (x <= 1000) & (x >= 500)
                x = x[mask]
                spectra = spectra[:, mask, :]

            if normalization is not None:
                assert not np.any(
                    np.linalg.norm(spectra[:, :, 1], axis=-1, ord=normalization) == 0
                ), f"Normalization failure due to zero division for {label_name}"
                spectra[:, :, 1] = spectra[:, :, 1] / np.linalg.norm(
                    spectra[:, :, 1], axis=-1, keepdims=True, ord=normalization
                )

                if adapt_to_tivita:
                    # Scale the spectrometer measurements to the same range as the TIVITA measurements
                    spectra[:, :, 1] = spectra[:, :, 1] * spectra.shape[1] / 100

            # transform spectrometer measurements to Tivita measurements
            if transform_to_tivita:
                spectra_transformed = np.zeros((spectra.shape[0], 100))
                spectrometer_wavelengths = spectra[0, :, 0]
                tivita_wavelengths = np.linspace(500, 1000, 101)
                for i in np.arange(len(tivita_wavelengths) - 1):
                    mask = (tivita_wavelengths[i] <= spectrometer_wavelengths) & (
                        spectrometer_wavelengths < tivita_wavelengths[i + 1]
                    )
                    if spectra[:, mask, 1].shape[-1] == 0:
                        spectra_transformed[:, i] = 0  # missing data is set to 0
                    else:
                        spectra_transformed[:, i] = np.mean(spectra[:, mask, 1], axis=1)

                return {
                    "wavelengths": spectra[0, :, 0],
                    "spectra": spectra[:, :, 1],
                    "median_spectrum": np.median(spectra[:, :, 1], axis=0),
                    "std_spectrum": np.std(spectra[:, :, 1], axis=0),
                    "median_spectrum_transformed": np.median(spectra_transformed, axis=0),
                    "std_spectrum_transformed": np.std(spectra_transformed, axis=0),
                }

            return {
                "wavelengths": spectra[0, :, 0],
                "spectra": spectra[:, :, 1],
                "median_spectrum": np.median(spectra[:, :, 1], axis=0),
                "std_spectrum": np.std(spectra[:, :, 1], axis=0),
            }

    def label_names(self) -> list[str]:
        """Returns a sorted list of label names in the data directory"""
        names = set()
        for f in self.data_dir.iterdir():
            match = re.search(r"^(\w+)_HRC", f.name)
            assert match is not None, f"Could not extract the label name from {f.name}"
            names.add(match.group(1))

        return sorted(names)
