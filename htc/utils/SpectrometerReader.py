# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
from pathlib import Path

import numpy as np


def custom_normalization(spec: np.array) -> tuple[np.array, np.array, np.array]:
    """
    Compute median and std of an array of spectrometer measurements, normalized such that it can easily be plotted together with Tivita measurements.

    Args:
        spec: Array of spectrometer measurements of shape a x b x 2 with a: number of samples, b: dimensionality of spectrum

    Returns: Tuple of wavelengths, median of spectra across samples a, std of spectra across samples a.
    """
    assert spec.ndim == 3, (
        "Array of spectra with shape a x b x 2 with a: number of samples, b: dimensionality of spectrum should be"
        f" given, instead shape is {spec.shape}"
    )
    x = spec[0, :, 0]
    mask = (x <= 1000) & (x >= 500)
    x = x[mask]
    spec_mask = spec[:, mask, 1]
    assert not np.any(np.linalg.norm(spec_mask, axis=-1, ord=1) == 0), "Normalization failure due to zero division"
    spec_mask = spec_mask / np.linalg.norm(spec_mask, ord=1, axis=-1, keepdims=True) * len(x) / 100
    y_median = np.median(spec_mask, axis=0)
    y_std = np.std(spec_mask, axis=0)

    return x, y_median, y_std


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
        >>> blue_sky = reader.read_spectrum("blue_sky", calibration=True, normalization=1, median=True)
        >>> blue_sky.shape
        (2048, 2)
        >>> blue_sky[:3, 0]  # First three wavelengths (in nm)
        array([187.255, 187.731, 188.206])
        >>> blue_sky[:3, 1]  # First three reflectance values from the median spectrum (calibrated, L1-normalized)
        array([0.00010494, 0.00010494, 0.00010494])

        Args:
            data_dir: Path to the directory which contains the spectrometer files (*.txt files).
        """
        self.data_dir = data_dir
        self.white = self.read_spectrum("white_calibration", calibration=False, normalization=None, median=True)
        self.dark = self.read_spectrum("dark_calibration", calibration=False, normalization=None, median=True)

    def read_spectrum(
        self, label_name: str, calibration: bool = False, normalization: int = None, median: bool = False
    ) -> np.ndarray:
        """
        Reads the spectrometer measurements for a specific label.

        Args:
            label_name: If label_name is a prefix, function will iterate over all files with that prefix in the data directory. To read a single spectrometer file, the path name should be passed.
            calibration: Whether white and dark balancement should be performed. Only possible if dark and white measurements exist in the data directory.
            normalization: Which normalization to perform. None: no normalization, 1: L1, 2: L2, ...
            median: If False, an array of shape a x b x 2 containing all available spectra is returned, otherwise, the median reflectance per wavelength across all measurements for this label is computed and an array of shape b x 2 is returned (see below).

        Returns:
            if median == False: Array of spectra with shape a x b x 2 with a: number of files, b: dimensionality of spectrum. spectra[:, :, 0] denotes the measurement wavelengths, spectra[:, :, 1] the corresponding intensity measurement.
            if median == True: Array of median spectrum with shape b x 2 with b: dimensionality of spectrum. spectra[:, 0] denotes the measurement wavelengths, spectra[:, 1] the corresponding intensity measurement.
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
                            print(f"{path.name}: Could not extract the spectra values from the line {line}")
                            continue

                    if ">>>>>Begin" in line:
                        spectra_start = True

                spectra.append(spectrum_file)

            spectra = np.asarray(spectra)

            if calibration:
                assert (
                    self.dark is not None and self.white is not None
                ), "white and/or dark calibration measurements are missing!"
                spectra[:, :, 1] = (spectra[:, :, 1] - self.dark[:, 1]) / (self.white[:, 1] - self.dark[:, 1])

            if normalization is not None:
                assert not np.any(
                    np.linalg.norm(spectra[:, :, 1], axis=-1, ord=normalization) == 0
                ), f"Normalization failure due to zero division for {label_name}"
                spectra[:, :, 1] = spectra[:, :, 1] / np.linalg.norm(
                    spectra[:, :, 1], axis=-1, keepdims=True, ord=normalization
                )

            if median:
                spectra = np.median(spectra, axis=0)

            return spectra

    def label_names(self) -> list[str]:
        """Returns a sorted list of label names in the data directory"""
        names = set()
        for f in self.data_dir.iterdir():
            match = re.search(r"^(\w+)_HRC", f.name)
            assert match is not None, f"Could not extract the label name from {f.name}"
            names.add(match.group(1))

        return sorted(names)
