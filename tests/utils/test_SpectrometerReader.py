# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pytest

from htc.utils.SpectrometerReader import SpectrometerReader


def test_spectrometer_reader(tmp_path: Path) -> None:
    my_file = """
bla
blablabla
blabla
>>>>>Begin Spectral Data<<<<<
0   12
1   13
2 15
3   -5
    """
    (tmp_path / "blub_HRC.txt").write_text(my_file)

    # test reading of single spectrum file
    reader = SpectrometerReader(tmp_path)
    spec = reader.read_spectrum("blub_HRC.txt")
    assert np.all(spec["wavelengths"] == np.array([0.0, 1.0, 2.0, 3.0])), "wavelength not correctly read"
    assert np.all(spec["spectra"] == np.array([[12.0, 13.0, 15.0, -5.0]])), "Spectrum not correctly read"

    # test error is thrown if file does not exist
    with pytest.raises(FileNotFoundError):
        reader.read_spectrum("ohoh.txt")

    assert reader.read_spectrum("fancy") is None, "Spectrum should be None if no files are found for prefix"

    # test calibration error
    with pytest.raises(AssertionError):
        reader.read_spectrum("blub_HRC.txt", calibration=True)

    # test calibration
    white_file = """
bla
blablabla
blabla
>>>>>Begin Spectral Data<<<<<
0   5
1   3
2   2
3   4
    """
    (tmp_path / "white_calibration_HRC.txt").write_text(white_file)

    dark_file = """
bla
blablabla
blabla
>>>>>Begin Spectral Data<<<<<
0   1
1   1
2 1
3   1
    """
    (tmp_path / "dark_calibration_HRC.txt").write_text(dark_file)

    reader = SpectrometerReader(tmp_path)
    assert np.all(reader.white["wavelengths"] == np.array([0.0, 1.0, 2.0, 3.0])), "wavelength not correctly read"
    assert np.all(reader.dark["wavelengths"] == np.array([0.0, 1.0, 2.0, 3.0])), "wavelength not correctly read"
    assert np.all(reader.white["spectra"] == np.array([[5.0, 3.0, 2, 4.0]])), "White spectrum not correctly read"
    assert np.all(reader.dark["spectra"] == np.array([[1, 1, 1, 1]])), "Dark spectrum not correctly read"

    spec = reader.read_spectrum("blub_HRC.txt", calibration=True)
    assert np.all(spec["wavelengths"] == np.array([0.0, 1.0, 2.0, 3.0])), "wavelength not correctly read"
    assert np.all(
        spec["spectra"]
        == np.array([
            [
                (12 - 1) / (5 - 1),
                (13 - 1) / (3 - 1),
                (15 - 1) / (2 - 1),
                (-5 - 1) / (4 - 1),
            ]
        ])
    ), "Spectrum not correctly calibrated"

    # test normalization
    spec_norm = reader.read_spectrum("blub_HRC.txt", calibration=True, normalization=1)
    spec_norm_manual = spec["spectra"] / np.linalg.norm(spec["spectra"], axis=-1, ord=1, keepdims=True)
    assert np.all(spec_norm["spectra"] == spec_norm_manual), "Error in normalization of spectrum"

    # test median
    my_file2 = """
bla
blablabla
blabla
>>>>>Begin Spectral Data<<<<<
0   8
1   7
2 5
3   25
    """
    (tmp_path / "blub_HRC2.txt").write_text(my_file2)
    spectra_median = reader.read_spectrum("blub")
    assert np.all(spectra_median["wavelengths"] == np.array([0.0, 1.0, 2.0, 3.0])), "wavelength not correctly read"
    assert np.all(spectra_median["median_spectrum"] == np.array([10.0, 10.0, 10.0, 10.0])), (
        "Median computation did not work correctly"
    )

    # test normalization error
    my_file3 = """
bla
blablabla
blabla
>>>>>Begin Spectral Data<<<<<
0   0
1   0
2 0
3   0
    """
    (tmp_path / "blub_HRC3.txt").write_text(my_file3)
    with pytest.raises(AssertionError):
        reader.read_spectrum("blub_HRC3.txt", normalization=1)

    # test custom normalization
    my_realistic_file = """
>>>>>Begin Spectral Data<<<<<
400   1
500   2
501   3
502   4
503   5
504   6
505   7
506   8
507   9
508   10
509   11
510   12
511   13
512   14
513   15
514   16
515   17
550   18
600   19
700   20
750   21
800   22
900   23
1000  24
1100  25
    """
    (tmp_path / "realistic_HRC.txt").write_text(my_realistic_file)
    (tmp_path / "realistic_HRC2.txt").write_text(my_realistic_file)

    wavelengths = np.array([
        500,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        550,
        600,
        700,
        750,
        800,
        900,
        1000,
    ])
    spec_sel = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    spec_norm = (spec_sel / 299 * 23) / 100
    adapted_spec = reader.read_spectrum("realistic", adapt_to_tivita=True, normalization=1)
    assert np.all(adapted_spec["wavelengths"] == wavelengths), "Mismatch between wavelengths"
    assert np.allclose(adapted_spec["median_spectrum"], spec_norm), "Mismatch for median spectrum"
    assert np.all(adapted_spec["std_spectrum"] == np.zeros(23)), "Mismatch for std"

    generated_transformed_spec = reader.read_spectrum("realistic", transform_to_tivita=True, normalization=1)
    transformed_spec = np.zeros(100, dtype=np.float64)
    transformed_spec[0] = np.mean(spec_sel[:5])
    transformed_spec[1] = np.mean(spec_sel[5:10])
    transformed_spec[2] = np.mean(spec_sel[10:15])
    transformed_spec[3] = np.mean(spec_sel[15])
    transformed_spec[10] = spec_sel[16]
    transformed_spec[20] = spec_sel[17]
    transformed_spec[40] = spec_sel[18]
    transformed_spec[50] = spec_sel[19]
    transformed_spec[60] = spec_sel[20]
    transformed_spec[80] = spec_sel[21]
    transformed_spec = transformed_spec / np.linalg.norm(transformed_spec, ord=1)
    assert all(np.nan_to_num(generated_transformed_spec["median_spectrum"]) == transformed_spec), (
        "Mismatch for median spectrum transformed"
    )

    generated_adapted_transformed_spec = reader.read_spectrum(
        "realistic", adapt_to_tivita=True, transform_to_tivita=True, normalization=1
    )
    assert all(
        generated_transformed_spec["median_spectrum"] == generated_adapted_transformed_spec["median_spectrum"]
    ), "Mismatch for adapted and non-adapted transformed spectrum"
