# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

from pytest_console_scripts import ScriptRunner

import htc.data_processing.run_l1_normalization as run_l1_normalization
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import decompress_file


def test_l1_normalization(make_tmp_example_data: Callable, script_runner: ScriptRunner) -> None:
    tmp_example_data = make_tmp_example_data()
    res = script_runner.run(run_l1_normalization.__file__, "--dataset-name", "2021_02_05_Tivita_multiorgan_semantic")
    assert res.success

    # Check that the newly created masks are identical to the precomputed ones
    existing_dir = settings.intermediates_dir_all / "preprocessing" / "L1"
    tmp_dir = tmp_example_data / "intermediates" / "preprocessing" / "L1"

    tmp_files = list(tmp_dir.iterdir())
    assert len(tmp_files) == len(list(DataPath.iterate(tmp_example_data / "data")))

    for tmp_f in tmp_files:
        f = existing_dir / tmp_f.name
        if f.exists():
            # Precomputed data available, make sure new data is identical
            # Blosc does not give reproducible compression results, but the decompressed data is always the same
            assert (decompress_file(f) == decompress_file(tmp_f)).all()
        else:
            # Precomputed data not available, make sure the semantic dir does not have the corresponding folder name
            assert not (
                settings.intermediates_dir_all.find_location("2021_02_05_Tivita_multiorgan_semantic")
                / "preprocessing"
                / "L1"
            ).exists()
