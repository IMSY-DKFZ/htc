# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
from pytest_console_scripts import ScriptRunner

import htc.data_processing.run_parameter_images as run_parameter_images
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.blosc_compression import decompress_file


def test_parameter_images(make_tmp_example_data: Callable, script_runner: ScriptRunner) -> None:
    tmp_example_data = make_tmp_example_data()
    res = script_runner.run(run_parameter_images.__file__, "--dataset-name", "2021_02_05_Tivita_multiorgan_semantic")
    assert res.success

    # Check that the newly created files are identical to the precomputed ones
    parameters = settings.intermediates_dir_all / "preprocessing" / "parameter_images"
    tmp_parameters = tmp_example_data / "intermediates" / "preprocessing" / "parameter_images"

    tmp_files = list(tmp_parameters.iterdir())
    assert len(tmp_files) == len(list(DataPath.iterate(tmp_example_data / "data")))

    for tmp_f in tmp_files:
        f = parameters / tmp_f.name
        assert f.exists()
        param = decompress_file(f)
        tmp_param = decompress_file(tmp_f)
        assert param.keys() == tmp_param.keys()
        assert all(np.allclose(param[k], tmp_param[k], atol=1e-05) for k in param.keys())
