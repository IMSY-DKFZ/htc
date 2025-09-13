# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import difflib
import os
import re
import warnings
from pathlib import Path

import pytest
from matplotlib import font_manager
from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner

import htc.data_processing.run_superpixel_prediction as run_superpixel_prediction
import htc.evaluation.run_ranking_lr as run_ranking_lr
import htc.models.run_inference_timeit as run_inference_timeit
import htc.rater_variability.run_nsd_thresholds as run_nsd_thresholds
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import execute_notebook


@pytest.mark.serial
@pytest.mark.slow
class TestPaperSemanticFiles:
    training_timestamps = (
        settings_seg.model_comparison_timestamp,
        settings_seg.dataset_size_timestamp,
        settings_seg.lr_experiment_timestamp,
        settings_seg.seed_experiment_timestamp,
    )

    def _training_runs(self) -> list[Path]:
        run_dirs = []

        for timestamp in TestPaperSemanticFiles.training_timestamps:
            for model_name in settings_seg.model_names:
                model_dir = settings.training_dir / model_name
                run_dirs.extend(sorted(model_dir.glob(f"{timestamp}*")))

        return run_dirs

    def test_files_match(
        self, path_test_files: Path, tmp_path: Path, script_runner: ScriptRunner, monkeypatch: MonkeyPatch
    ) -> None:
        # This test generates all paper figures and the variable file based on the trained networks
        # This includes all the intermediate result files (e.g. challengeR runs)

        # Note: If this test fails due to a mismatch of the newly computed and existing files, it can be difficult to find the cause of the problem. To debug this:
        # - Run the test again manually in a docker container (htc docker bash and then py.test -s tests/paper/test_paper_semantic_files.py)
        # - While the container is still running (docker ps), copy the generated files from the docker container to the host (e.g., docker cp <container_id>:/path/from/error/message /host/path)
        # - Compare the generated files with the existing files (/mnt/E130-Projekte/Biophotonics/Projects/2021_02_05_hyperspectral_tissue_classification/test_files/paper_semantic)
        # If the difference is small and negligible (e.g., due to an update of matplotlib), you may select the newly generated file as new reference (by copying it to the test_files folder). However, if the changes are e.g., due to an API change, you should update the code so that the results will be identical again

        fonts = font_manager.findSystemFonts()
        if len([f for f in fonts if "LibertinusSerif-Regular" in f]) == 0:
            pytest.skip("Libertinus font (https://github.com/alerque/libertinus) is not installed")

        run_dirs = self._training_runs()
        if not (len(run_dirs) == 3 * settings_seg.n_algorithms + len(settings_seg.model_colors) + 5):
            pytest.skip("Training files are not available")

        tmp_results = tmp_path / "results_semantic"

        # Make all relevant training run directories available for the test
        for run_dir in run_dirs:
            tmp_model_dir = tmp_results / "training" / run_dir.parent.name
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            (tmp_model_dir / run_dir.name).symlink_to(run_dir)

        # ScriptRunner always use subprocesses so they make use of the new results_dir
        monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_results))
        # For this test, we explicitly do not want other results to be available (everything should just be based on the trained models)
        for env_name in os.environ.keys():
            if env_name.startswith("PATH_HTC_RESULTS_"):
                monkeypatch.setenv(env_name, "")
        monkeypatch.setenv("HTC_ADD_NETWORK_ALTERNATIVES", "false")

        # Prepare all intermediate results
        for script in [run_ranking_lr, run_nsd_thresholds, run_superpixel_prediction, run_inference_timeit]:
            settings.log.info(f"Running script {script}")
            res = script_runner.run(script.__file__)
            assert res.success

        # Run all notebooks in the paper/MIA2022 folder (this creates and saves all paper figures)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="the file is not specified with any extension", category=UserWarning
            )
            for notebook_file in sorted((settings.src_dir / "paper/MIA2022").glob("*ipynb")):
                settings.log.info(f"Running notebook file {notebook_file}")
                execute_notebook(notebook_path=notebook_file)

        # Also generate the variables for the LaTeX document
        res = script_runner.run(settings.src_dir / "paper/MIA2022/run_generate_variables.py")
        assert res.success

        # Now, we start comparing the generated files with the test files (= last known results)
        files_existing = sorted((path_test_files / "paper_semantic").iterdir())
        files_new = sorted((tmp_results / settings_seg.paper_dir.name).iterdir())
        assert [p.name for p in files_existing] == [p.name for p in files_new]

        def read_variable_file(path: Path) -> str:
            with path.open() as f:
                content = f.read()

            # We remove varInferenceTime variable from the comparison since the exact number can't be reproduced
            return re.sub(r"^.*varInferenceTime.*$", "", content, flags=re.MULTILINE)

        # PDF comparison library requires MagickWand library to be installed (https://docs.wand-py.org)
        # This requires special system-wide settings to work (see Dockerfile)
        from wand.image import Image as WandImage

        # PDFs can be quite different even when there is no visible change
        diff_threshold = 0.98
        diff_ratios = []
        for file_existing, file_new in zip(files_existing, files_new, strict=True):
            if file_existing.name == "generated_vars.tex":
                content_existing = read_variable_file(file_existing)
                content_new = read_variable_file(file_new)

                assert content_existing == content_new
                diff_ratios.append(1)
            elif file_existing.suffix == ".pdf":
                # RAW PDf files can be very difficult to compare (cf. https://stackoverflow.com/a/54932474)
                # Here, we are comparing images generated from the PDF file instead
                with (
                    WandImage(filename=f"pdf:{file_existing}", resolution=150) as img_existing,
                    WandImage(filename=f"pdf:{file_new}", resolution=150) as img_new,
                ):
                    diff_ratio = 1 - img_existing.compare(img_new, metric="root_mean_square")[1]
                    diff_ratios.append(diff_ratio)
                    if diff_ratio <= diff_threshold:
                        settings.log.error(f"File: {file_new} (diff ratio: {diff_ratio})")
            else:
                with file_existing.open("rb") as f_existing, file_new.open("rb") as f_new:
                    content_existing = f_existing.read()
                    content_new = f_new.read()

                # Exact comparison is not possible (e.g. due to parts of the PDF which always change) so we use a concept of almost equal here
                diff_ratio = difflib.SequenceMatcher(None, content_existing, content_new).quick_ratio()
                diff_ratios.append(diff_ratio)
                if diff_ratio <= diff_threshold:
                    settings.log.error(f"File: {file_new} (diff ratio: {diff_ratio})")

        assert len(diff_ratios) == len(files_existing)
        assert all(d > diff_threshold for d in diff_ratios)
