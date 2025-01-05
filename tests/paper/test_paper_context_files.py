# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import difflib
import os
import warnings
from pathlib import Path

import pytest
from matplotlib import font_manager
from pytest import MonkeyPatch
from pytest_console_scripts import ScriptRunner

from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import execute_notebook
from htc_projects.context.settings_context import settings_context


@pytest.mark.serial
@pytest.mark.slow
class TestPaperContextFiles:
    def test_files_match(
        self, path_test_files: Path, tmp_path: Path, script_runner: ScriptRunner, monkeypatch: MonkeyPatch
    ) -> None:
        # This test generates all paper figures and the variable file based on the trained networks
        # Note: In case this test fails, see the test_paper_semantic_files.py script for debugging instructions

        fonts = font_manager.findSystemFonts()
        if len([f for f in fonts if "LibertinusSerif-Regular" in f]) == 0:
            pytest.skip("Libertinus font (https://github.com/alerque/libertinus) is not installed")

        run_dirs = (
            list(settings_context.best_transform_runs.values())
            + list(settings_context.best_transform_runs_rgb.values())
            + list(settings_context.glove_runs.values())
            + list(settings_context.glove_runs_rgb.values())
        )

        # We also need the MIA runs
        for model_name in settings_seg.model_names:
            for run_dir in sorted(
                (settings.training_dir / model_name).glob(f"{settings_seg.model_comparison_timestamp}*")
            ):
                run_dirs.append(run_dir)

        if len(run_dirs) == 0:
            pytest.skip("Training files are not available")
        assert len(run_dirs) == 16 + settings_seg.n_algorithms

        tmp_results = tmp_path / "results_context"

        # Make all relevant training run directories available for the test
        for run_dir in run_dirs:
            tmp_model_dir = tmp_results / "training" / run_dir.parent.name
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            os.symlink(run_dir, tmp_model_dir / run_dir.name)

        # Make all the additional files which we need for the figures available
        required_files = [
            # We need to compute the NSD metric
            settings.results_dir / "rater_variability" / "nsd_thresholds_semantic.csv",
            # Inference results for the old MIA2022 runs
            settings.results_dir / "neighbour_analysis" / "masks_isolation",
            settings.results_dir / "neighbour_analysis" / "organ_isolation_0",
            settings.results_dir / "neighbour_analysis" / "organ_isolation_cloth",
            settings.results_dir / "neighbour_analysis" / "organ_removal_0",
            settings.results_dir / "neighbour_analysis" / "organ_removal_cloth",
            # Predictions for the example images
            settings.results_dir / "predictions" / "2023-02-21_23-14-44_glove_baseline",
            settings.results_dir / "predictions" / "2023-02-21_23-14-55_glove_organ_transplantation_0.8",
        ]

        for f in required_files:
            assert f.exists()
            tmp_f = tmp_results / f.relative_to(settings.results_dir)
            if not tmp_f.parent.exists():
                tmp_f.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(f, tmp_f)

        # ScriptRunner always use subprocesses so they make use of the new results_dir
        monkeypatch.setenv("PATH_HTC_RESULTS", str(tmp_results))
        # For this test, we explicitly do not want other results to be available (everything should just be based on the trained models)
        for env_name in os.environ.keys():
            if env_name.startswith("PATH_HTC_RESULTS_"):
                monkeypatch.setenv(env_name, "")
        monkeypatch.setenv("HTC_ADD_NETWORK_ALTERNATIVES", "false")

        # Run all notebooks in the paper/MICCAI2023 folder (this creates and saves all paper figures)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="the file is not specified with any extension", category=UserWarning
            )
            for notebook_file in sorted((settings.src_dir / "paper/MICCAI2023").glob("*ipynb")):
                settings.log.info(f"Running notebook file {notebook_file}")
                execute_notebook(notebook_path=notebook_file)

        # Also generate the variables for the LaTeX document
        res = script_runner.run(settings.src_dir / "paper/MICCAI2023/run_generate_variables.py")
        assert res.success

        # Now, we start comparing the generated files with the test files (= last known results)
        files_existing = sorted((path_test_files / "paper_context").iterdir())
        files_new = sorted((tmp_results / settings_context.paper_dir.name).iterdir())
        assert [p.name for p in files_existing] == [p.name for p in files_new]

        def read_variable_file(path: Path) -> str:
            with path.open() as f:
                content = f.read()

            return content

        # PDF comparison library requires MagickWand library to be installed (https://docs.wand-py.org)
        # This requires special system-wide settings to work (see Dockerfile)
        from wand.image import Image as WandImage

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
