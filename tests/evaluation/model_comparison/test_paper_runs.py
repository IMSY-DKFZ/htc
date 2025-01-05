# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pytest

from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs
from htc.settings import settings
from htc.settings_seg import settings_seg


@pytest.mark.parametrize(
    "timestamp", [settings_seg.model_comparison_timestamp, f"{settings_seg.lr_experiment_timestamp}*lr=0.0001"]
)
def test_collect_comparison_runs(timestamp: str) -> None:
    possible_runs1 = []
    possible_runs2 = []
    for model_name in settings_seg.model_names:
        model_dir = settings.training_dir / model_name

        possible_runs1 += sorted(model_dir.glob(f"{timestamp}*model_comparison"))
        possible_runs2 += sorted(model_dir.glob(timestamp))

    if len(possible_runs1) + len(possible_runs2) == 0:
        pytest.skip("Training runs are not available")

    df = collect_comparison_runs(timestamp)
    assert len(df) == 5
    assert set(df.columns.to_list()) == {
        "model",
        "name",
        "main_loss",
        "run_rgb",
        "run_param",
        "run_hsi",
        "model_image_size",
    }
    assert set(df["model"].unique()) == {"image", "patch", "superpixel_classification", "pixel"}

    for i, row in df.iterrows():
        assert "rgb" in row["run_rgb"]
        assert "param" in row["run_param"]
        assert "rgb" not in row["run_hsi"] and "param" not in row["run_hsi"]

        if row["name"] == "patch_64":
            assert "default_64" in row["run_rgb"]
            assert "default_64" in row["run_param"]
            assert "default_64" in row["run_hsi"]
