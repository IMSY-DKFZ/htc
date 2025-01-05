# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from htc_projects.context.models.context_evaluation import aggregate_removal_table, find_best_transform_run
from htc_projects.context.settings_context import settings_context


def test_aggregate_removal_table(tmp_path: Path) -> None:
    df_data = pd.DataFrame(
        [
            [[1, 2], 0, [0.1, 0.3], "I1", "T1"],
            [[0, 2], 1, [0.4, 0.2], "I1", "T1"],
            [[0, 1], 2, [0.5, 0.8], "I1", "T1"],
        ],
        columns=["used_labels", "target_label", "dice_metric", "image_name", "timestamp"],
    )
    df_data.to_pickle(tmp_path / "test_table.pkl.xz")

    df_expected = pd.DataFrame(
        [
            [[0, 1, 2], [0.4, 0.1, 0.2], "I1", "T1"],
        ],
        columns=["used_labels", "dice_metric", "image_name", "timestamp"],
    )

    df_new = aggregate_removal_table(tmp_path / "test_table.pkl.xz")
    assert_frame_equal(df_expected, df_new)


def test_best_transform_runs() -> None:
    for name, best_run in settings_context.best_transform_runs.items():
        assert find_best_transform_run(name).name == best_run.name
