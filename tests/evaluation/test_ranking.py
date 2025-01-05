# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from htc.evaluation.ranking import BootstrapRanking, BootstrapRankingSubjects


@pytest.mark.parametrize("bootstrap_subjects", [False, True])
def test_bootstrap_ranking(bootstrap_subjects: bool) -> None:
    np.random.seed(0)

    if bootstrap_subjects:
        df_sample = pd.DataFrame(
            [
                ["T1", "A1", "C1", "S1", 0.6],
                ["T1", "A1", "C1", "S2", 0.65],
                ["T1", "A1", "C2", "S1", 0.8],
                ["T1", "A1", "C2", "S2", 0.7],
                ["T1", "A1", "C3", "S1", 0.75],
                ["T1", "A1", "C3", "S2", 0.8],
                ["T1", "A2", "C1", "S1", 0.5],
                ["T1", "A2", "C1", "S2", 0.55],
                ["T1", "A2", "C2", "S1", 0.82],
                ["T1", "A2", "C2", "S2", 0.82],
                ["T1", "A2", "C3", "S1", 0.76],
                ["T1", "A2", "C3", "S2", 0.7],
            ],
            columns=["Task", "Algorithm", "Case", "Subject", "Value"],
        )
        df_bootstraps = BootstrapRankingSubjects(
            df_sample, task="Task", algorithm="Algorithm", case="Case", value="Value", subject_column="Subject"
        ).bootstraps
    else:
        df_sample = pd.DataFrame(
            [
                ["T1", "A1", "C1", 0.6],
                ["T1", "A1", "C2", 0.8],
                ["T1", "A1", "C3", 0.8],
                ["T1", "A2", "C1", 0.5],
                ["T1", "A2", "C2", 0.82],
                ["T1", "A2", "C3", 0.7],
            ],
            columns=["Task", "Algorithm", "Case", "Value"],
        )
        df_bootstraps = BootstrapRanking(
            df_sample, task="Task", algorithm="Algorithm", case="Case", value="Value"
        ).bootstraps

    df_rel = (
        df_bootstraps.groupby(["task", "algorithm", "rank"], as_index=True)[["rank"]]
        .count()
        .rename(columns={"rank": "count"})
        .reset_index()
    )
    df_rel["count"] = df_rel["count"] / 1000
    assert (
        df_rel.query("algorithm == 'A1' and rank == 1")["count"].item()
        > df_rel.query("algorithm == 'A1' and rank == 2")["count"].item()
    )
    assert (
        df_rel.query("algorithm == 'A2' and rank == 1")["count"].item()
        == df_rel.query("algorithm == 'A1' and rank == 2")["count"].item()
    )
    assert (
        df_rel.query("algorithm == 'A2' and rank == 2")["count"].item()
        == df_rel.query("algorithm == 'A1' and rank == 1")["count"].item()
    )
