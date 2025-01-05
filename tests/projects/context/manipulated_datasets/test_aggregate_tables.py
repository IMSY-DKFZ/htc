# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from htc.settings import settings
from htc_projects.context.manipulated_datasets.run_aggregate_tables import (
    aggregate_isolation_table,
    aggregate_removal_table,
)


@pytest.mark.parametrize(
    "name, agg_func", [("isolation", aggregate_isolation_table), ("removal", aggregate_removal_table)]
)
def test_aggregation(name: str, agg_func: Callable) -> None:
    # For the image model, we already have both an aggregated table and subtables (produced by different scripts)
    experiment_dir = (
        settings.results_dir
        / f"neighbour_analysis/organ_{name}_0/patch/2022-02-03_22-58-44_generated_default_model_comparison"
    )
    df_true = pd.read_pickle(experiment_dir / f"test_table_{name}_0.pkl.xz")
    df_agg = agg_func(experiment_dir, table_name="test_table_ttt")

    assert len(df_true) == len(df_agg)
    assert (df_agg.image_name.values == df_true.image_name.values).all()
    assert all((df_true.iloc[i].used_labels == df_agg.iloc[i].used_labels).all() for i in range(len(df_true)))
    assert (
        np.max([np.sum(np.abs(df_true.iloc[i].dice_metric - df_agg.iloc[i].dice_metric)) for i in range(len(df_true))])
        < 0.01
    )
