# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Iterator

import pandas as pd

from htc.settings import settings


def iterate_datasets() -> Iterator[dict[str, str | pd.DataFrame]]:
    """
    Iterate over all datasets which should automatically appear in the dataset documentation.

    Yields: Dictionary with the dataset name and the corresponding metadata table.
    """
    for name, info in settings.datasets:
        if name in ["2022_08_03_Tivita_unsorted_images", "2024_IPCAI_Simulation_dataset"]:
            continue

        df_meta = pd.read_feather(info["path_intermediates"] / "tables" / f"{name}@meta.feather")
        if "dataset_settings_path" in df_meta and df_meta["dataset_settings_path"].nunique() > 1:
            for dsettings in df_meta["dataset_settings_path"].unique():
                df_sub = df_meta[df_meta["dataset_settings_path"] == dsettings]
                parts = dsettings.split("/")
                if len(parts) > 1:
                    yield {"name": f"{name}#{parts[0]}", "df_meta": df_sub}
                else:
                    yield {"name": name, "df_meta": df_sub}
        else:
            yield {"name": name, "df_meta": df_meta}
