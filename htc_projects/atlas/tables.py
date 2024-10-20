# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc.utils.helper_functions import median_table
from htc_projects.atlas.settings_atlas import settings_atlas


def median_cam_table(all_cameras: bool = False) -> pd.DataFrame:
    """
    Return the median spectra table with all images from the tissue atlas (based on the labels and paper_tag).

    Args:
        all_cameras: If True, select data from all available cameras. If False, only the valid cameras will be included.

    Returns: Table with median spectra.
    """
    df = median_table(dataset_name="2021_02_05_Tivita_multiorgan_masks")

    # We only use a subset of the labels
    labels = settings_atlas.labels
    df = df.query("label_name in @labels")

    # We only use tissue atlas data
    df = df[df["paper_tags"].str.contains(settings_atlas.paper_tag, regex=False, na=False)]

    if not all_cameras:
        # We use only cameras where we have recordings for every organ
        df = df.query("camera_name in @settings_atlas.valid_cameras")

    assert not np.any(pd.isna(df["camera_name"]).values), (
        "All nan camera ids should have been replaced by the unknown camera. This indicates that the corresponding meta"
        " files were not found. Maybe you need to update the tables (e.g. median table) due to new data?"
    )

    return df.reset_index(drop=True)


def standardized_recordings(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Returns the selection of data corresponding to the standardized recordings.

    Args:
        df: If not None, will make the selection on the given table. This is for example useful if data from all annotators should be included.

    Returns: Table with median spectra.
    """
    if df is None:
        df = median_cam_table()

    # In the public dataset we only have standardized recordings
    df = df[
        df["paper_tags"].str.contains(settings_atlas.paper_tag_standardized, regex=False, na=False)
        & ~pd.isna(df["situs"])
    ]

    return df.reset_index(drop=True)
