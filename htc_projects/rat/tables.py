# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd

from htc.utils.helper_functions import median_table, sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc_projects.rat.settings_rat import settings_rat


def standardized_recordings_rat(label_mapping: LabelMapping, Camera_CamID: str = None):
    """
    Returns the selection of data corresponding to the standardized recordings of the rat dataset.

    Args:
        label_mapping: The selection of labels to use.
        Camera_CamID: If not None, will make the selection on the given camera.

    Returns: Table with median spectra.
    """
    df = median_table("2023_12_07_Tivita_multiorgan_rat", label_mapping=label_mapping)
    df.drop(columns=["label_name"], inplace=True)
    df.rename(columns={"label_name_mapped": "label_name"}, inplace=True)

    df = df[df["subject_name"].isin(settings_rat.standardized_subjects)]
    if Camera_CamID is not None:
        df = df[df["Camera_CamID"] == Camera_CamID]
    df = df.loc[(~pd.isna(df[["situs", "angle", "repetition"]])).any(axis=1)]  # Only standardized recordings

    df = sort_labels(df)
    return df.reset_index(drop=True)
