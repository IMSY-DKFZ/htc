# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from datetime import datetime

import pandas as pd

from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.LabelMapping import LabelMapping


def first_inclusion(target: str = None, label_name: str = None) -> pd.DataFrame:
    """
    Table with the first image for each subject and label.
    """
    df = median_table(dataset_name="2022_10_24_Tivita_sepsis_ICU#subjects")
    df = first_timepoint_filter(df)
    if target is not None:
        if target == "sepsis":
            clear_labels = ["sepsis", "no_sepsis"]
            df = df.query("sepsis_status in @clear_labels")
        elif target == "survival":
            df = df[~pd.isna(df.survival_30_days_post_inclusion)]
        elif target == "shock":
            df = df[~pd.isna(df.shock)]
        elif target == "septic_shock":
            df = df.query("sepsis_status == 'sepsis'")  # goal is to discrimante septic shock from sepsis
            df = df[~pd.isna(df.shock)]
        else:
            raise ValueError("Unknown target")

    if label_name is not None:
        df = df.query("label_name == @label_name")

    return df


def first_timepoint_filter(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Filter for the first timepoint for each subject and label.
    """
    if "date" not in df.columns:
        df["date"] = [
            datetime.strptime(timestamp, settings.tivita_timestamp_format).date() for timestamp in df.timestamp.values
        ]
    if "label_name" in df.columns:
        df = df.groupby(["subject_name", "label_name"], as_index=False).apply(lambda x: x[x["date"] == x.date.min()])
    else:
        df = df.groupby(["subject_name"], as_index=False).apply(lambda x: x[x["date"] == x.date.min()])
    df = df.reset_index(drop=True)

    return df


def valid_labels_filter(df: pd.DataFrame, config: Config, **kwargs) -> pd.DataFrame:
    """
    Filter for valid labels.
    """
    mapping = LabelMapping.from_config(config)
    df = df[mapping.is_index_valid(df["image_labels"].values)]  # drop unknown labels

    return df
