# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table
from htc.utils.import_extra import requires_extra
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu

try:
    from pypdf import PdfReader

    _missing_library = ""
except ImportError:
    _missing_library = "pypdf"


def config_meta_selection(config: Config, attribute_names: list[str] | str) -> Config:
    """
    Selects meta attributes for the given config based on the provided attribute names.

    >>> config = Config("htc_projects/sepsis_icu/configs/sepsis-inclusion_palm_meta.json")
    >>> config = config_meta_selection(config, attribute_names=["vital", "CRP"])
    >>> [attribute["name"] for attribute in config["input/meta/attributes"]]
    ['heart_frequency', 'sinusrhythm', 'MAP', 'systolic_blood_pressure', 'temperature', 'SpO2', 'CRP']

    Args:
        config: The configuration object where the meta attributes should be added. Existing attributes will be removed.
        attribute_names: Either a list of names or a string where names are concatenates via name1+name2. Each name must either refer to a specific meta attribute or to a group name.

    Returns: The updated configuration object with the selected meta attributes. But please note that the configuration object is also updated in place.
    """
    config_all = Config("htc_projects/sepsis_icu/configs/meta_all.json")
    config["input/meta/attributes"] = []

    if type(attribute_names) == str:
        attribute_names = attribute_names.split("+")

    used_names = set()
    for name in attribute_names:
        # First try whether the name matches to a specific meta attribute
        match = [attribute for attribute in config_all["input/meta/attributes"] if attribute["name"] == name]
        if len(match) > 0:
            config["input/meta/attributes"] += match
            used_names.add(name)
        else:
            # Next, try whether it matches to a group
            for attribute in config_all["input/meta/attributes"]:
                attribute_group = settings_sepsis_icu.metadata_groups.get(attribute["name"], "")
                if attribute_group == name:
                    config["input/meta/attributes"].append(attribute)
                    used_names.add(name)

    assert sorted(used_names) == sorted(attribute_names), (
        f"Requested names {set(attribute_names) - set(used_names)} not found"
    )

    return config


def target_to_subgroup(target: str) -> list[list[str] | str, int]:
    """
    Maps the target to the corresponding subgroups and target dimension.

    Args:
        target: A string denoting the target of the classification, e.g., "sepsis" or "survival".

    Returns: A list containing the subgroups (e.g. ["all", "septic_shock"]) and the target dimension. Target dimension refers to the class label that is of most interest. For sepsis, this is the sepsis class (target_dim = 1), for survival, this is the non-survivors class (target_dim = 0).
    """
    subgroups = ["all"]
    if target == "sepsis":
        target_dim = 1  # select class, that is of most interest - here sepsis class
        subgroups.append("septic_shock")
    elif target == "shock":
        target_dim = 1  # select class, that is of most interest - here shock class
    elif target == "survival":
        target_dim = 0  # select class, that is of most interest - here non-survivors class

    else:
        raise ValueError("Unknown target")

    return subgroups, target_dim


def target_to_label(target: str) -> str:
    """
    Maps the target to the corresponding label.

    Args:
        target: A string denoting the target of the classification, e.g., "sepsis" or "survival".

    Returns: The corresponding label for the target.
    """
    if target == "sepsis":
        label = "sepsis_status"
    elif target == "survival":
        label = "survival_30_days_post_inclusion"
    elif target == "shock":
        label = "septic_shock"
    elif target == "septic_shock":
        label = "septic_shock"
    else:
        raise ValueError("Unknown target")

    return label


def shorten_run_name(run_dir: Path | str, target: str, wrap: bool = True) -> str:
    """
    Shortens the run name by removing the target and seed information.

    Args:
        run_dir: A Path or string of the run name that should be shortened.
        target: The target of the classification, e.g., "sepsis" or "survival".
        wrap: If True, the run name is wrapped to fit into a plot.

    Returns: The shortened run name.
    """
    if isinstance(run_dir, Path):
        run_dir = run_dir.name[20:]
    run_dir = re.sub(f"{target}-inclusion[-_]", "", run_dir)
    run_dir = run_dir.replace("seed=*_", "")

    if wrap:
        run_dir = textwrap.wrap(re.sub(r"\b", " ", run_dir), width=40)
        run_dir = "<br>".join(run_dir)
        run_dir = run_dir.replace(" ", "")

    return run_dir


def config_from_baseline_name(run_dir: str, target: str) -> Config:
    """
    Loads the Config object for the given baseline method, including the appropriate choice of the config parent file and potential selection of meta attributes.

    Args:
        run_dir: The name of the baseline method.
        target: The target of the classification, e.g., "sepsis" or "survival".

    Returns: The configuration object for the given baseline method.
    """
    if "median" in run_dir or "tpi" in run_dir:
        # Can also load meta attributes additionally
        config = Config(f"htc_projects/sepsis_icu/configs/{target}-inclusion_palm_median.json")
        if "median" in run_dir:
            config["input/feature_columns"] = ["median_normalized_spectrum"]
        else:
            config["input/feature_columns"] = []
    elif "meta" in run_dir:
        config = Config(f"htc_projects/sepsis_icu/configs/{target}-inclusion_palm_meta.json")
    else:
        raise ValueError("Unknown baseline method")

    if "tpi" in run_dir:
        for tpi_name in ["median_sto2", "median_nir", "median_thi", "median_twi"]:
            config["input/feature_columns"].append(tpi_name)

    if "meta" in run_dir:
        _, attributes = run_dir.split("@")
        config_meta_selection(config, attribute_names=attributes)

    return config


def sepsis_status_stable(path: DataPath, n_days: int = 1) -> bool:
    """
    Checks whether the sepsis status is stable for the given DataPath and the given number of days.

    Args:
        path: DataPath object for which the stability of the sepsis_status label in the upcoming days should be checked.
        n_days: The number of days that should be checked.

    Returns: True if the sepsis status is stable, False otherwise.
    """
    current_date = np.datetime64(path.datetime().date())
    subject_name = path.subject_name

    df = median_table("2022_10_24_Tivita_sepsis_ICU#subjects", table_name="recalibrated")
    mask = (
        (df["subject_name"] == subject_name)
        & (df["date"] >= current_date)
        & (df["date"] <= current_date + pd.Timedelta(days=n_days))
    )
    df = df.loc[mask]
    df = df[["date", "sepsis_status"]].drop_duplicates().reset_index(drop=True)
    # if len(df) == 1:
    #     return np.nan
    # else:
    return df["sepsis_status"].nunique() == 1


@requires_extra(_missing_library)
def save_figure(path: Path, fig: plt.Figure, **kwargs) -> None:
    fig.savefig(path, metadata={"CreationDate": None}, **kwargs)

    # Read the width of the PDF (https://stackoverflow.com/a/75801524)
    pdf_reader = PdfReader(path)
    cm_per_inch = 2.54
    points = 72
    pdf_width = float(pdf_reader.pages[0].mediabox.width) / points * cm_per_inch
    if not 18.3 < pdf_width < 18.5:
        settings.log.warning(
            f"The width of the figure {path.name} is {pdf_width:.3f} cm instead of 18.415 cm ({15 - pdf_width:.3f} missing)"
        )
