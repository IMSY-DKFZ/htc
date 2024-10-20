# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import configparser
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from htc.settings import settings
from htc.tivita.DataPath import DataPath


def generate_metadata_table(paths: Iterable[DataPath]) -> pd.DataFrame:
    """
    Collects information from all *meta.log and (if available) .xml files from all images and generates a table will all the metadata per image.

    Note: This function always adds the camera name to the resulting table.

    Args:
        paths: list of all image folders to scan.

    Returns: Table with the metadata (each image corresponds to a row).
    """
    rows = []

    for path in paths:
        current_row = {"image_name": path.image_name()}
        try:
            current_row |= path.image_name_typed()
        except NotImplementedError:
            pass

        if (meta := path.read_camera_meta()) is not None:
            current_row |= meta
        if (pat_meta := path.read_patient_meta()) is not None:
            current_row |= pat_meta
        if "Camera_CamID" not in current_row:
            current_row["Camera_CamID"] = "unknown"

        # Always set the camera_name
        meta_labels = path.read_annotation_meta()
        if meta_labels is not None and "camera_name" in meta_labels:
            # Use the camera name if it is explicitly set
            current_row["camera_name"] = meta_labels["camera_name"]
        else:
            cam = current_row["Camera_CamID"]
            if "camera_name_changes" in path.dataset_settings and cam in path.dataset_settings["camera_name_changes"]:
                cam_change_info = path.dataset_settings["camera_name_changes"][cam]
                time_image = path.datetime()
                time_change = datetime.strptime(cam_change_info["change_date"], settings.tivita_timestamp_format)
                suffix = (
                    cam_change_info["suffix_before"] if time_image < time_change else cam_change_info["suffix_after"]
                )
            else:
                # We assume a correct yellow filter per default
                suffix = "correct-1"

            current_row["camera_name"] = f"{cam}_{suffix}"

        rows.append(current_row)

    return pd.DataFrame(rows)


def read_meta_file(path: Path) -> dict:
    """
    Read the values of the meta log file of an image.

    >>> from htc.settings import settings
    >>> path = Path(settings.data_dirs.semantic / "subjects/P041/2019_12_14_12_00_16/2019_12_14_12_00_16_meta.log")
    >>> read_meta_file(path)["Camera_CamID"]
    '0102-00085'

    If possible, it is recommended to use the DataPath class instead of this (low-level) function:
    >>> from htc.tivita.DataPath import DataPath
    >>> path = DataPath.from_image_name("P041#2019_12_14_12_00_16")
    >>> path.meta("Camera_CamID")
    '0102-00085'

    Args:
        path: Path to the meta log file (e.g. "[...]/2020_07_20_18_17_26/2020_07_20_18_17_26_meta.log").

    Returns: Dictionary with all meta data or None if the image does not have a meta file (can happen for older images).
    """
    assert path.exists() and path.is_file(), f"Meta file {path} does not exist or is not a file"

    config = configparser.ConfigParser()
    config.optionxform = str  # Avoid automatic lowercase transformation
    config.read(path, encoding="cp1252")  # ANSI encoding

    values = {}
    for section in config:
        for key in config[section]:
            value = config[section][key].strip('"')  # Remove quotes from strings: "Reflektanz" --> Reflektanz
            if re.search(r"\d+,\d+", value) is not None:
                value = value.replace(",", ".")  # 7,000000 --> 7.000000
            if key == "Fremdlicht erkannt?":
                value = config[section].getboolean(key)

            # Convert strings to numbers if possible
            try:
                value = pd.to_numeric(value)
            except Exception:
                pass

            values[f"{section}_{key}"] = value

    return values


def read_meta_patient(path: Path) -> dict:
    """
    Read the patient meta file (xml file) which contains meta information about the patient if provided during image acquisition in the Tivita system.

    >>> from htc.settings import settings
    >>> path = Path(
    ...     settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13" / "calibration_white.xml"
    ... )
    >>> meta = read_meta_patient(path)
    >>> meta["PatientID"]
    'calibration_white'

    If possible, it is recommended to use the DataPath class instead of this (low-level) function:
    >>> from htc.tivita.DataPath import DataPath
    >>> path = DataPath(settings.data_dirs.studies / "2022_09_29_Surgery2_baseline" / "2022_09_29_17_04_13")
    >>> path.meta("PatientID")
    'calibration_white'

    Args:
        path: Path to the xml file.

    Returns: Dictionary with all the meta information in the file. Keys correspond to the name attribute of the element tags and dictionary values correspond to the element tag values.
    """
    assert path.exists() and path.is_file(), f"Patient meta file {path} does not exist or is not a file"

    file = ET.parse(path)
    root = file.getroot()
    meta = {}
    for element in root:
        meta[element.attrib["name"]] = element.text

    return meta
