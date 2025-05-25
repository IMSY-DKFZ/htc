# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import torch
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets.config import Config as NBConfig

from htc.cpp import automatic_numpy_conversion
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils._MedianTableHelper import _MedianTableHelper
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc.utils.Task import Task


def basic_statistics(
    dataset_name: str = None,
    spec: str | Path | DataSpecification = None,
    label_mapping: LabelMapping = None,
    annotation_name: str | list[str] = None,
    paths: list[DataPath] = None,
    image_names: list[str] = None,
) -> pd.DataFrame:
    """
    Basic statistics about a dataset.

    >>> df = basic_statistics(
    ...     "2021_02_05_Tivita_multiorgan_semantic",
    ...     "pigs_semantic-only_5foldsV2.json",
    ...     label_mapping=settings_seg.label_mapping,
    ... )
    >>> print(df.head().to_string())
        annotation_name                image_name  label_index        label_name  label_valid set_type subject_name            timestamp  n_pixels
    0  semantic#primary  P041#2019_12_14_12_00_16            0        background         True    train         P041  2019_12_14_12_00_16    158786
    1  semantic#primary  P041#2019_12_14_12_00_16            4             colon         True    train         P041  2019_12_14_12_00_16     67779
    2  semantic#primary  P041#2019_12_14_12_00_16            5       small_bowel         True    train         P041  2019_12_14_12_00_16     65634
    3  semantic#primary  P041#2019_12_14_12_00_16            9           bladder         True    train         P041  2019_12_14_12_00_16     10594
    4  semantic#primary  P041#2019_12_14_12_00_16           13  fat_subcutaneous         True    train         P041  2019_12_14_12_00_16      4251

    Args:
        dataset_name: Name of the dataset (folder name on the network drive).
        spec: Name or path to a data specification file or an existing data specification object. A set_type column will be added indicating for each image whether it is part of the train, test set or not part of the specification. Please note that the specification is not used to select images.
        label_mapping: Optional label mapping which is applied to the statistics table. It will rename all labels, remove invalid labels and give the sum of pixels for the new labels (in case multiple labels like blue_cloth or metal map to the same name like background).
        annotation_name: Optional parameter. If not None, the table will only include the annotations corresponding zu the given annotation_name. Otherwise, all available annotations will be included.
        paths: List of DataPath objects which should be included in the table. Passed on to the `median_table()` function.
        image_names: List of image names which should be included in the table. Passed on to the `median_table()` function.

    Returns: Statistics in table format.
    """
    df_median = median_table(
        dataset_name=dataset_name,
        paths=paths,
        image_names=image_names,
        label_mapping=label_mapping,
        annotation_name=annotation_name,
    )
    df = df_median[["image_name", "annotation_name", "subject_name", "timestamp", "label_name", "n_pixels"]].copy()

    # Add a set_type column based on the data specification file
    if spec is not None:
        if isinstance(spec, str):
            spec = DataSpecification(spec)

        with spec.activated_test_set():
            image_names_train = [p.image_name() for p in spec.paths("^train")]
            image_names_test = [p.image_name() for p in spec.paths("^test")]
    else:
        image_names_train = []
        image_names_test = []

    set_types = []
    for image_name in df["image_name"]:
        if image_name in image_names_train:
            set_types.append("train")
        elif image_name in image_names_test:
            set_types.append("test")
        else:
            set_types.append(None)

    df["set_type"] = set_types

    if label_mapping is not None:
        df["label_index"] = df_median["label_index_mapped"]
        df["label_name"] = df_median["label_name_mapped"]
        df["label_valid"] = [label_mapping.is_index_valid(i) for i in df["label_index"]]

        # Sum together the pixels for labels with the same name
        df = df.groupby(sorted(set(df.columns.to_list()) - {"n_pixels"}), as_index=False, observed=True, dropna=False)[
            "n_pixels"
        ].sum()
        df = df.sort_values(by=["image_name", "label_index"])

    return df.reset_index(drop=True)


def median_table(
    dataset_name: str = None,
    table_name: str = "",
    paths: list[DataPath] = None,
    image_names: list[str] = None,
    label_mapping: LabelMapping = None,
    keep_mapped_columns: bool = True,
    annotation_name: str | list[str] = None,
    additional_mappings: dict[str, LabelMapping] = None,
    image_labels_column: list[dict[str, list[str] | LabelMapping]] = None,
    config: Config = None,
    sorting_kwargs: dict[str, Any] = None,
) -> pd.DataFrame:
    """
    This function is the general entry point for reading the median spectra tables. You can either read the table from a specific dataset or provide image names for which you want to have the spectra (also works if the names come from different datasets).

    >>> df = median_table(dataset_name="2021_02_05_Tivita_multiorgan_semantic")
    >>> df.iloc[0]  # doctest: +ELLIPSIS
    image_name ... P041#2019_12_14_12_01_09...
    subject_name ... P041...
    median_normalized_spectrum ... [0.0038273174, 0.0038260417, 0.0040428545, 0.0...

    Besides basic info about the image and the median spectra (`median_normalized_spectrum`), all available metadata is included in the table as well:
    >>> df.columns.to_list()
    ['image_name', 'subject_name', 'timestamp', 'label_index', 'label_name', 'median_spectrum', 'std_spectrum', 'median_normalized_spectrum', 'std_normalized_spectrum', 'n_pixels', 'median_sto2', 'std_sto2', 'median_nir', 'std_nir', 'median_twi', 'std_twi', 'median_ohi', 'std_ohi', 'median_thi', 'std_thi', 'median_tli', 'std_tli', 'image_labels', 'Camera_CamID', 'Camera_Exposure', 'Camera_analoger Gain', 'Camera_digitaler Gain', 'Camera_Speed', 'SW_Name', 'SW_Version', 'Fremdlichterkennung_Fremdlicht erkannt?', 'Fremdlichterkennung_PixelmitFremdlicht', 'Fremdlichterkennung_Breite LED Rot', 'Fremdlichterkennung_Breite LED Gruen', 'Fremdlichterkennung_Grenzwert Pixelanzahl', 'Fremdlichterkennung_Intensity Grenzwert', 'Aufnahme_Aufnahmemodus', 'camera_name', 'path', 'dataset_settings_path', 'ethics', 'annotation_name']

    This function can also be used to select specific annotations, either globally per dataset:
    >>> df = median_table(dataset_name="2021_02_05_Tivita_multiorgan_semantic", annotation_name="semantic#intra1")
    >>> df["annotation_name"].unique().tolist()
    ['semantic#intra1']

    or individually per image:
    >>> df = median_table(
    ...     image_names=["P091#2021_04_24_12_02_50@polygon#annotator1&polygon#annotator2", "P041#2019_12_14_12_01_39"]
    ... )
    >>> sorted(df["annotation_name"].unique())
    ['polygon#annotator1', 'polygon#annotator2', 'semantic#primary']

    Note: In the original table, one row denotes one label of one image from one annotator which also corresponds to the default of this function since the default annotation is used (similar to DataPath.read_segmentation()). If more than one annotation name is requested, a row is unique by its image_name, label_name and annotation_name.

    Args:
        dataset_name: Name of the dataset from which you want to have the median spectra table. The name may include a # to specify a subdataset, e.g. `2021_02_05_Tivita_multiorgan_semantic#context_experiments` for the context_experiments folder inside the semantic data directory. If a dataset consists only of subdatasets (e.g., 2022_10_24_Tivita_sepsis_ICU), it is also possible to use the name of the main dataset to get all tables from the subdatasets (e.g., 2022_10_24_Tivita_sepsis_ICU to get 2022_10_24_Tivita_sepsis_ICU#calibrations + 2022_10_24_Tivita_sepsis_ICU#subjects).
        table_name: For each dataset, there may be multiple tables for different purposes (e.g. tables with recalibrated data). With this switch, you specify which table should be loaded. The format of these tables on disk is `dataset_name@table_name@median_spectra@annotation_name.feather`. Per default, the normal table with the original data is loaded corresponding to tables on disk with the format `dataset_name@median_spectra@annotation_name.feather`, i.e. without the optional `@table_name`. Requested image names (`image_names` argument) are only considered from the tables matching the given `table_name`. It is not possible to select images from tables with different table names with this function since they may contain the same images.
        paths: List of DataPath objects from which you want to have the median spectra. If annotation names are specified with a data path object, those names will be used. If specified, image_names must be None.
        image_names: List of image names to search for (similar to the paths parameter). Image names may also include annotation names (e.g. subject#timestamp@name1&name2). It is not ensured that the resulting table contains all requested images because some images may lack annotations or are filtered out by the label_mapping. If specified, paths must be None.
        label_mapping: The target label mapping. There will be a new label_index_mapped column (and a new label_name_mapped column with the new names defined by the mapping) and the old label_index column will be removed (since the label_index is not unique across datasets). Only valid labels will be included in the resulting table. If set to None, then mapping is not carried out.
        keep_mapped_columns: If True, the columns label_index_mapped and label_name_mapped are kept in the table. If False, they are removed and replace the label_index and label_name columns. This parameter has no effect if no label mapping is given.
        annotation_name: Unique name of the annotation(s) for cases where multiple annotations exist (e.g. inter-rater variability). If None, will use the default from the dataset. If the dataset does not have a default (i.e. the annotation_name_default is missing in the dataset_settings.json file), all annotations are returned. It is also possible to explicitly retrieve all annotations by setting this parameter to 'all'.
        additional_mappings: Additional label mappings for other columns. The keys are the column names and the values are the LabelMapping objects for the respective columns. For each specified column, a new column with _index appended will be added.
        image_labels_column: Specify how multiple columns should be mapped into one `image_labels` column indicating one or more image labels. Each entry in the list specifies one dimension in the `image_labels` columns and the dictionary contains information from which columns values should be mapped from. It is possible to map values from different columns to one image label and to have multiple image labels. The specification is similar to `input/image_labels` in the config file. See tests for examples.
        config: Load median spectra based on the settings of the config. This can be used to automatically retrieve common options (e.g., label_mapping) which otherwise have to be passed to this function. If no dataset_name, paths or image_names is given, the data specification is loaded from the config object and all non-test paths are used. Options passed as arguments have precedence over the config options.
        sorting_kwargs: Keyword arguments which are passed on to the `sort_labels()` function. This can be used to control the sorting behavior, e.g., to use different columns for sorting.

    Returns: Median spectra data frame. The table is always sorted via the `sort_labels()` function to ensure our default label sorting for tables is used.
    """
    if additional_mappings is None:
        additional_mappings = {}
    if sorting_kwargs is None:
        sorting_kwargs = {}

    if config is not None:
        if table_name == "":
            table_name = config.get("input/table_name", "")
        if dataset_name is None and paths is None and image_names is None and config["input/data_spec"]:
            spec = DataSpecification.from_config(config)
            paths = spec.paths()
        if label_mapping is None and config["label_mapping"]:
            label_mapping = LabelMapping.from_config(config, task=Task.SEGMENTATION)
        if annotation_name is None:
            annotation_name = config.get("input/annotation_name", None)
        if image_labels_column is None and config["input/image_labels"]:
            image_labels_column = config["input/image_labels"]

            # Make sure the label mapping objects are created
            for image_label_entry_index, data in enumerate(image_labels_column):
                data["image_label_mapping"] = LabelMapping.from_config(
                    config, task=Task.CLASSIFICATION, image_label_entry_index=image_label_entry_index
                )

    df = _MedianTableHelper(
        dataset_name=dataset_name,
        table_name=table_name,
        paths=paths,
        image_names=image_names,
        label_mapping=label_mapping,
        keep_mapped_columns=keep_mapped_columns,
        annotation_name=annotation_name,
        additional_mappings=additional_mappings,
        image_labels_column=image_labels_column,
    )()

    # Sorting is done here to avoid cyclic imports
    df = sort_labels(df, **sorting_kwargs)
    return df


def add_times_table(df: pd.DataFrame, groups: list[str] = None) -> None:
    """
    Adds a column "time" to the table with the timestamp converted to a datetime object. If groups is given, another "rel_time" column is added which contains the relative time (in seconds) within each grouping (e.g. time for all images of one subject relative to the first image `groups=["subject_name"]`).

    Args:
        df: The table to add the columns to (in-place).
        groups: A list of column names for grouping of the relative time.
    """
    if groups is None:
        groups = ["subject_name"]

    df["time"] = pd.to_datetime(df["timestamp"], format=settings.tivita_timestamp_format)

    if groups is not None:
        df_group_times = df.groupby(groups)["time"].min()

        rel_times = []
        for _, row in df.iterrows():
            rel_times.append(row["time"] - df_group_times.loc[tuple([row[g] for g in groups])])
        df["rel_time"] = rel_times


def group_median_spectra(df: pd.DataFrame, additional_columns: list[str] = None) -> pd.DataFrame:
    """
    Groups median spectra per subject by averaging all median spectra from that subject.

    Args:
        df: Table with the median spectra.
        additional_columns: List of additional columns to include in the resulting table. Note: these values must be unique per pig and organ.

    Returns: Table with the averaged median spectra per pig and organ.
    """
    if additional_columns is None:
        additional_columns = []

    assert {
        "subject_name",
        "label_name",
    }.issubset(set(df.columns))
    rows = []

    for subject_name in df["subject_name"].unique():
        df_subject = df.query("subject_name == @subject_name")
        for label in df_subject["label_name"].unique():
            df_label = df_subject.query("label_name == @label")
            if len(df_label) > 0:
                current_row = {
                    "subject_name": subject_name,
                    "label_name": label,
                }

                for c in df.columns:
                    if "spectrum" in c and "median" in c:
                        current_row[c] = np.mean(np.stack(df_label[c]), axis=0)
                    elif "spectrum" in c and "std" in c:
                        std_spectrum = np.stack(df_label[c])
                        std_spectrum = std_spectrum[~np.any(np.isnan(std_spectrum), axis=1)]
                        std_spectrum = np.mean(std_spectrum, axis=0)
                        current_row[c] = std_spectrum

                for c in additional_columns:
                    values = np.asarray(df_label[c].unique())
                    assert len(values) == 1, (
                        f"The additional column {c} has more than one value ({subject_name = }, {label = }): {values}"
                    )
                    current_row[c] = values.item()

                rows.append(current_row)

    return pd.DataFrame(rows)


def run_info(run_dir: Path) -> dict:
    config = Config(run_dir / "config.json")

    model_name = run_dir.parent.name
    if model_name == "patch":
        model_name = f"{model_name}_{config['input/patch_size'][0]}"

    if "parameters" in run_dir.name:
        model_type = "param"
    elif "rgb" in run_dir.name:
        model_type = "rgb"
    else:
        model_type = "hsi"

    return {
        "model_name": model_name,
        "model_type": model_type,
        "config": config,
    }


def load_util_log(folder: Path) -> dict:
    log_path = list(folder.glob("system_log*.json"))
    assert len(log_path) == 1, f"Could not find the utilization log file for the folder {folder}"
    log_path = log_path[0]

    with log_path.open() as f:
        data = json.load(f)

    n_different_gpus = len(np.unique([len(v) for v in data["gpus_load"]]))
    if n_different_gpus != 1:
        # It can happen that for some timepoints GPU values are missing. In this case, the list may be empty
        # We fill the list with np.nan and then interpolate the missing values
        # Note: This only works for one GPU
        gpu_data = np.array([[v[0] if len(v) > 0 else np.nan] for v in data["gpus_load"]])
        gpu_data_mask = np.isnan(gpu_data)
        gpu_data[gpu_data_mask] = np.interp(
            np.flatnonzero(gpu_data_mask), np.flatnonzero(~gpu_data_mask), gpu_data[~gpu_data_mask]
        )

        data["gpus_load"] = gpu_data.tolist()
        gpus_load = data["gpus_load"]
    else:
        # Find the used GPU based on the load
        all_gpus_load = np.array(data["gpus_load"]).mean(axis=0)
        gpu_indices = np.where(all_gpus_load > 0.1)[
            0
        ].tolist()  # Remove GPUs with very low utilization (probably from a system with 2 GPUs)
        if len(gpu_indices) == 0:
            settings.log.warning(
                f"No GPU with a utilization of at least 0.1 found (log_path={log_path}, all_gpus_load={all_gpus_load})"
            )

        gpus_load = np.array(data["gpus_load"])[:, gpu_indices]

    # Average over time
    gpu_load_mean = np.mean(gpus_load, axis=1)  # Average across all used GPUs
    cpu_load_mean = np.mean(np.array(data["cpus_load"]), axis=1)  # Average across all CPUs

    if len(cpu_load_mean) > len(gpu_load_mean):
        # Some runs (on the cluster) are broken and cannot measure the GPU over the full training time --> append with nans in this case
        gpu_load_mean = np.append(gpu_load_mean, np.zeros(len(cpu_load_mean) - len(gpu_load_mean)) + np.nan)

    assert gpu_load_mean.shape == cpu_load_mean.shape, "Different shape for GPU and CPU data"

    return {"gpu_load_mean": gpu_load_mean, "cpu_load_mean": cpu_load_mean, "raw_data": data}


def utilization_table(run_dir: Path) -> pd.DataFrame:
    rows = []
    fold_dirs = sorted(run_dir.glob("fold*"))
    for fold_dir in fold_dirs:
        event_file = list(fold_dir.glob("events.out.tfevents.*"))
        assert len(event_file) == 1, "There must be exactly one event file per fold"
        event_file = event_file[0]

        node_name = re.search(r"^events\.out\.tfevents\.\d+\.([^.]+)", event_file.name).group(1)

        data = load_util_log(fold_dir)
        time = np.array(data["raw_data"]["time"]) - data["raw_data"]["time"][0]
        time /= 3600
        gpu_mean = np.mean(data["gpu_load_mean"])
        gpu_std = np.std(data["gpu_load_mean"])
        cpu_mean = np.mean(data["cpu_load_mean"])
        cpu_std = np.std(data["cpu_load_mean"])

        rows.append([fold_dir.name, node_name, time[-1], gpu_mean, gpu_std, cpu_mean, cpu_std])

    return pd.DataFrame(
        rows, columns=["fold", "node", "hours", "gpu_util_mean", "gpu_util_std", "cpu_util_mean", "cpu_util_std"]
    )


def sort_labels(
    storage: np.ndarray | list | set | dict | pd.DataFrame,
    label_ordering: dict[str, str | int] | list[str] = None,
    sorting_cols: list[str] = None,
    dataset_name: str = None,
) -> np.ndarray | list | dict | pd.DataFrame:
    """
    Sort the organs in the storage according to the label ordering of the surgeons.

    >>> sort_labels(["colon", "stomach"])
    ['stomach', 'colon']
    >>> sort_labels(["a", "b"], label_ordering={"a": 2, "b": 1})
    ['b', 'a']

    If no ordering information is available, labels are sorted alphabetically:
    >>> sort_labels(["b", "a"])
    ['a', 'b']

    Args:
        storage: The storage to sort: numpy array, list, dict or dataframe. If dataframe, it will sort by label_name, image_name and annotation_name (if available).
        label_ordering: Alternative sort order for the labels. Either a mapping which defines a key for each label and something sortable as values (e.g. integer values) or a list of label names in the sorting order.
        sorting_cols: Explicit list of columns which should be used to sort the dataframe. If None, will sort by label_name, image_name (if available) and annotation_name (if available).
        dataset_name: Name of a dataset which is accessible via settings.data_dirs and which contains a dataset settings with a defined label ordering.

    Returns: The sorted storage.
    """
    if type(label_ordering) == list:
        label_ordering = {label: i for i, label in enumerate(label_ordering)}

    if label_ordering is None and dataset_name is not None:
        dsettings = DatasetSettings(settings.data_dirs[dataset_name])
        label_ordering = dsettings.get("label_ordering", None)

    if label_ordering is None:
        # The masks dataset has a very comprehensive list of label order, try to use this as first default
        dsettings = DatasetSettings(settings.data_dirs.masks)
        label_ordering = dsettings.get("label_ordering", None)

    if label_ordering is None:
        # Last option, check every available dataset
        for _, entry in settings.datasets:
            dsettings = DatasetSettings(entry["path_data"])
            label_ordering = dsettings.get("label_ordering", None)
            if label_ordering is not None:
                break

    if label_ordering is None:
        settings.log.warning("Could not find a label ordering. Storage remains unsorted")
        return storage

    # 9999_ unknown labels are sorted alphabetically after the known labels
    if type(storage) == dict:
        storage = dict(sorted(storage.items(), key=lambda pair: label_ordering.get(pair[0], f"9999_{pair[0]}")))
    elif type(storage) == list or type(storage) == np.ndarray or type(storage) == set:
        storage = sorted(storage, key=lambda element: label_ordering.get(element, f"9999_{element}"))
    elif type(storage) == pd.DataFrame:
        sorter = lambda col: [label_ordering.get(v, f"9999_{v}") for v in col] if col.name == "label_name" else col
        if sorting_cols is None:
            sorting_cols = ["label_name"] if "label_name" in storage.columns else []
            if "image_name" in storage:
                sorting_cols.append("image_name")
            if "annotation_name" in storage:
                sorting_cols.append("annotation_name")

        if len(sorting_cols) > 0:
            storage = storage.sort_values(by=sorting_cols, key=sorter, ignore_index=True)
    else:
        settings.log.warning(f"Unsupported input type: {type(storage)}")

    return storage


@automatic_numpy_conversion
def sort_labels_cm(
    cm: torch.Tensor | np.ndarray, cm_order: list[str], target_order: list[str]
) -> torch.Tensor | np.ndarray:
    """
    Sorts the rows/columns in a cm to a target order.

    >>> cm = np.array([[0, 10, 3], [1, 2, 3], [8, 6, 4]])
    >>> cm_order = ["b", "a", "c"]
    >>> target_order = ["a", "b", "c"]
    >>> sort_labels_cm(cm, cm_order, target_order)
    array([[ 2,  1,  3],
           [10,  0,  3],
           [ 6,  8,  4]])

    Args:
        cm: Confusion matrix.
        cm_order: Name of the current rows/columns in the confusion matrix.
        target_order: Name of the new ordering of the confusion matrix.

    Returns: Sorted confusion matrix (based on the target order).
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1], "cm must be square"
    assert len(cm_order) == len(target_order) and set(cm_order) == set(target_order), (
        "The same names must occur in the cm and the target order"
    )
    assert sorted(set(cm_order)) == sorted(cm_order), "The names must be unique"

    # Swap rows
    switched_cm = torch.zeros_like(cm)
    ordering_indices = [cm_order.index(l) for l in target_order]
    for i, idx in enumerate(ordering_indices):
        switched_cm[i, :] = cm[idx, :]

    # Swap columns
    switched_cm_final = torch.zeros_like(cm)
    for j, idx in enumerate(ordering_indices):
        switched_cm_final[:, j] = switched_cm[:, idx]

    return switched_cm_final


def execute_notebook(
    notebook_path: Path, output_path: Path = None, parameters: dict[str, Any] = None, html_only: bool = True
) -> None:
    """
    Runs the given notebook and stores it in a new location. Additionally, a compressed HTML version of the notebook can be stored at the output location.

    Note: This function provides similar functionality to papermill (https://papermill.readthedocs.io/en/latest/).

    Args:
        notebook_path: Path to the base notebook (e.g. ExperimentAnalysis.ipynb).
        output_path: Path to the output file. If a directory, the same name as the notebook will be used. If None, the notebook is only executed without saving it.
        parameters: Dictionary with (key, value) pairs denoting the parameters for the notebook. Please add a cell to your notebook with the tag `parameters` and add your defaults there. A new cell will be inserted after the tagged cell so that the new values instead of the defaults are used. If None, then the notebook is not changed.
        html_only: If True, store only the HTML file and no notebook at the output location. Note that the notebook is stored uncompressed and can hence be much larger than the HTML file so this option is useful to save some space.
    """
    from htc.utils.visualization import compress_html

    nb = nbformat.read(notebook_path, as_version=nbformat.NO_CONVERT)

    if parameters is not None:
        # Create a new cell with the parameters
        src = "# Injected parameters\n"
        src += "\n".join([f"{k} = {v!r}" for k, v in parameters.items()])
        parameters_cell = nbformat.v4.new_code_cell(src)

        # We need to insert the parameter cell after the existing parameter cell so that we overwrite existing defaults
        insertion_index = None

        for i, cell in enumerate(nb.cells):
            if "tags" in cell["metadata"] and "parameters" in cell["metadata"]["tags"]:
                insertion_index = i
                break
        assert insertion_index is not None
        insertion_index += 1

        nb.cells = [*nb.cells[:insertion_index], parameters_cell, *nb.cells[insertion_index:]]

    # Notebook execution docs: https://nbclient.readthedocs.io/en/latest/client.html
    # Exporter docs: https://nbconvert.readthedocs.io/en/latest/nbconvert_library.html
    c = NBConfig()
    execution_dir = notebook_path.parent
    c.HTMLExporter.preprocessors = [ExecutePreprocessor(timeout=600, resources={"metadata": {"path": execution_dir}})]
    exporter = HTMLExporter(config=c)

    # Run the notebook
    (html, _) = exporter.from_notebook_node(nb)

    if output_path is not None:
        if output_path.is_dir():
            output_path = output_path / notebook_path.name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save everything
        compress_html(output_path.with_suffix(".html"), html)
        if not html_only:
            nbformat.write(exporter.preprocessors[0].nb, output_path)


def get_valid_run_dirs(training_dir: Path = None) -> list[Path]:
    # If a run folder starts with one of the following prefixes, it should be ignored
    # The dataset size experiment is very special and does not have all the required files
    excluded_prefixes = ("running", "test", "special", "error", settings_seg.dataset_size_timestamp)

    # The benchmarking runs are also special and don't need aggregated tables
    excluded_names = ("benchmarking",)

    run_dirs = []
    if training_dir is None:
        training_dir = settings.training_dir
    for run_dir in sorted(training_dir.glob("*/*")):
        if settings.datasets.network in run_dir.parents:
            continue
        if not run_dir.is_dir():
            continue
        if run_dir.stem.startswith(excluded_prefixes):
            continue
        if any(n in run_dir.name for n in excluded_names):
            continue

        run_dirs.append(run_dir)

    if len(run_dirs) == 0:
        settings.log.warning("Could not find any valid run directories")

    return sorted(run_dirs)


def checkpoint_path(fold_dir: Path) -> tuple[Path, int]:
    """
    Searches for the checkpoint path.

    Args:
        fold_dir: Path to the fold directory.

    Returns: Tuple with the found checkpoint path and the index of the best epoch (either from the checkpoint name or from the validation table).
    """
    ckpt_file = list(fold_dir.glob("*.ckpt"))
    assert len(ckpt_file) == 1, f"Could not find the checkpoint file in {fold_dir}"
    ckpt_file = ckpt_file[0]

    if ckpt_file.stem == "last":
        # Find the last epoch (which is also the best in this case)
        table_path = fold_dir / "validation_results.pkl.xz"
        if table_path.exists():
            df = pd.read_pickle(table_path)
            best_epoch_index = df["epoch_index"].max()
        else:
            # The validation_results table is not always available (e.g., pretrained models) but the aggregated validation_table is
            table_path = fold_dir.parent / "validation_table.pkl.xz"
            df = pd.read_pickle(table_path)
            best_epoch_index = df[df.fold_name == fold_dir.name]["epoch_index"].max()
    else:
        # Find the best epoch id from the checkpoint name
        match = re.search(r"epoch=(\d+)", str(ckpt_file))
        assert match is not None, f"Could not extract the best epoch_index from the checkpoint name ({ckpt_file})"
        best_epoch_index = int(match.group(1))

    return ckpt_file, best_epoch_index


def get_nsd_thresholds(mapping: LabelMapping, aggregation_method: str = None, name: str = "semantic") -> list[float]:
    """
    Load precomputed NSD thresholds from a file.

    Args:
        mapping: Label mapping of the training run which is used to make a selection of labels.
        aggregation_method: Aggregation method (e.g. mean). Must correspond to a column name in the table.
        name: Name of the table (e.g. semantic for the MIA2022 thresholds).

    Returns: Tolerance value for each class in the order defined in the label mapping.
    """
    if (
        settings.results_dir is not None
        and (table_path := settings.results_dir / "rater_variability" / f"nsd_thresholds_{name}.csv").exists()
    ):
        df = pd.read_csv(table_path)
    else:
        # If not available, try to download the file
        df = pd.read_csv(f"https://e130-hyperspectal-tissue-classification.s3.dkfz.de/models/nsd_thresholds_{name}.csv")
    tolerance_column = settings_seg.nsd_aggregation.split("_")[-1] if aggregation_method is None else aggregation_method
    tolerance_column = f"tolerance_{tolerance_column}"

    tolerances = []
    for i in range(len(mapping)):
        name = mapping.index_to_name(i)
        tolerances.append(df.query("label_name == @name")[tolerance_column].item())

    return tolerances
