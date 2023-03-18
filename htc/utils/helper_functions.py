# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import re
import warnings
from pathlib import Path
from typing import Any, Union

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
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping


def basic_statistics(
    dataset_name: str,
    specs_name: str = None,
    label_mapping: LabelMapping = None,
    annotation_name: Union[str, list[str]] = None,
) -> pd.DataFrame:
    """
    Basic statistics about a dataset.

    >>> df = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json", label_mapping=settings_seg.label_mapping)
    >>> print(df.head().to_string())
                     image_name  label_index   label_name  label_valid set_type subject_name            timestamp  n_pixels
    0  P041#2019_12_14_12_00_16            0   background         True    train         P041  2019_12_14_12_00_16    158786
    1  P041#2019_12_14_12_00_16            4        colon         True    train         P041  2019_12_14_12_00_16     67779
    2  P041#2019_12_14_12_00_16            5  small_bowel         True    train         P041  2019_12_14_12_00_16     65634
    3  P041#2019_12_14_12_00_16            9      bladder         True    train         P041  2019_12_14_12_00_16     10594
    4  P041#2019_12_14_12_00_16           13          fat         True    train         P041  2019_12_14_12_00_16      4251

    Args:
        dataset_name: Name of the dataset (folder name on the network drive).
        specs_name: Name or path to a data specification file. A set_type column will be added indicating for each image whether it is part of the train or test set.
        label_mapping: Optional label mapping which is applied to the statistics table. It will rename all labels, remove invalid labels and give the sum of pixels for the new labels (in case multiple labels like blue_cloth or metal map to the same name like background).
        annotation_name: Optional parameter. If not None, the table will only include the annotations corresponding zu the given annotation_name. Otherwise, all available annotations will be included.

    Returns: Statistics in table format.
    """
    df = median_table(dataset_name=dataset_name, annotation_name=annotation_name)
    df = df[["image_name", "subject_name", "timestamp", "label_name", "n_pixels"]]

    # Add a set_type column based on the data specification file
    if specs_name is not None:
        specs = DataSpecification(specs_name)
        specs.activate_test_set()

        image_names_train = [p.image_name() for p in specs.paths("^train")]
        image_names_test = [p.image_name() for p in specs.paths("^test")]
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
        # Apply label mapping and group together labels with the same name (n_pixels will be summed up)
        mapping_dataset = LabelMapping.from_data_dir(settings.data_dirs[dataset_name])
        df["label_index"] = [mapping_dataset.name_to_index(l) for l in df["label_name"]]
        df["label_index"] = label_mapping.map_tensor(df["label_index"].values, mapping_dataset)

        df["label_valid"] = [label_mapping.is_index_valid(i) for i in df["label_index"]]
        df["label_name"] = [label_mapping.index_to_name(i) for i in df["label_index"]]

        df = df[df["label_valid"]]
        df = df.groupby(sorted(set(df.columns.to_list()) - {"n_pixels"}), as_index=False)["n_pixels"].sum()
        df = df.sort_values(by=["image_name", "label_index"])

    return df.reset_index(drop=True)


def median_table(
    dataset_name: str = None,
    image_names: list[str] = None,
    label_mapping: LabelMapping = None,
    annotation_name: Union[str, list[str]] = None,
) -> pd.DataFrame:
    """
    This function is the general entry point for reading the median spectra tables. You can either read the table from a specific dataset or provide image names for which you want to have the spectra (also works if the names come from different datasets).

    Note: In the original table, one row denotes one label of one image from one annotator but the default of this function is to return only the default annotation (similar to DataPath.read_segmentation()).

    Args:
        dataset_name: Name of the dataset from which you want to have the median spectra table.
        image_names: List of image ids to search for.
        label_mapping: The target label mapping. There will be a new label_index_mapped column (and a new label_name_mapped column with the new names defined by the mapping) and the old label_index column will be removed (since the label_index is not unique across datasets). If set to None, then mapping is not carried out.
        annotation_name: Unique name of the annotation(s) for cases where multiple annotations exist (e.g. inter-rater variability). If None, will use the default from the dataset. If the dataset does not have a default (i.e. the annotation_name_default is missing in the dataset_settings.json file), all annotations are returned. It is also possible to explicitly retrieve all annotations by setting this parameter to 'all'.

    Returns: Median spectra data frame. The table is either sorted by image names (if image_names is not None) or by the sort_labels() function (if dataset_name is used).
    """
    tables = {}
    for path in sorted((settings.intermediates_dir / "tables").glob("*median_spectra*.feather")):
        parts = path.stem.split("@")
        assert 2 <= len(parts) <= 3
        if len(parts) == 2:
            _dataset_name, _table_type = path.stem.split("@")
            _annotation_name = None
        else:
            _dataset_name, _table_type, _annotation_name = path.stem.split("@")
        assert _table_type == "median_spectra"

        if _dataset_name not in tables:
            tables[_dataset_name] = {}
        tables[_dataset_name][_annotation_name] = path

    def read_table(dataset_name: str, annotation_name: Union[str, list[str], None]) -> pd.DataFrame:
        # Find the default annotation_name
        if annotation_name is None:
            data_dir = settings.data_dirs[dataset_name]
            if data_dir is not None:
                dsettings = DatasetSettings(data_dir)
                annotation_name = dsettings.get("annotation_name_default")

        if annotation_name is None or annotation_name == "all":
            annotation_name = list(tables[dataset_name].keys())

        if type(annotation_name) == str:
            annotation_name = [annotation_name]

        df = []
        for name in annotation_name:
            df_a = pd.read_feather(tables[dataset_name][name])
            if name is not None:
                df_a["annotation_name"] = name
            else:
                assert len(annotation_name) == 1
            df.append(df_a)

        needs_sorting = len(df) > 1
        df = pd.concat(df)

        if len(df) > 0 and label_mapping is not None:
            # Mapping from path to config (the mapping depends on the dataset and must be done separately)
            df_mapping = df.query("label_name in @label_mapping.label_names(all_names=True)").copy()
            if len(df_mapping) > 0:
                df = df_mapping
                label_indices = torch.from_numpy(df["label_index"].values)
                assert (
                    settings.data_dirs[dataset_name] is not None
                ), f"Cannot find the path to the dataset {dataset_name} but this is required for remapping the labels"
                original_mapping = LabelMapping.from_data_dir(settings.data_dirs[dataset_name])
                label_mapping.map_tensor(label_indices, original_mapping)
                df["label_index_mapped"] = label_indices
                df["label_name_mapped"] = [label_mapping.index_to_name(i) for i in df["label_index_mapped"]]

        if needs_sorting:
            df = sort_labels(df, dataset_name=dataset_name)

        return df.reset_index(drop=True)

    if dataset_name is not None:
        return read_table(dataset_name, annotation_name)

    assert image_names is not None, "image_names must be supplied if dataset_names is None"
    dfs = []
    image_names = pd.unique(image_names)  # Unique without sorting
    remaining_images = set(image_names)
    considered_tables = []

    for dataset_name in tables.keys():
        df = read_table(dataset_name, annotation_name)
        df = df.query("image_name in @remaining_images")

        if len(df) > 0:
            dfs.append(df)
            remaining_images = remaining_images - set(df["image_name"].values)
            considered_tables.append(dataset_name)

            if len(remaining_images) == 0:
                # We already have all image_names, we can stop looping over the tables
                break

    assert len(dfs) > 0, (
        f"Could not find all the requested images ({remaining_images = }). This could mean that some of the"
        " intermediate files are missing or that you do not have access to them (e.g. human data)"
    )
    with warnings.catch_warnings():
        # The same columns might have different dtypes in the dataframes depending on missing values
        warnings.filterwarnings("ignore", message=".*object-dtype columns with all-bool values", category=FutureWarning)
        df = pd.concat(dfs)
    if len(dfs) > 1:
        # label_index is potentially incorrect when paths from multiple datasets are used, so it is safer to remove it
        df.drop(columns="label_index", inplace=True)

    # Same order as defined by the paths
    df["image_name"] = df["image_name"].astype("category")
    df["image_name"] = df["image_name"].cat.set_categories(image_names)
    df.sort_values("image_name", inplace=True, ignore_index=True)

    # Make sure we have all requested image_names (it is possible that some image_names are missing if they contain only labels which were filtered out by the label mapping)
    image_names_df = set(df["image_name"].unique())
    assert image_names_df.issubset(image_names), (
        "Could not find all image_names in the median spectra tables. Please make sure that the median table exists for"
        " every dataset where the image_names come from"
    )

    if label_mapping is not None:
        assert set(df["label_index_mapped"].values).issubset(
            set(label_mapping.label_indices())
        ), "Found at least one label_index which is not part of the mapping"
    if len(image_names_df) < len(image_names):
        settings.log.warning(
            f"{len(image_names) - len(image_names_df)} image_names are not used because they were filtered out (e.g. by"
            f" the label mapping). The following tables were considered: {considered_tables}"
        )

    return df


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
                    values = df_label[c].unique()
                    assert (
                        len(values) == 1
                    ), f"The additional column {c} has more than one value ({subject_name = }, {label = }): {values}"
                    current_row[c] = values.item()

                rows.append(current_row)

    return pd.DataFrame(rows)


def run_info(run_dir: Path) -> dict:
    config = Config(run_dir / "config.json")

    model_name = run_dir.parent.name
    if model_name == "patch":
        model_name = f'{model_name}_{config["input/patch_size"][0]}'

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
        # This is weird. Different number of GPUs measured over time --> just take the first measurement
        data["gpus_load"] = [[v[0]] for v in data["gpus_load"] if len(v) > 0]  # Fix the data
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
    storage: Union[np.ndarray, list, set, dict, pd.DataFrame],
    label_ordering: dict[str, Union[str, int]] = None,
    sorting_cols: list[str] = None,
    dataset_name: str = None,
) -> Union[np.ndarray, list, dict, pd.DataFrame]:
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
        label_ordering: Alternative sort order for the labels. The mapping must define a key for each label and something sortable as values (e.g. integer values).
        sorting_cols: Explicit list of columns which should be used to sort the dataframe. If None, will sort by label_name, image_name (if available) and annotation_name (if available).
        dataset_name: Name of a dataset which is accessible via settings.data_dirs and which contains a dataset settings with a defined label ordering.

    Returns: The sorted storage.
    """
    if label_ordering is None and dataset_name is not None:
        dsettings = DatasetSettings(settings.data_dirs[dataset_name])
        label_ordering = dsettings.get("label_ordering", None)

    if label_ordering is None:
        # The masks dataset has a very comprehensive list of label order, try to use this as first default
        dsettings = DatasetSettings(settings.data_dirs.masks)
        label_ordering = dsettings.get("label_ordering", None)

    if label_ordering is None:
        # Last option, check every available dataset
        for _, entry in settings.data_dirs:
            dsettings = DatasetSettings(entry["path_data"])
            label_ordering = dsettings.get("label_ordering", None)
            if label_ordering is not None:
                break

    if label_ordering is None:
        settings.log.warning("Could not find a label ordering. Storage remains unsorted")
        return storage

    # 9999_ unknown labels are sorted alphabetically after the known labels
    if type(storage) == dict:
        storage = sorted(storage.items(), key=lambda pair: label_ordering.get(pair[0], f"9999_{pair[0]}"))
        storage = {key: value for key, value in storage}
    elif type(storage) == list or type(storage) == np.ndarray or type(storage) == set:
        storage = sorted(storage, key=lambda element: label_ordering.get(element, f"9999_{element}"))
    elif type(storage) == pd.DataFrame:
        sorter = lambda col: [label_ordering.get(v, f"9999_{v}") for v in col] if col.name == "label_name" else col
        if sorting_cols is None:
            sorting_cols = ["label_name"]
            if "image_name" in storage:
                sorting_cols.append("image_name")
            if "annotation_name" in storage:
                sorting_cols.append("annotation_name")
        storage = storage.sort_values(by=sorting_cols, key=sorter, ignore_index=True)
    else:
        settings.log.warning(f"Unsupported input type: {type(storage)}")

    return storage


@automatic_numpy_conversion
def sort_labels_cm(
    cm: Union[torch.Tensor, np.ndarray], cm_order: list[str], target_order: list[str]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Sorts the rows/columns in a cm to a target order.

    >>> cm = np.array([[0, 10, 3], [1, 2, 3], [8, 6, 4]])
    >>> cm_order = ['b', 'a', 'c']
    >>> target_order = ['a', 'b', 'c']
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
    assert len(cm_order) == len(target_order) and set(cm_order) == set(
        target_order
    ), "The same names must occur in the cm and the target order"
    assert sorted(set(cm_order)) == sorted(cm_order), "The names must be unique"

    # Swap rows
    switched_cm = torch.zeros_like(cm)
    ordering_indices = [cm_order.index(l) for l in target_order]
    for i, id in enumerate(ordering_indices):
        switched_cm[i, :] = cm[id, :]

    # Swap columns
    switched_cm_final = torch.zeros_like(cm)
    for j, id in enumerate(ordering_indices):
        switched_cm_final[:, j] = switched_cm[:, id]

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
    excluded_prefixes = ("running", "test", "special", "error")

    run_dirs = []
    if training_dir is None:
        training_dir = settings.training_dir
    for run_dir in sorted(training_dir.glob("*/*")):
        if settings.data_dirs.network in run_dir.parents:
            continue
        if not run_dir.is_dir():
            continue
        if run_dir.stem.startswith(excluded_prefixes):
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
        df = pd.read_pickle(fold_dir / "validation_results.pkl.xz")
        best_epoch_index = df["epoch_index"].max()
    else:
        # Find the best epoch id from the checkpoint name
        match = re.search(r"epoch=(\d+)", str(ckpt_file))
        assert match is not None, f"Could not extract the best epoch_index from the checkpoint name ({ckpt_file})"
        best_epoch_index = int(match.group(1))

    return ckpt_file, best_epoch_index


def get_nsd_thresholds(mapping: LabelMapping, aggregation_method: str = None) -> list[float]:
    df = pd.read_csv(settings_seg.nsd_tolerances_path)
    tolerance_column = settings_seg.nsd_aggregation.split("_")[-1] if aggregation_method is None else aggregation_method
    tolerance_column = f"tolerance_{tolerance_column}"

    tolerances = []
    for i in range(len(mapping)):
        name = mapping.index_to_name(i)
        tolerances.append(df.query("label_name == @name")[tolerance_column].item())

    return tolerances
