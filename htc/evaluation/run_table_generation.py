# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import copy
import filecmp
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.helper_functions import checkpoint_path, get_valid_run_dirs, run_experiment_notebook
from htc.utils.parallel import p_map


def merge_fold_tables(run_dir: Path, table_filename: str) -> pd.DataFrame:
    fold_dirs = sorted(run_dir.glob("fold*"))
    table_files = sorted(run_dir.rglob(table_filename))
    assert len(fold_dirs) == len(table_files), (
        f"Could not merge the {table_filename} tables for the run{run_dir} because there are {len(fold_dirs)} but only"
        f" {len(table_files)} table files"
    )

    all_dfs = []
    for f in table_files:
        df = pd.read_pickle(f)
        df["fold_name"] = f.parent.name

        if "validation" in table_filename:
            _, best_epoch_index = checkpoint_path(f.parent)
            df["best_epoch_index"] = best_epoch_index

            config = Config(f.parent / "config.json")
            expected_epoch_index = (
                config["trainer_kwargs/max_epochs"] if config["swa_kwargs"] else config["trainer_kwargs/max_epochs"] - 1
            )
            if df["epoch_index"].max() != expected_epoch_index:
                settings.log.warning(
                    f'The maximum epoch_index in the validation data ({df["epoch_index"].max()}) is different from the'
                    f" expected epoch_index based on the number of trained epochs ({expected_epoch_index}) for the run"
                    f" {run_dir}"
                )

        all_dfs.append(df)

    return pd.concat(all_dfs)


def generate_validation_table(run_dir: Path, table_stem: str = "validation_results") -> pd.DataFrame:
    def validation_table_npz(run_dir: Path) -> pd.DataFrame:
        # First collect all possible metric names
        metric_names = set()

        for fold_dir in sorted(run_dir.glob("fold*")):
            results_path = fold_dir / f"{table_stem}.npz"
            assert results_path.exists(), f"The run {fold_dir} does not contain any results"

            data = np.load(results_path, allow_pickle=True)["data"]
            for epoch_index, epoch_data in enumerate(data):
                for dataset_index, dataset in enumerate(epoch_data.values()):
                    for img_data in dataset.values():
                        metric_names.update(img_data.keys())

        # Then collect the actual result values
        rows = []
        metric_names = sorted(metric_names)

        for fold_dir in sorted(run_dir.glob("fold*")):
            results_path = fold_dir / f"{table_stem}.npz"
            config = Config(fold_dir / "config.json")
            _, best_epoch_index = checkpoint_path(fold_dir)

            # Generate a nice table based on the validation data structure
            data = np.load(results_path, allow_pickle=True)["data"]
            if len(data) < config["trainer_kwargs/max_epochs"]:
                settings.log.warning(
                    f"The number of epochs in the validation data ({len(data)}) is smaller than the epoch length in the"
                    f' config file {config.path_config} ({config["trainer_kwargs/max_epochs"]}) for the run {run_dir}'
                )

            for epoch_index, epoch_data in enumerate(data):
                for dataset_index, dataset in enumerate(epoch_data.values()):
                    for image_name, img_data in dataset.items():
                        current_row = [epoch_index, best_epoch_index, dataset_index, fold_dir.name, image_name]
                        for metric_name in metric_names:
                            if metric_name in img_data:
                                current_row.append(img_data[metric_name])
                            else:
                                current_row.append(None)

                        rows.append(current_row)

        return pd.DataFrame(
            rows, columns=["epoch_index", "best_epoch_index", "dataset_index", "fold_name", "image_name"] + metric_names
        )

    if len(sorted(run_dir.rglob(f"{table_stem}.npz"))) > 0:
        # E.g. old segmentation tasks
        return validation_table_npz(run_dir)
    else:
        # E.g. camera problem
        return merge_fold_tables(run_dir, f"{table_stem}.pkl.xz")


def save_validation_table(run_dir: Path) -> None:
    """
    Saves a generated table containing the validation results from all folds.

    Args:
        run_dir: Path to the run folder with subfolders for each fold.
    """
    table_path = run_dir / "validation_table.pkl.xz"
    if table_path.exists():
        # Skip run if results are already aggregated
        return None

    df = generate_validation_table(run_dir)
    df.to_pickle(table_path)

    # Also merge additional validation tables in case they are available
    additional_results = sorted(run_dir.rglob("*validation_results_*"))
    if len(additional_results) > 0:
        for stem in {f.name.split(".")[0] for f in additional_results}:
            print(stem)
            df = generate_validation_table(run_dir, table_stem=stem)
            df.to_pickle(run_dir / f"{stem.replace('results', 'table')}.pkl.xz")


def save_test_table(run_dir: Path) -> None:
    """
    Saves a generated table containing the test results from all folds.

    Args:
        run_dir: Path to the run folder with subfolders for each fold.
    """
    table_path = run_dir / "test_table.pkl.xz"
    if table_path.exists():
        # Skip run if results are already aggregated
        return None

    if len(sorted(run_dir.rglob("test_results.pkl.xz"))) > 0:
        df = merge_fold_tables(run_dir, "test_results.pkl.xz")
        df.to_pickle(table_path)


def check_run(run_dir: Path) -> None:
    # Check whether necessary files are available
    necessary_files = [
        "config.json",
        "log.txt",
        "*ckpt",
        "events.out.tfevents*",
        "system_log*.json",
        "validation_results.*",
    ]
    error_occurred = False

    fold_dirs = []
    for d in sorted(run_dir.iterdir()):
        if d.is_file():
            continue

        if d.name.startswith("running"):
            settings.log.error(
                f"The fold {d} is still prefixed with [var]running[/] meaning that the training for this fold did not"
                " complete successfully"
            )
            error_occurred = True
        elif d.name.startswith("fold"):
            fold_dirs.append(d)

    assert len(fold_dirs) > 0, f"At least one fold required for the run {run_dir}"

    for wildcard in necessary_files:
        for fold_dir in fold_dirs:
            if len(list(fold_dir.glob(f"{wildcard}"))) != 1:
                settings.log.warning(f"Not exactly one {wildcard} file found in the run directory {fold_dir}")
                error_occurred = True

    # Check log files
    for fold_dir in fold_dirs:
        with (fold_dir / "log.txt").open() as f:
            log_text = f.read()

        if "WARNING" in log_text:
            settings.log.warning(f"The log of the fold {fold_dir} contains warnings")
        if "ERROR" in log_text or "CRITICAL" in log_text:
            settings.log.error(f"The log of the fold {fold_dir} contains errors")
            error_occurred = True

    # Check config files
    try:
        config = Config.load_config_fold(
            run_dir
        )  # This also checks that the config files inside the folds are identical
        config.save_config(run_dir / "config.json")
    except AssertionError:
        settings.log.exception("Error in config loading")
        error_occurred = True

    # Check data files
    for fold in fold_dirs:
        if not filecmp.cmp(fold_dirs[0] / "data.json", fold / "data.json", shallow=False):
            settings.log.warning(
                f"The data specification for every fold must be identical but the fold {fold} does not match with the"
                f" first fold {fold_dirs[0]}"
            )
            error_occurred = True

    shutil.copy2(fold_dirs[0] / "data.json", run_dir / "data.json")

    return error_occurred


def check_tables(run_dir: Path) -> None:
    specs = DataSpecification(run_dir / "data.json")
    specs.activate_test_set()
    assert len(specs) > 0
    config = Config(run_dir / "config.json")
    df_val = pd.read_pickle(run_dir / "validation_table.pkl.xz")
    assert len(df_val) > 0

    if (run_dir / "test_table.pkl.xz").exists():
        df_test = pd.read_pickle(run_dir / "test_table.pkl.xz")
        assert len(df_test) > 0
    else:
        df_test = None

    error = False

    # Is the validation table consistent across epochs?
    epoch_counts = df_val.groupby("epoch_index").count().values
    if not (epoch_counts == epoch_counts[0]).all():
        settings.log.error(
            f"The validation table for the run {run_dir} is not consistent across epochs (the number of rows per epoch"
            " is different)"
        )
        error = True

    # Find columns where we can compute unique
    hashable_columns = []
    for c in df_val.columns:
        try:
            df_val.iloc[:2][c].unique()
            hashable_columns.append(c)
        except TypeError:
            pass

    # The ID columns should be identical across epochs
    unique_counts = df_val[hashable_columns].groupby("epoch_index").nunique()
    unique_counts = unique_counts.drop(columns=[c for c in unique_counts.columns if not c.endswith("_id")])
    unique_counts = unique_counts.values
    if not (unique_counts == unique_counts[0]).all():
        settings.log.error(
            f"The validation table for the run {run_dir} is not consistent across epochs (not the same unique ids are"
            " used for all epochs)"
        )
        error = True

    # Check that the best epoch always exists
    if "dataset_index" not in df_val:
        df_val["dataset_index"] = 0
    image_names_best = set(
        df_val.query("dataset_index == 0 and epoch_index == best_epoch_index")["image_name"].unique()
    )
    image_names_all = set(df_val.query("dataset_index == 0")["image_name"].unique())
    if image_names_best != image_names_all:
        settings.log.error(
            "The rows for the best_epoch_index cannot be found for every image (considering only dataset_index == 0)."
            " The following images have a best_epoch_index which does not occur in the validation:"
            f" {image_names_all - image_names_best}"
        )
        error = True

    # No intersection between validation and test?
    if df_test is not None:
        for fold_name in specs.fold_names():
            val_image_names = set(df_val.query("fold_name == @fold_name")["image_name"])
            if "fold_name" in df_test:
                test_image_names = set(df_test.query("fold_name == @fold_name")["image_name"])
            else:
                test_image_names = set(df_test["image_name"])

            common_image_names = val_image_names.intersection(test_image_names)
            if len(common_image_names) > 0:
                settings.log.error(
                    f"{len(common_image_names)} images are defined in the test and validation table (fold_name ="
                    f" [var]{fold_name}[/]) for the run {run_dir}"
                )
                error = True

    # Is every image in the specs defined in the tables?
    def check_table(df: pd.DataFrame, set_name: str):
        global error

        for fold_name, dataset in specs:
            fold_paths = set()
            for name, paths in dataset.items():
                if name.startswith(set_name):
                    fold_paths.update(paths)

            assert len(fold_paths) > 0

            if "fold_name" in df:
                table_image_names = set(df.query("fold_name == @fold_name")["image_name"].unique())
            else:
                table_image_names = set(df["image_name"].unique())

            fold_image_names = {p.image_name() for p in fold_paths}
            if fold_image_names != table_image_names:
                settings.log.error(
                    f"The [var]{set_name}[/] table for the run {run_dir} misses"
                    f" [var]{len(fold_image_names - table_image_names)}[/] paths which are defined in the data"
                    f" specification file (fold_name = [var]{fold_name}[/])"
                )
                error = True

            # Check for some images whether the labels of the image are also in the table
            if "used_labels" in df or "label_index" in df:
                config_new = copy.copy(config)
                config_new["input/no_features"] = True
                config_new["input/preprocessing"] = None
                dataset = DatasetImage(sorted(fold_paths), train=False, config=config_new)

                for i, sample in enumerate(dataset):
                    image_name = sample["image_name"]
                    if config["input/annotation_name"] and not config["input/merge_annotations"]:
                        img_labels = set()
                        for name in config["input/annotation_name"]:
                            img_labels.update(sample[f"labels_{name}"][sample[f"valid_pixels_{name}"]].unique().numpy())
                    else:
                        img_labels = set(sample["labels"][sample["valid_pixels"]].unique().numpy())

                    df_img = df.query("image_name == @image_name")
                    if "used_labels" in df:
                        table_labels = df_img.iloc[0]["used_labels"]
                    elif "label_index" in df:
                        table_labels = df_img["label_index"].unique()
                    else:
                        raise ValueError(f"Cannot find the labels in the image {image_name}")

                    if img_labels != set(table_labels):
                        settings.log.error(
                            f"The labels for the [var]{set_name}[/] table of the run {run_dir} do no match with the"
                            f" labels of the image {image_name}"
                        )
                        settings.log.error(f"Labels images: {img_labels}")
                        settings.log.error(f"Labels table: {table_labels}")
                        error = True

                    # We can't check every image because this would take too much work, so we only check the first few samples
                    if i >= 10:
                        break
            else:
                settings.log.info(
                    "Labels are not checked against example images because no labels could be found in the dataframe"
                    f" (columns={df.columns})"
                )

    check_table(df_val.query("epoch_index == best_epoch_index"), "val")
    if df_test is not None:
        check_table(df_test, "test")

    return error


def create_experiment_notebooks(run_dir: Path, base_notebook: str) -> None:
    possible_paths = [
        Path(base_notebook),
        Path.cwd() / base_notebook,
        Path(__file__).parent / base_notebook,
        settings.htc_package_dir / base_notebook,
    ]
    input_path = None
    for path in possible_paths:
        if path.exists():
            input_path = path
            break

    assert input_path is not None, (
        f"Cannot find the base notebook {base_notebook} which is necessary for the experiment visualizations. Tried at"
        f" the following locations: {possible_paths}"
    )

    output_path = run_dir / input_path.name
    if output_path.exists() and output_path.with_suffix(".html").exists():
        return None

    settings.log.info(f"Using the notebook {input_path}")
    run_experiment_notebook(
        notebook_path=input_path, output_path=output_path, parameters={"run_dir": str(run_dir)}, html_only=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the validation on the trained models. This includes aggregation of the fold information and a"
            " corresponding notebook generation in the results folder"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--notebook",
        default="ExperimentAnalysis.ipynb",
        help=(
            "Relative or absolute path to the notebook for the experiment visualizations (if set to an empty string, no"
            " notebook will be created)."
        ),
    )
    args = parser.parse_args()

    # Find runs which do not have any of the following files
    required_files = ["validation_table.pkl.xz", "ExperimentAnalysis.html"]
    run_dirs = []
    for r in get_valid_run_dirs():
        # If a run should be computed again, delete all of the required files
        if not any([(r / f).exists() for f in required_files]):
            run_dirs.append(r)

    if len(run_dirs) > 0:
        settings.log.info("Will generate results for the following runs:")
        for run_dir in run_dirs:
            settings.log.info(f"{run_dir.parent.name}/{run_dir.name}")

        errors = p_map(check_run, run_dirs)
        assert not any(errors), "At least one run folder misses some files. Aborting..."

        p_map(save_validation_table, run_dirs)
        p_map(save_test_table, run_dirs)

        errors = p_map(check_tables, run_dirs)
        assert not any(errors), "Something wrong with the validation and/or test tables"

        if args.notebook != "":
            settings.log.info("Creating notebooks...")
            p_map(
                partial(create_experiment_notebooks, base_notebook=args.notebook), run_dirs, num_cpus=12
            )  # If you encounter errors, please run the execution sequentially (unfortunately, errors are not shown when run in parallel)
            # for run in run_dirs:
            #     create_experiment_notebooks(run, base_notebook=args.notebook)
    else:
        settings.log.info("All runs complete. Nothing to do")
