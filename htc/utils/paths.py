# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
from typing import Callable, Union

from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath


def filter_semantic_labels_only(path: "DataPath") -> bool:
    labels = path.annotated_labels()
    if any([l in settings_seg.labels[1:] for l in labels]):
        # We are only interested in images with one of the organs which we use for training (without background)
        return True
    else:
        return False


def filter_min_labels(path: "DataPath", min_labels: int = 1) -> bool:
    labels = path.annotated_labels(annotation_name="all")
    if len(labels) >= min_labels:
        return True
    else:
        return False


def all_masks_paths() -> list[DataPath]:
    paths = list(DataPath.iterate(settings.data_dirs.masks))
    paths += list(DataPath.iterate(settings.data_dirs.masks / "overlap"))

    return paths


class ParserPreprocessing:
    def __init__(self, description: str):
        self.parser = argparse.ArgumentParser(
            description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.parser.add_argument(
            "--specs",
            required=False,
            type=Path,
            default=None,
            help=(
                "Path or name to the data specification file in which case all paths from this file will be used"
                " (including the test set). Please note that you should still either provide the --dataset-name"
                " argument for a default intermediates folder or the --output-path argument."
            ),
        )
        self.parser.add_argument(
            "--dataset-path",
            required=False,
            type=Path,
            default=None,
            help=(
                "Path to the directory from which DataPaths should be collected (e.g. an existing data directory)."
                " Please note that you should still either provide the --dataset-name argument for a default"
                " intermediates folder or the --output-path argument."
            ),
        )
        self.parser.add_argument(
            "--dataset-name",
            required=False,
            default=None,
            help=(
                "Name of the dataset (e.g. name of the corresponding folder on the network drive). This will also be"
                " used to set the default intermediates directory, i.e. the generated files will be stored in the"
                " intermediates directory corresponding to the given dataset name. If both --specs and --dataset-path"
                " are None, then the paths will be collected from the dataset."
            ),
        )
        self.parser.add_argument(
            "--output-path",
            required=False,
            type=Path,
            default=None,
            help=(
                "Path to the directory where the generated files should be stored (e.g."
                " '...intermediates/preprocessing' to store parameter images in"
                " '...intermediates/preprocessing/parameter_images')."
            ),
        )
        self.parser.add_argument(
            "--file-type",
            default="blosc",
            choices=["npy", "blosc"],
            help="Output file type for scripts which produce one file per image (e.g. L1 preprocessing).",
        )
        self.parser.add_argument(
            "--regenerate",
            action="store_true",
            default=False,
            help=(
                "To regenerate the files, even if the files are already stored in the output location (for scripts with"
                " output files e.g. L1 processing)."
            ),
        )

    def get_paths(self, filters: Union[list[Callable[["DataPath"], bool]], None] = None) -> list[DataPath]:
        self.args = self.parser.parse_args()
        if self.args.specs is not None:
            assert self.args.dataset_path is None, "--dataset-path is not used if --specs is given"
            specs = DataSpecification(self.args.specs)
            specs.activate_test_set()
            paths = specs.paths()
        elif self.args.dataset_path is not None:
            assert self.args.specs is None, "--specs is not used if --dataset-path is given"
            paths = list(DataPath.iterate(self.args.dataset_path))
        else:
            if self.args.dataset_name == "2021_02_05_Tivita_multiorgan_masks":
                paths = list(DataPath.iterate(settings.data_dirs[self.args.dataset_name], filters))
                paths += list(DataPath.iterate(settings.data_dirs[self.args.dataset_name] / "overlap", filters))
            else:
                paths = list(DataPath.iterate(settings.data_dirs[self.args.dataset_name], filters))

        if self.args.dataset_name is not None:
            # From now on, we write to the intermediates directory of the selected dataset
            settings.intermediates_dir.set_default_location(self.args.dataset_name)

        return paths
