# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from htc.tivita.DataPath import DataPath
from htc.utils.helper_functions import sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.mitk.mitk_masks import segmentation_to_nrrd
from htc.utils.parallel import p_map


class PathAnnotations:
    def __init__(self, target_dir: Path, name_to_path: dict[str, DataPath]):
        """
        This class can be used to create an folder for MITK with images which should be annotated.

        Args:
            target_dir: The directory where the files for MITK should be stored.
            name_to_path: Dictionary which lists all paths (as values) which should be annotated and corresponding names (as keys) which should be used (e.g. for the file names).
        """
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.target_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.seg_dir = self.target_dir / "results"
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.name_to_path = name_to_path

    def create_dataset_labels(self, mapping: LabelMapping) -> None:
        """
        Create the dataset_labels.json file which lists all labels of the mapping.

        Args:
            mapping: The mapping which will be used to extract the label names from.
        """
        labels = sort_labels(mapping.label_names())
        dataset_labels = []

        for i, label in enumerate(labels):
            dataset_labels.append({
                "name": f"{i + 1:02d}_{label}",
                "color": mapping.name_to_color(label),
            })

        with (self.target_dir / "dataset_labels.json").open("w") as f:
            json.dump(dataset_labels, f)

    def create_task_list(self, default_task_name_prefix: bool = True) -> None:
        """
        Creates the task_list.json file which lists all images for annotation.

        Args:
            default_task_name_prefix: If True, the task names will be prefixed with a global image number. If False, the name of the name2path mapping will be used.
        """
        tasks = {
            "FileFormat": "MITK Segmentation Task List",
            "Version": 1,
            "Name": "Nature Paper Classification",
            "Defaults": {"LabelNameSuggestions": "dataset_labels.json"},
            "Tasks": [],
        }

        n_digits = int(np.ceil(np.log10(len(self.name_to_path))))
        for i, name in enumerate(self.name_to_path.keys()):
            if default_task_name_prefix:
                task_name = str(i + 1).zfill(n_digits) + "_" + name
            else:
                task_name = name

            tasks["Tasks"].append({
                "Name": task_name,
                "Image": f"images/{name}.png",
                "Result": f"results/{name}.nrrd",
            })

        with (self.target_dir / "task_list.json").open("w") as f:
            json.dump(tasks, f, indent=4)

    def copy_images(self) -> None:
        """
        Copy the RGB images for all paths to the images directory.
        """
        # For efficient parallel processing, we need to call an extra function (and no class method)
        p_map(
            partial(_copy_image_path, images_dir=self.images_dir),
            self.name_to_path.keys(),
            self.name_to_path.values(),
            task_name="Copy images",
        )

    def create_initial_annotations(self, name_to_mask: dict[str, np.ndarray]) -> None:
        """
        Create initial annotations for the given images in the results directory. This is useful if you want to provide basic annotations (e.g. polygon or weak labels) for the annotators.

        Args:
            name_to_mask: Dictionary which lists the masks for the images which should be annotated. The same names as in the name_to_path dictionary must be used.
        """
        for name, mask in name_to_mask.items():
            mapping = LabelMapping.from_path(self.name_to_path[name])
            segmentation_to_nrrd(nrrd_file=self.seg_dir / f"{name}.interim.nrrd", mask=mask, mapping_mask=mapping)

    def semantic_info_table(self) -> None:
        """
        Prepares a CSV file which lists all images for annotation. This can copied to the online sheet for the annotators to enter their names.
        """
        rows = []
        for i, name in enumerate(self.name_to_path.keys()):
            rows.append({
                "index": i + 1,
                "file_name": name,
                "image_name": self.name_to_path[name].image_name(),
                "annotator": "",
                "correction_annotator": "",
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.target_dir / "semantic_annotation_info.csv", index=False)


def _copy_image_path(name: str, path: DataPath, images_dir: Path) -> None:
    rgb = path.read_rgb_reconstructed()
    rgb = Image.fromarray(rgb)
    rgb.save(images_dir / f"{name}.png", optimize=True)
