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


class AnnotationPath:
    def __init__(
        self,
        path: DataPath,
        filename: str = None,
        mask: np.ndarray = None,
        layer_names: list[str] = None,
        label_mapping: LabelMapping = None,
    ):
        """
        Small helper data object to store additional information for every path which should be annotated.

        Args:
            path: The data path to the image.
            filename: The filename without suffix which should be used for the images and results. Defaults to using the image name.
            mask: Existing annotations for the image which should be visible in MITK.
            layer_names: Names for the layers as shown in MITK.
            label_mapping: The label mapping which should be used to interpret the mask. If None, the default mapping of the data path will be used.
        """
        self.path = path
        self.filename = filename if filename is not None else path.image_name()
        self.mask = mask
        self.layer_names = layer_names
        self.label_mapping = label_mapping


class PathAnnotations:
    def __init__(self, target_dir: Path, paths: list[DataPath] | list[AnnotationPath]):
        """
        This class can be used to create an folder for MITK with images which should be annotated.

        The following example will guide you through the basic steps of how MITK can be used to annotate our images.
        >>> import tempfile
        >>> tmp_dir_handle = tempfile.TemporaryDirectory()
        >>> tmp_dir = Path(tmp_dir_handle.name)

        Let's prepare two pig images for annotation:
        >>> from htc_projects.seg_open.settings_seg_open import settings_seg_open
        >>> paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35"), DataPath.from_image_name("P044#2020_02_01_09_51_15")]
        >>> annotations = PathAnnotations(tmp_dir / "project_pig_example", paths)

        The task list is required by MITK and lists all images which should be annotated. `Name` specifies how the image will show up in MITK and is per default numbered from 1 to n. `Image` and `Result` specify where MITK can find the RGB image and where the segmentation mask should be stored.
        >>> annotations.create_task_list()
        >>> print((tmp_dir / "project_pig_example" / "task_list.json").read_text())
        {
            "FileFormat": "MITK Segmentation Task List",
            "Version": 1,
            "Name": "Nature Paper Classification",
            "Defaults": {
                "LabelNameSuggestions": "dataset_labels.json"
            },
            "Tasks": [
                {
                    "Name": "1_P043#2019_12_20_12_38_35",
                    "Image": "images/P043#2019_12_20_12_38_35.png",
                    "Result": "results/P043#2019_12_20_12_38_35.nrrd"
                },
                {
                    "Name": "2_P044#2020_02_01_09_51_15",
                    "Image": "images/P044#2020_02_01_09_51_15.png",
                    "Result": "results/P044#2020_02_01_09_51_15.nrrd"
                }
            ]
        }

        Copy the RGB images:
        >>> annotations.copy_images()
        [...]

        Create a table with all images for annotation. Please upload this table to the shared table sheet so that the annotators and correctors can put their name to the image on which they worked on.
        >>> annotations.semantic_info_table()

        Optionally, you can explicitly specify which labels should appear in MITK. You can also leave this step out and copy the default dataset_labels.json file from another project.
        >>> annotations.create_dataset_labels(settings_seg_open.label_mapping_pig)

        The generated folder can be copied to the Nextcloud folder (except for the `semantic_annotation_info.csv` file). Additionally, copy the `default_empty_labels.lsetp`, `launch_mitk.scpt` and `MitkWorkbench.bat` files from another project folder.
        >>> for f in sorted(tmp_dir.rglob("*")):
        ...     print(f.relative_to(tmp_dir))
        project_pig_example
        project_pig_example/dataset_labels.json
        project_pig_example/images
        project_pig_example/images/P043#2019_12_20_12_38_35.png
        project_pig_example/images/P044#2020_02_01_09_51_15.png
        project_pig_example/results
        project_pig_example/semantic_annotation_info.csv
        project_pig_example/task_list.json

        >>> tmp_dir_handle.cleanup()

        Args:
            target_dir: The directory where the files for MITK should be stored.
            paths: List of images which should be annotated. The entries in the list can either be DataPath objects or AnnotationPath objects. The latter allows more fine-grained control like changing the image filenames or providing default annotations.
        """
        self.target_dir = target_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.target_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.seg_dir = self.target_dir / "results"
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.images = paths
        for i in range(len(self.images)):
            if isinstance(self.images[i], DataPath):
                self.images[i] = AnnotationPath(self.images[i])

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

        n_digits = int(np.ceil(np.log10(len(self.images))))
        for i, image in enumerate(self.images):
            if default_task_name_prefix:
                task_name = str(i + 1).zfill(n_digits) + "_" + image.filename
            else:
                task_name = image.filename

            info = {
                "Name": task_name,
                "Image": f"images/{image.filename}.png",
                "Result": f"results/{image.filename}.nrrd",
            }
            if image.path.meta("description") is not None:
                info["Description"] = image.path.meta("description")

            tasks["Tasks"].append(info)

        with (self.target_dir / "task_list.json").open("w") as f:
            json.dump(tasks, f, indent=4)

    def copy_images(self) -> None:
        """
        Copy the RGB images for all paths to the images directory.
        """
        # For efficient parallel processing, we need to call an extra function (and no class method)
        p_map(partial(_copy_image_path, images_dir=self.images_dir), self.images, task_name="Copy images")

    def create_initial_annotations(self) -> None:
        """
        Create initial annotations for the given images in the results directory. This is useful if you want to provide basic annotations (e.g. polygon or weak labels) for the annotators.
        """
        mapping_file = None
        for image in self.images:
            if image.mask is None:
                continue

            if image.label_mapping is None:
                mapping_file = LabelMapping.from_path(image.path)
            else:
                mapping_file = image.label_mapping

            segmentation_to_nrrd(
                nrrd_file=self.seg_dir / f"{image.filename}.interim.nrrd",
                mask=image.mask,
                mapping_mask=mapping_file,
                layer_names=image.layer_names,
            )

        assert mapping_file is not None, (
            "At least one image must have a mask when calling the create_initial_annotations() function."
        )

    def semantic_info_table(self) -> None:
        """
        Prepares a CSV file which lists all images for annotation. This can copied to the online sheet for the annotators to enter their names.
        """
        rows = []
        for i, image in enumerate(self.images):
            rows.append({
                "index": i + 1,
                "file_name": image.filename,
                "image_name": image.path.image_name(),
                "annotator": "",
                "correction_annotator": "",
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.target_dir / "semantic_annotation_info.csv", index=False)


def _copy_image_path(image: AnnotationPath, images_dir: Path) -> None:
    rgb = image.path.read_rgb_reconstructed()
    rgb = Image.fromarray(rgb)
    rgb.save(images_dir / f"{image.filename}.png", optimize=True)
