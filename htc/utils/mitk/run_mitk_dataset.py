# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from htc import LabelMapping
from htc.tivita.DataPath import DataPath
from htc.utils.mitk.mitk_masks import segmentation_to_nrrd
from htc.utils.parallel import p_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collects all images from a dataset and converts the existing annotations to MITK nrrd files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to a dataset where data paths should be collected.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Path to the output directory where the MITK files (images and results directory and task_list.json) should"
            " be stored."
        ),
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    assert input_dir.exists(), f"The input directory {input_dir} does not exist"

    images_dir = output_dir / "images"
    results_dir = output_dir / "results"
    task_list_file = output_dir / "task_list.json"
    for f in [images_dir, results_dir, task_list_file]:
        assert not f.exists(), (
            f"The output directory {output_dir} already contains {f}. Please select a different output directory or"
            " clear it first"
        )

    images_dir.mkdir(exist_ok=True, parents=True)
    results_dir.mkdir(exist_ok=True, parents=True)

    tasks = {
        "FileFormat": "MITK Segmentation Task List",
        "Version": 1,
        "Name": "Segmentation",
        "Defaults": {"LabelNameSuggestions": "dataset_labels.json"},
        "Tasks": [],
    }

    paths = list(DataPath.iterate(args.input_dir))
    assert len(paths) > 0, f"Could not find any images in {input_dir}"

    def handle_path(path: DataPath) -> dict[str, str]:
        rgb = path.read_rgb_reconstructed()
        rgb = Image.fromarray(rgb)
        rgb.save(images_dir / f"{path.image_name()}.png", optimize=True)

        mask = path.read_segmentation(annotation_name="all")
        if type(mask) == dict:
            mask = np.stack(list(mask.values()))

        segmentation_to_nrrd(
            nrrd_file=results_dir / f"{path.image_name()}.nrrd",
            mask=mask,
            mapping_mask=LabelMapping.from_path(path),
        )

        return {
            "Name": f"{path.image_name()}",
            "Image": f"images/{path.image_name()}.png",
            "Result": f"results/{path.image_name()}.nrrd",
        }

    tasks["Tasks"] = p_map(handle_path, paths)

    with task_list_file.open("w") as f:
        json.dump(tasks, f, indent=4)
