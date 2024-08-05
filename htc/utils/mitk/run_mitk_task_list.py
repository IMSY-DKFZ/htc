# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from rich.progress import track

from htc import read_tivita_rgb, safe_copy, settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Searches for all Tivita images in a folder and creates a task list for MITK to annotated those images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the folder with the images which should be annotated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Path to the output directory where the MITK files (images directory and task_list.json) should be stored."
        ),
    )
    parser.add_argument(
        "--wildcard",
        type=str,
        default="*_RGB-Image.png",
        required=False,
        help="Wildcard file pattern which should be used to select RGB files.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    assert input_dir.exists(), f"The input directory {input_dir} does not exist"

    images_dir = output_dir / "images"
    task_list_file = output_dir / "task_list.json"
    zip_file = output_dir / "mitk.zip"

    assert not images_dir.exists(), (
        f"The output directory {output_dir} already contains an images folder. Please delete it or select a different"
        " output directory"
    )
    assert not task_list_file.exists(), (
        f"The output directory {output_dir} already contains a task_list.json. Please delete it or select a different"
        " output directory"
    )
    assert not zip_file.exists(), (
        f"The zip file {zip_file} already exists in the output directory. Please delete it or select a different output"
        " directory"
    )

    # Find all images in the input directory
    images_dir.mkdir(exist_ok=True, parents=True)
    paths = {}
    for p in sorted(input_dir.rglob(args.wildcard)):
        # We use a dict to get a sorted list of unique images
        paths[p] = True

    assert len(paths) > 0, f"Could not find any images in {input_dir}"
    print(f"Found {len(paths)} images in {input_dir}")

    # Create task list and copy RGB images
    tasks = {
        "FileFormat": "MITK Segmentation Task List",
        "Version": 1,
        "Name": "Segmentation",
        "Defaults": {"LabelNameSuggestions": "dataset_labels.json"},
        "Tasks": [],
    }

    n_digits = math.ceil(math.log10(len(paths)))
    for i, p in track(enumerate(paths.keys()), total=len(paths)):
        timestamp = p.stem.removesuffix(args.wildcard.removeprefix("*"))
        image_name = str(i + 1).rjust(n_digits, "0") + f"_{timestamp}"

        try:
            rgb = read_tivita_rgb(p)
            rgb = Image.fromarray(rgb)
            rgb.save(images_dir / f"{image_name}.png", optimize=True)
        except Exception:
            settings.log_once.info(
                "Could not read the Tivita RGB image. The RGB file will be copied instead. This is fine if the image"
                " does not contain black borders"
            )
            safe_copy(p, images_dir / f"{image_name}.png")

        tasks["Tasks"].append({
            "Name": f"{image_name}",
            "Image": f"images/{image_name}.png",
            "Result": f"results/{image_name}.nrrd",
        })

    with task_list_file.open("w") as f:
        json.dump(tasks, f, indent=4)

    # Create zip file of the task list and the images
    with ZipFile(zip_file, mode="w") as archive:
        archive.write(task_list_file, task_list_file.name)
        for p in sorted(images_dir.iterdir()):
            archive.write(p, f"images/{p.name}")

    print(f"Stored the images folder at {images_dir}")
    print(f"Stored the task_list.json at {task_list_file}")
    print(f"Stored zip file at {zip_file}")
