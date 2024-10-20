# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from htc.models.data.SpecsGeneration import SpecsGeneration
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.paths import filter_semantic_labels_only


class SpecsGenerationBench(SpecsGeneration):
    def __init__(self):
        super().__init__(name="pigs_semantic-all_train-only")

    def generate_folds(self) -> list[dict]:
        data_specs = []

        # Every image with semantic annotations
        paths = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_semantic_labels_only]))
        paths += list(
            DataPath.iterate(settings.data_dirs.semantic / "context_experiments", filters=[filter_semantic_labels_only])
        )
        assert len(paths) > 0

        imgs = [p.image_name() for p in paths]
        assert len(set(imgs)) == len(imgs), "Duplicate paths"

        settings.log.info(f"Found {len(imgs)} images with semantic annotations")

        data_specs.append({
            "fold_name": "fold_all",
            "train_semantic": {
                "image_names": [p.image_name() for p in paths],
            },
        })

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Creates the data specification file for the benchmarking dataset (all paths in a single training fold)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=settings.htc_package_dir / "benchmarking/data",
        help="Directory where the resulting data specification file should be stored.",
    )
    args = parser.parse_args()

    SpecsGenerationBench().generate_dataset(args.output_dir)
