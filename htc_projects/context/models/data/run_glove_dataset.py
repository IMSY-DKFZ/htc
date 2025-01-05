# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

from htc.models.data.DataSpecification import DataSpecification
from htc.models.data.SpecsGeneration import SpecsGeneration


class SpecsGenerationGlove(SpecsGeneration):
    def __init__(self):
        super().__init__(name="pigs_semantic-only_5foldsV2_glove")

    def generate_folds(self) -> list[dict]:
        data_specs = []

        specs = DataSpecification("pigs_semantic-only_5foldsV2.json")
        specs.activate_test_set()

        for fold_name, splits in specs:
            fold_specs = {"fold_name": fold_name}

            for name, split_paths in splits.items():
                names = [p.image_name() for p in split_paths if "glove" not in p.annotated_labels()]
                fold_specs[name] = {"image_names": names}

            names = [p.image_name() for p in splits["test"] if "glove" in p.annotated_labels()]
            assert not any(n in fold_specs["test"]["image_names"] for n in names)
            fold_specs["test_ood"] = {"image_names": names}

            data_specs.append(fold_specs)

        return data_specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Creates the data specification file for the glove OOD experiment (no glove images in training but separate"
            " test dataset with glove images)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory where the resulting data specification file should be stored.",
    )
    args = parser.parse_args()

    SpecsGenerationGlove().generate_dataset(args.output_dir)
