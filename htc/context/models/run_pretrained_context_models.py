# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.context.settings_context import settings_context
from htc.models.run_pretrained_semantic_models import compress_run
from htc.utils.file_transfer import upload_file_s3


def compress_model_comparison_runs() -> None:
    target_dir = settings_context.results_dir / "pretrained_models"
    target_dir.mkdir(exist_ok=True, parents=True)

    run_dirs = [
        settings_context.best_transform_runs["organ_transplantation"],
        settings_context.best_transform_runs_rgb["organ_transplantation"],
    ]

    known_models = {}
    for run_dir in run_dirs:
        print(run_dir)
        name = f"{run_dir.parent.name}@{run_dir.name}"
        known_models[name] = {
            "sha256": compress_run(run_dir, output_path=target_dir / f"{name}.zip"),
            "url": upload_file_s3(local_path=target_dir / f"{name}.zip", remote_path=f"models/{name}.zip"),
        }

    # Update known_models of HTCModel with the output of this script
    print(known_models)


if __name__ == "__main__":
    compress_model_comparison_runs()
