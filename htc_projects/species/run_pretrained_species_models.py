# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.run_pretrained_semantic_models import compress_run
from htc.settings import settings
from htc.utils.file_transfer import upload_file_s3
from htc_projects.species.settings_species import settings_species


def compress_species_runs() -> None:
    target_dir = settings_species.results_dir / "pretrained_models"
    target_dir.mkdir(exist_ok=True, parents=True)

    run_dirs = sorted((settings.training_dir / "image").glob(f"{settings_species.model_timestamp}*"))

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
    compress_species_runs()
