# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path

from htc.models.run_pretrained_semantic_models import compress_run
from htc.settings import settings
from htc.utils.file_transfer import list_files_s3, upload_file_s3
from htc.utils.general import sha256_file
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


def upload_projection_matrices() -> None:
    files_local = sorted((settings.results_dir / "projection_matrices").glob("*.blosc"))
    for file in files_local:
        upload_file_s3(local_path=file, remote_path=f"projection_matrices/{file.name}")

    files_remote = list_files_s3("projection_matrices")
    assert {f.name for f in files_local} == {Path(f).name for f in files_remote}, (
        f"The local and remote files of the projection matrices do not match:\n{files_local}\n{files_remote}"
    )

    known_projection_matrices = {}
    for f_local, f_remote in zip(files_local, files_remote, strict=True):
        known_projection_matrices[f_local.stem] = {
            "sha256": sha256_file(f_local),
            "url": f_remote,
        }

    print("Please update the known_projection_matrices of ProjectionTransform with the output of this script:")
    print(f"{known_projection_matrices = }\n")

    table_lines = [
        "| base name | projection mode | experiment type | source species |",
        "| ----------- | ----------- | ----------- | ----------- |",
    ]
    for f in files_local:
        projection_mode, experiments_type, source_species, _ = f.stem.split("_")
        table_lines.append(f"| {f.stem} | {projection_mode} | {experiments_type} | {source_species} |")

    print("Please update the documentation of ProjectionTransform with the following table:")
    print("\n".join(table_lines))


if __name__ == "__main__":
    compress_species_runs()
    upload_projection_matrices()
