# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import hashlib
from pathlib import Path
from zipfile import ZipFile

from rich.progress import track

from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.file_transfer import upload_file_s3
from htc.utils.general import sha256_file


def compress_run(run_dir: Path, output_path: Path) -> str:
    """
    Compresses a trained model (run directory) into a zip archive. All folds of the run are included.

    Args:
        run_dir: Path to the run directory
        output_path: Path where the zip file should be stored.

    Returns: SHA256 hash value of the folder (=hash of the file hashes).
    """
    model = run_dir.parent.name
    files_to_copy = {}
    hash_cat = ""

    # Collect all files we want to put into the archive
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            files_to_copy[f] = f.name

    for f in sorted(run_dir.rglob("fold_*/*")):
        if f.suffix == ".ckpt" or f.name.startswith(("events", "trainings_stats")):
            files_to_copy[f] = f.relative_to(str(run_dir))

    # Correct sorting for the folder hash (relies on the same sorting)
    files_to_copy = dict(sorted(files_to_copy.items()))

    # Put everything into the archive
    with ZipFile(output_path, mode="w") as archive:
        for f, path_archive in files_to_copy.items():
            # Also create intermediate folders for model and run_folder to keep the same folder structure also in the cache dir
            archive.write(f, Path(model) / run_dir.name / path_archive)
            hash_cat += sha256_file(f)

    hash_folder = hashlib.sha256(hash_cat.encode()).hexdigest()
    return hash_folder


def compress_model_comparison_runs() -> None:
    target_dir = settings_seg.results_dir / "pretrained_models"
    target_dir.mkdir(exist_ok=True, parents=True)

    # Create zip archives for the pretrained models of our semantic organ segmentation paper
    run_dirs = []
    for i, row in collect_comparison_runs(settings_seg.model_comparison_timestamp).iterrows():
        for model_type in ["rgb", "param", "hsi"]:
            run_dir = settings.training_dir / row["model"] / row[f"run_{model_type}"]
            run_dirs.append(run_dir)

    known_models = {}
    for run_dir in track(run_dirs):
        name = f"{run_dir.parent.name}@{run_dir.name}"
        known_models[name] = {
            "sha256": compress_run(run_dir, output_path=target_dir / f"{name}.zip"),
            "url": upload_file_s3(local_path=target_dir / f"{name}.zip", remote_path=f"models/{name}.zip"),
        }

    print(known_models)


if __name__ == "__main__":
    compress_model_comparison_runs()
