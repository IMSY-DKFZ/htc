# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re

from htc.cluster.ClusterConnection import ClusterConnection
from htc.cluster.RunGenerator import RunGenerator
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import subprocess_run
from htc_projects.species.settings_species import settings_species


def sync_projection_matrices():
    """
    Sync the projection matrices with the cluster.
    """
    local_dir = settings.results_dir / "projection_matrices"
    assert local_dir.is_dir(), (
        f"The local directory containing the projection matrices ({local_dir}) does not exist. Please compute the"
        " projection matrices first (run_create_projections.py)."
    )

    with ClusterConnection() as connection:
        cluster_path = f"{connection.checkpoints_dir}/projection_matrices"
        try:
            connection.ftp.stat(cluster_path)
            settings.log.info(
                "projection_matrices folder exists in the user directory on the cluster. The files will be updated"
            )
        except FileNotFoundError:
            connection.ftp.mkdir(cluster_path)
            settings.log.info(
                "projection_matrices folder does not exist in the user directory on the cluster. The files will be"
                " copied"
            )

        res = subprocess_run(
            f"rsync -a --delete --info=progress2 {local_dir}/ {settings.dkfz_userid}@{connection.host}:{cluster_path}/",
            shell=True,
        )
        assert res.returncode == 0, "Could not sync the projection matrices with the cluster."


def nested_cv_adjustment(base_name: str, config: Config, nested_fold_index: int, **kwargs) -> str:
    match = re.search(r"nested-\d+-(\d+)", config["input/data_spec"])
    assert match is not None, f"Could not find the nested fold index in the data spec: {config['input/data_spec']}"
    max_nested_index = int(match.group(1))

    assert (
        nested_fold_index <= max_nested_index
    ), f"The nested fold index {nested_fold_index} cannot be greater than the maximum nested index {max_nested_index}"
    config["input/data_spec"] = config["input/data_spec"].replace("nested-0", f"nested-{nested_fold_index}")
    return base_name + f"_nested-{nested_fold_index}-{max_nested_index}"


def projection_adjustment(base_name: str, config: Config, source_species: str, **kwargs) -> str:
    config["input/transforms_gpu"] = [
        {
            "class": "KorniaTransform",
            "transformation_name": "RandomAffine",
            "translate": [0.0625, 0.0625],
            "scale": [0.9, 1.1],
            "degrees": 45,
            "padding_mode": "reflection",
            "p": 0.5,
        },
        {
            "class": "KorniaTransform",
            "transformation_name": "RandomHorizontalFlip",
            "p": 0.25,
        },
        {
            "class": "KorniaTransform",
            "transformation_name": "RandomVerticalFlip",
            "p": 0.25,
        },
        {
            "class": "htc_projects.species.species_transforms>ProjectionTransform",
            "base_name": settings_species.species_projection[source_species],
            "interpolate": True,
            "target_labels": ["kidney"],
            "p": 0.8,
        },
        {
            "class": "htc_projects.context.context_transforms>OrganTransplantation",
            "p": 0.8,
        },
        {
            "class": "Normalization",
        },
    ]

    return base_name.replace("baseline_", f"projected_{source_species}2")


def joint_training_baseline_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/data_spec"] = config["input/data_spec"].replace("semantic-only", "semantic-only+pig-p+rat-p")
    config["input/target_domain"] = ["species_index"]

    return base_name.replace("baseline_", "joint_pig-p+rat-p2")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run all species training runs on the cluster (baseline and projected runs). Make sure that the"
            " projection_matrices folder (from your results directory) is available at the checkpoints directory on the"
            " cluster."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Explicit timestamp to use for the run names. If set, only missing runs will be computed.",
    )
    args = parser.parse_args()

    sync_projection_matrices()

    rg = RunGenerator(args.timestamp)
    for config_name in [
        "baseline_pig",
        "baseline_rat",
        "baseline_human",
    ]:
        for i in range(settings_species.n_nested_folds):
            config = Config(f"species/configs/{config_name}.json")
            rg.generate_run(
                config, config_adjustments=[nested_cv_adjustment], memory="16G", single_submit=True, nested_fold_index=i
            )

        target_species = config_name.split("_")[-1]
        for source_species in ["pig", "rat"]:
            if source_species == target_species:
                continue

            for i in range(settings_species.n_nested_folds):
                config = Config(f"species/configs/{config_name}.json")
                rg.generate_run(
                    config,
                    config_adjustments=[nested_cv_adjustment, projection_adjustment],
                    memory="16G",
                    single_submit=True,
                    nested_fold_index=i,
                    source_species=source_species,
                )

    # Joint training for the human baseline run
    for i in range(settings_species.n_nested_folds):
        config = Config("species/configs/baseline_human.json")
        rg.generate_run(
            config,
            config_adjustments=[nested_cv_adjustment, joint_training_baseline_adjustment],
            memory="16G",
            single_submit=True,
            nested_fold_index=i,
        )

    rg.submit_jobs()
