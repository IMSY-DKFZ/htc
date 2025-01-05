# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
from pathlib import Path

import torch

from htc.model_processing.run_tables import TableTestPredictor
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc_projects.species.tables import baseline_table, ischemic_table


def compute_perfusion_scores(run_dir: Path, recalculate: bool) -> None:
    match = re.search(r"(pig|rat|human)_nested-(\d+)-(\d+)$", run_dir.name)
    assert match is not None, f"Could not infer species from run folder name {run_dir.name}"
    species = match.group(1)
    nested_index = int(match.group(2))
    max_nested_index = int(match.group(3))

    if nested_index != 0:
        # We compute the perfusion scores only for the first nested fold
        return

    table_name = f"test_table_{species}_perfusion"
    if not recalculate and (run_dir / f"{table_name}.pkl.xz").exists():
        return

    config = Config(run_dir / "config.json")
    mapping = LabelMapping.from_config(config)

    # Compute scores for everything except the baseline tables (which are already part of the training data)
    df_baseline = baseline_table(label_mapping=mapping).query("species_name == @species")
    df_ischemic = ischemic_table(label_mapping=mapping).query("species_name == @species")

    names = sorted(set(df_ischemic["image_name"]) - set(df_baseline["image_name"]))

    df_selection = df_ischemic[df_ischemic["image_name"].isin(names)][["image_name", "annotation_name"]]
    df_selection = df_selection.drop_duplicates()
    assert sorted(df_selection["image_name"].tolist()) == sorted(names)

    paths = DataPath.from_table(df_selection)
    assert len(paths) > 0, f"No paths found for species {species}"

    settings.log.info(
        f"Computing {species} perfusion scores for the run {run_dir.name} ({len(paths)} perfusion images)"
    )

    run_dirs = []
    for i in range(max_nested_index + 1):
        run_dirs.append(
            run_dir.with_name(
                run_dir.name.replace(f"_nested-{nested_index}-{max_nested_index}", f"_nested-{i}-{max_nested_index}")
            )
        )

    predictor = TableTestPredictor(
        run_dirs,
        config=config,
        test_table_name=table_name,
        metrics=["DSC", "CM"],
        paths=paths,
    )

    with torch.autocast(device_type="cuda"):
        predictor.start(task_queue=None, hide_progressbar=False)
    predictor.save_table(run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute test scores for the perfusion datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True,
        help="The timestamp to select runs for which perfusion scores should be calculated.",
    )
    parser.add_argument(
        "--recalculate",
        default=False,
        action="store_true",
        help="Always compute the tables, even if they already exist (overwrites existing files).",
    )
    args = parser.parse_args()

    run_dirs = []
    for model_dir in sorted(settings.training_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        for run_dir in sorted(model_dir.glob(f"{args.timestamp}*")):
            compute_perfusion_scores(run_dir, args.recalculate)
