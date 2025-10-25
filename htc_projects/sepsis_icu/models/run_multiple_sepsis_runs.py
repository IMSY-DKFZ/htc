# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import json
from collections.abc import Callable
from datetime import datetime
from functools import partial

import numpy as np

from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.general import subprocess_run
from htc.utils.parallel import p_map
from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.utils import config_meta_selection


def generate_run(
    timestring: str,
    config_name: str,
    config_adjustments: list[Callable[[str, Config], str | None]],
    **kwargs,
) -> dict[str, str]:
    config_dir = settings.htc_projects_dir / "sepsis_icu/configs"
    config = Config(config_dir / f"{config_name}.json")
    base_name = config_name

    for adjuster in config_adjustments:
        base_name = adjuster(base_name, config, **kwargs)

    # No progress bar needed if jobs are run in parallel
    config["trainer_kwargs/enable_progress_bar"] = False

    # Store config
    generated_config_name = f"generated_{base_name}"
    config["config_name"] = generated_config_name
    filename = generated_config_name + ".json"
    config.save_config(config_dir / filename)

    run_name = f"{timestring}_{base_name}"
    if "_image" in config_name:
        model = "image"
    elif "_median" in config_name:
        model = "median_pixel"
    elif "_meta" in config_name:
        model = "meta"
    else:
        raise ValueError(f"Cannot extract model name from {config_name}")

    config_file = config_dir.relative_to(settings.htc_projects_dir) / filename
    return {
        "command": f"htc training --model {model} --config {config_file} --run-folder {run_name}",
        "run_folder": run_name,
    }


def run_multiple_sepsis_runs(runs_input: list[dict[str, str]], n_cpus: int = 3) -> None:
    known_commands = set()
    runs = []
    for r in runs_input:
        if r["command"] not in known_commands:
            known_commands.add(r["command"])
            runs.append(r)

    n_generated_runs = len(runs)

    # Check which runs already exist
    if args.timestamp is not None:
        existing_run_folders = set()
        for model_dir in sorted(settings.training_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue

                existing_run_folders.add(run_dir.name)

        runs = [r for r in runs if r["run_folder"] not in existing_run_folders]

    if len(runs) == 0:
        print(f"All {n_generated_runs} runs already exist")
        return

    commands = [r["command"] for r in runs]
    print(f"Starting the training of {len(runs)} runs:")
    print("\n".join(commands))

    print("Number of generated runs:", n_generated_runs)
    print("Number of runs that already exist:", n_generated_runs - len(runs))
    print("Number of new runs:", len(runs))

    res = p_map(partial(subprocess_run, shell=True), commands, num_cpus=n_cpus, use_threads=True)
    if not all(r.returncode == 0 for r in res):
        for r, c in zip(res, commands, strict=True):
            if r.returncode != 0:
                print("The following run failed:")
                print(f"Command: {c}")

        raise ValueError("Some runs failed")


def nested_cv_adjustment(base_name: str, config: Config, nested_fold_index: int, max_fold_index: int, **kwargs) -> str:
    config["input/data_spec"] = config["input/data_spec"].replace("test-0.25", f"nested-{nested_fold_index}")
    return base_name + f"_nested-{nested_fold_index}-{max_fold_index}"


def train_all_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/data_spec"] = config["input/data_spec"].replace("inclusion", "inclusion-train-all")
    return base_name.replace("inclusion", "inclusion-train-all")


def finger_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/data_spec"] = config["input/data_spec"].replace("palm", "finger")
    return base_name.replace("palm", "finger")


def septic_shock_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/n_classes"] = 2
    config["input/image_labels"] = [
        {
            "meta_attributes": ["septic_shock"],
            "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>shock_label_mapping",
        }
    ]
    config["input/data_spec"] = config["input/data_spec"].replace("sepsis", "septic_shock")

    return base_name.replace("sepsis", "septic_shock")


def shock_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/n_classes"] = 2
    config["input/image_labels"] = [
        {
            "meta_attributes": ["shock"],
            "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>shock_label_mapping",
        }
    ]
    config["input/data_spec"] = config["input/data_spec"].replace("sepsis", "shock")

    return base_name.replace("sepsis", "shock")


def survival_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/n_classes"] = 2
    config["input/image_labels"] = [
        {
            "meta_attributes": ["survival_30_days_post_inclusion"],
            "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>survival_label_mapping",
        }
    ]
    config["input/data_spec"] = config["input/data_spec"].replace("sepsis", "survival")

    return base_name.replace("sepsis", "survival")


def meta_adjustment(base_name: str, config: Config, meta_set_name: str, meta_base_name: str = None, **kwargs) -> str:
    if "meta" not in base_name:
        return base_name

    groups = meta_set_name.split("+")
    config_meta_selection(config, groups)

    assert len(config["input/meta/attributes"]) > 0, f"No attributes found for groups {groups}"

    if meta_base_name is not None:
        return base_name + f"_{meta_base_name}"
    else:
        return base_name + f"_{meta_set_name}"


def tpi_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/preprocessing"] = "parameter_images_recalibrated_crop_reshape_224+224"
    config["input/parameter_names"] = ["StO2", "NIR", "TWI", "THI"]
    config["input/n_channels"] = 4
    return base_name + "_tpi"


def stacking_adjustment(base_name: str, config: Config, **kwargs) -> str:
    config["input/preprocessing"] = "L1_recalibrated_crop_reshape_224+224_stacked"
    config["input/n_channels"] = 200
    return base_name.replace("palm", "palm_stacked")


def lr_adjustment(base_name: str, config: Config, lr: float, **kwargs) -> str:
    config["optimization/optimizer/lr"] = lr
    return base_name + f"_lr={lr}"


def weight_decay_adjustment(base_name: str, config: Config, weight_decay: float, **kwargs) -> str:
    config["optimization/optimizer/weight_decay"] = weight_decay
    return base_name + f"_weight_decay={weight_decay}"


def gamma_adjustment(base_name: str, config: Config, gamma: float, **kwargs) -> str:
    config["optimization/lr_scheduler/gamma"] = gamma
    return base_name + f"_gamma={gamma}"


def batch_size_adjustment(base_name: str, config: Config, batch_size: int, **kwargs) -> str:
    config["dataloader_kwargs/batch_size"] = batch_size
    return base_name + f"_batch_size={batch_size}"


def epoch_size_adjustment(base_name: str, config: Config, epoch_size: int, **kwargs) -> str:
    config["input/epoch_size"] = epoch_size
    return base_name + f"_epoch_size={epoch_size}"


def dropout_adjustment(base_name: str, config: Config, dropout: float, **kwargs) -> str:
    config["model/dropout"] = dropout
    return base_name + f"_dropout={dropout}"


def seed_adjustment(base_name: str, config: Config, seed: int, max_seed_index: int, **kwargs) -> str:
    config["seed"] = seed
    return base_name + f"_seed-{seed}-{max_seed_index}"


def random_mixup_augmentation(base_name: str, config: Config, probability: float = 0.8, **kwargs) -> str:
    transform_dict = {
        "class": "htc_projects.context.context_transforms>RandomMixUp",
        "p": probability,
    }
    config["input/transforms_gpu"].append(transform_dict)

    return base_name + "_mixup"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all sepsis and survival training runs", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Explicit timestamp to use for the run names. If set, only missing runs will be computed.",
    )
    args = parser.parse_args()

    if args.timestamp is not None:
        timestring = args.timestamp
    else:
        timestring = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    seeds = [0, 1, 2]
    nested_fold_indices = np.arange(5)

    # Add baseline runs
    baseline_runs = []
    for config_name in [
        "sepsis-inclusion_palm_image",
        "sepsis-inclusion_palm_image_rgb",
        "sepsis-inclusion_palm_image-meta",
        "sepsis-inclusion_palm_median",
    ]:
        for meta_set_name in [
            "demographic+vital+BGA+diagnosis+ventilation+catecholamines",
            "demographic+vital+BGA+diagnosis+ventilation+catecholamines+lab",
        ]:
            if config_name == "sepsis-inclusion_palm_image":
                use_tpis = [False, True]
            else:
                use_tpis = [False]

            if "image-meta" not in config_name:
                use_fingers = [False, True]
            else:
                use_fingers = [False]
            for use_tpi in use_tpis:
                for use_finger in use_fingers:
                    for seed in seeds:
                        for nested_fold_index in nested_fold_indices:
                            for target in [None, "survival", "shock", "septic_shock"]:
                                adjustments = []
                                if target == "survival":
                                    adjustments.append(survival_adjustment)
                                elif target == "shock":
                                    adjustments.append(shock_adjustment)
                                elif target == "septic_shock":
                                    adjustments.append(septic_shock_adjustment)

                                if use_tpi:
                                    adjustments.append(tpi_adjustment)
                                adjustments.append(meta_adjustment)
                                if use_finger:
                                    adjustments.append(finger_adjustment)
                                adjustments.append(nested_cv_adjustment)
                                adjustments.append(seed_adjustment)

                                r = generate_run(
                                    timestring,
                                    config_name,
                                    adjustments,
                                    meta_set_name=meta_set_name,
                                    seed=seed,
                                    max_seed_index=np.max(seeds),
                                    nested_fold_index=nested_fold_index,
                                    max_fold_index=np.max(nested_fold_indices),
                                )
                                baseline_runs.append(r)
    run_multiple_sepsis_runs(baseline_runs)

    # Add runs for the evaluation of combining palm and finger measurements
    stacked_runs = []
    config_name = "sepsis-inclusion_palm_image"
    for target in [None, "survival", "shock", "septic_shock"]:
        for seed in seeds:
            for nested_fold_index in nested_fold_indices:
                adjustments = []
                if target == "survival":
                    adjustments.append(survival_adjustment)
                elif target == "shock":
                    adjustments.append(shock_adjustment)
                elif target == "septic_shock":
                    adjustments.append(septic_shock_adjustment)
                adjustments.append(nested_cv_adjustment)
                adjustments.append(seed_adjustment)
                adjustments.append(stacking_adjustment)

                r = generate_run(
                    timestring,
                    config_name,
                    adjustments,
                    seed=seed,
                    max_seed_index=np.max(seeds),
                    nested_fold_index=nested_fold_index,
                    max_fold_index=np.max(nested_fold_indices),
                )
                stacked_runs.append(r)
    run_multiple_sepsis_runs(stacked_runs, n_cpus=2)

    # Add runs for testing the sequential addition of metadata features according to their feature importance ranking
    metadata_ranking_path = settings_sepsis_icu.results_dir / "feature_importance_rankings.json"
    with metadata_ranking_path.open("r") as f:
        metadata_ranking_dict = json.load(f)

    metadata_adding_runs = []
    config_name = "sepsis-inclusion_palm_image-meta"
    for target in ["survival", "sepsis"]:
        for timedelta in [1, 10]:
            ranking_order = metadata_ranking_dict[target][str(timedelta)]
            for i in np.arange(1, len(ranking_order) + 1):
                meta_set_name = "+".join(ranking_order[:i])
                meta_base_name = f"top-{i}-features-{timedelta}hrs"
                for seed in seeds:
                    for nested_fold_index in nested_fold_indices:
                        adjustments = []
                        if target == "survival":
                            adjustments.append(survival_adjustment)
                        adjustments.append(meta_adjustment)
                        adjustments.append(nested_cv_adjustment)
                        adjustments.append(seed_adjustment)

                        r = generate_run(
                            timestring,
                            config_name,
                            adjustments,
                            meta_set_name=meta_set_name,
                            seed=seed,
                            max_seed_index=np.max(seeds),
                            nested_fold_index=nested_fold_index,
                            max_fold_index=np.max(nested_fold_indices),
                            meta_base_name=meta_base_name,
                        )
                        metadata_adding_runs.append(r)

    run_multiple_sepsis_runs(metadata_adding_runs)
