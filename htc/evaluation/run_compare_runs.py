# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import argparse
import re
from functools import partial
from pathlib import Path

import pandas as pd

from htc.models.common.MetricAggregation import MetricAggregation
from htc.settings import settings
from htc.utils.Config import Config
from htc.utils.helper_functions import get_valid_run_dirs
from htc.utils.import_extra import requires_extra
from htc.utils.parallel import p_map
from htc.utils.sqldf import sqldf

try:
    from challenger_pydocker import ChallengeR

    _missing_library = ""
except ImportError:
    _missing_library = "challenger_pydocker"


def get_algorithm(run_dir: Path, parameter: str | None) -> str:
    if parameter is None:
        return re.sub(r"^[\d-]+_[\d-]+_(?:generated_)?", "", run_dir.name).replace("_64", "")
    else:
        regex = parameter + r"=([^,]+)"
        match = re.search(regex, run_dir.name)
        assert match is not None, f"Cannot find the parameter with the regex {regex} in the run name {run_dir.name}"

        return match.group(1)


def get_task(run_dir: Path) -> str:
    model = run_dir.parent.name
    if model == "patch":
        config = Config(run_dir / "config.json")
        model = f"{model}_{config['input/patch_size'][0]}"

    return model


def table_dice_metric_image(
    run_dir: Path, table_name: str, parameter: str = None, single_task: bool = False
) -> pd.DataFrame:
    df_val = MetricAggregation(run_dir / table_name).grouped_metrics(mode="image_level")
    df_val["algorithm"] = get_algorithm(run_dir, parameter)
    df_val["task"] = "single" if single_task else get_task(run_dir)
    df_val.rename(columns={"dice_metric": "metric_value", "subject_name": "case_id"}, inplace=True)

    return df_val


def table_dice_metric_class(
    run_dir: Path, table_name: str, parameter: str = None, single_task: bool = False
) -> pd.DataFrame:
    config = Config(run_dir / "config.json")
    df_val = MetricAggregation(run_dir / table_name, config).grouped_metrics(mode="class_level")

    df = pd.DataFrame({"metric_value": df_val["dice_metric"], "case_id": df_val["label_name"]})
    df["algorithm"] = get_algorithm(run_dir, parameter)
    df["task"] = "single" if single_task else get_task(run_dir)

    return df


@requires_extra(_missing_library)
def main(args: argparse.Namespace) -> None:
    run_dirs = get_valid_run_dirs()
    run_dirs = [r for r in run_dirs if re.search(args.filter, str(r)) is not None]
    assert len(run_dirs) > 0, f"Could not find any runs with the filter {args.filter}"

    table_name = "test_table.pkl.xz" if args.test else "validation_table.pkl.xz"
    f = table_dice_metric_class if args.per_class else table_dice_metric_image
    df = pd.concat(
        p_map(partial(f, table_name=table_name, parameter=args.parameter, single_task=args.single_task), run_dirs)
    )

    target_dir = settings.results_dir / "challengeR"
    target_dir.mkdir(exist_ok=True, parents=True)
    report_file = target_dir / "challengeR_comparison.html"

    settings.log.info("Creating a challengeR report based on the following runs:")
    settings.log.info(sqldf("SELECT DISTINCT algorithm, task FROM df").to_string())
    settings.log.info(f"The report will be stored in {report_file}")

    with ChallengeR() as c:
        c.add_data_matrix(df)
        c.create_challenge(algorithm="algorithm", case="case_id", value="metric_value", task="task", smallBetter=False)
        c.generate_report(target_file=report_file, title="Comparison")


if __name__ == "__main__":
    # python run_compare_runs.py --filter "(?:generated_default_background_weight|generated_default_class_weight_method=1)"
    parser = argparse.ArgumentParser(
        description="This script allows to compare different runs from different models with the help of challengeR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--filter", required=True, type=str, help="Regex to filter specific run dirs.")
    parser.add_argument("--parameter", default=None, type=str, help="Name of the parameter which is used as algorithm.")
    parser.add_argument(
        "--single-task",
        default=False,
        action="store_true",
        help="Do not use model names (patch_32, image, etc.) as tasks",
    )
    parser.add_argument("--test", default=False, action="store_true", help="Run comparison on the test set.")
    parser.add_argument(
        "--per-class", default=False, action="store_true", help="Compute metric per class instead across images."
    )

    main(parser.parse_args())
