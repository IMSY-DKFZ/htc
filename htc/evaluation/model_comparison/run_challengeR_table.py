# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd

from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs, model_comparison_table
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.utils.import_extra import requires_extra

try:
    from challenger_pydocker import ChallengeR

    _missing_library = ""
except ImportError:
    _missing_library = "challenger_pydocker"


def challengeR_table(
    timestamp: str = settings_seg.model_comparison_timestamp,
    test: bool = False,
    metrics: list[str] = None,
    algorithm_group: str = None,
) -> pd.DataFrame:
    """
    Generates the results table with the ranking information which can be passed on to challengeR. You can group certain cases together to compare only the models (image, patch, superpixel, pixel) or the modality (hsi, tpi, rgb).

    Args:
        algorithm_group: What to be considered as algorithm, either 'all', 'model' or 'modality'.
        timestamp: Timestamp of the comparison runs which should be collected.
        test: Table for the test or validation set.
        metrics: Name of the metrics to include. Note that the metrics must exist in all the (validation|test)_table.pkl.xz files.

    Returns: Table suitable for challengeR.
    """

    # To use challengeR, the table needs to have the following columns:
    #   - task identifier (here the kind of metric)
    #   - test case identifier (here the test pig identifier)
    #   - algorithm identifier (here an identifier describing the kind of model and the used spectral resolution e.g. HSI/param/RGB)
    #   - the value of the metric

    if metrics is None:
        metrics = ["dice_metric_image", "surface_distance_metric_image", settings_seg.nsd_aggregation]

    df_runs = collect_comparison_runs(timestamp)
    df_all = model_comparison_table(df_runs, metrics=metrics, test=test)

    rows = []
    for i, row in df_all.iterrows():
        for metric in metrics:
            metric_value = 1 - row[metric] if "dice" in metric else row[metric]

            # We already simplify the names here because challengeR uses these names also for the plots (less manual adjustments needed)
            model_type = row["model_type"]
            for name_old, name_new in settings_seg.modality_names.items():
                model_type = model_type.replace(name_old, name_new)

            model_name = row["model_name"].replace("superpixel_classification", "superpixel")

            if algorithm_group == "model":
                algorithm = model_name
                case_id = row["subject_name"] + "#" + model_type
            elif algorithm_group == "modality":
                algorithm = model_type
                case_id = row["subject_name"] + "#" + model_name
            elif algorithm_group is None:
                algorithm = model_name + "#" + model_type
                case_id = row["subject_name"]
            else:
                raise ValueError("Invalid algorithm_group")

            rows.append(
                {
                    "metric": metric,
                    "case_id": case_id,
                    "algorithm": algorithm,
                    "metric_value": metric_value,
                }
            )

    df_challenge = pd.DataFrame(rows)

    return df_challenge


@requires_extra(_missing_library)
def main() -> None:
    target_dir = settings.results_dir / "challengeR/model_comparison"
    target_dir.mkdir(parents=True, exist_ok=True)

    with ChallengeR() as c:
        # Grouped tables
        df_test = challengeR_table(algorithm_group=None, test=True)
        for group in ["model", "modality"]:
            df_test = challengeR_table(algorithm_group=group, test=True)
            c.add_data_matrix(df_test)
            c.create_challenge(
                algorithm="algorithm", case="case_id", value="metric_value", task="metric", smallBetter=True
            )
            c.generate_report(target_file=target_dir / f"challengeR_test_{group}.html", title="Test Data (model type)")

        # Validation
        df_val = challengeR_table()
        c.add_data_matrix(df_val)
        c.create_challenge(algorithm="algorithm", case="case_id", value="metric_value", task="metric", smallBetter=True)
        c.generate_report(target_file=target_dir / "challengeR_validation_ranking.html", title="Validation Data")

        # Test
        df_test = challengeR_table(test=True)
        c.add_data_matrix(df_test)
        c.create_challenge(algorithm="algorithm", case="case_id", value="metric_value", task="metric", smallBetter=True)
        c.generate_report(target_file=target_dir / "challengeR_test_ranking.html", title="Test Data")


if __name__ == "__main__":
    main()
