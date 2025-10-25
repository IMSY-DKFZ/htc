# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
from rich.progress import Progress, TimeElapsedColumn

from htc.model_processing.Runner import Runner
from htc.model_processing.SinglePredictor import SinglePredictor

if __name__ == "__main__":
    runner = Runner(
        description=(
            "Create a test table based on a trained median spectra model for a set of paths or the test paths if"
            " neither --spec nor --input-dir nor --paths-variable is set."
        ),
        default_output_to_run_dir=True,
    )
    runner.add_argument("--table-name", default="test_table_new", type=str, help="Name of the generated table")

    predictor = SinglePredictor(runner.args.model, runner.run_dir.name, test=True, config=runner.config)

    df = {
        "image_labels": [],
        "predictions": [],
        "image_name": [],
    }
    with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), refresh_per_second=1) as progress:
        task_paths = progress.add_task("Predicted paths...", total=len(runner.paths))
        for batch, logits in predictor.predict_paths(runner.paths):
            df["image_labels"] += batch["image_labels"].tolist()
            df["predictions"] += logits["class"].tolist()
            df["image_name"] += batch["image_name"]
            progress.advance(task_paths, advance=len(batch["image_name"]))

    # Already expand the image name
    meta = pd.DataFrame([p.image_name_typed() for p in runner.paths])
    meta = meta.to_dict(orient="list")
    df |= meta

    df = pd.DataFrame(df)

    df.to_pickle(runner.output_dir / f"{runner.args.table_name}.pkl.xz")
