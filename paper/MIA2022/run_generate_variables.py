# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re

import numpy as np
import pandas as pd

from htc.evaluation.metrics.scores import normalize_grouped_cm
from htc.evaluation.model_comparison.paper_runs import collect_comparison_runs, model_comparison_table
from htc.models.common.MetricAggregation import MetricAggregation
from htc.models.data.run_size_dataset import label_mapping_dataset_size
from htc.models.pixel.LightningPixel import LightningPixel
from htc.rater_variability.rater_evaluation import rater_evaluation
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.Config import Config
from htc.utils.helper_functions import basic_statistics, utilization_table
from htc.utils.sqldf import sqldf


class NumberMapping(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_mapping = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
            10: "ten",
        }
        # Numbers after 10 are written as numbers

    def __getitem__(self, number: int) -> str:
        if number in self.num_mapping:
            return self.num_mapping[number]
        else:
            return str(number)


class VariableGeneration:
    def __init__(self):
        self.num_mapping = NumberMapping()
        self.metric_mapping = {
            "DSC": "dice_metric_image",
            "ASD": "surface_distance_metric_image",
            "NSD": settings_seg.nsd_aggregation,
        }

        self.df_runs = collect_comparison_runs(settings_seg.model_comparison_timestamp)
        metrics = [*list(self.metric_mapping.values()), "confusion_matrix"]
        self.df_test = model_comparison_table(self.df_runs, test=True, metrics=metrics)

        self.vars = {}
        self.commands = []

    def _model_to_name(self, model: str) -> str:
        model = model.replace("superpixel_classification", "superpixel")

        if model == "patch_32":
            name = "PatchThreeTwo"
        elif model == "patch_64":
            name = "PatchSixFour"
        else:
            name = model.capitalize().replace("_", "")

        return name

    def add_model_info(self) -> None:
        for model, color in settings_seg.model_colors.items():
            self.commands.append("\\definecolor{" + self._model_to_name(model) + "}{HTML}{" + color[1:] + "}")

        for config_name, modality in [("default", "HSI"), ("default_parameters", "TPI"), ("default_rgb", "RGB")]:
            config = Config.from_model_name(config_name, "pixel")
            model = LightningPixel(dataset_train=None, datasets_val=[], config=config)
            self.vars[f"varPixel{modality}TotalWeights"] = (
                "\\num{" + str(sum(p.numel() for p in model.parameters())) + "}"
            )

    def add_dataset_statistics(self) -> None:
        dataset_settings = DatasetSettings(settings.data_dirs.semantic)
        mapping = settings_seg.label_mapping

        df = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json")
        df["label_index"] = [mapping.name_to_index(l) for l in df["label_name"]]
        df["label_valid"] = [mapping.is_index_valid(i) for i in df["label_index"]]

        mapping_size = label_mapping_dataset_size()
        labels_size = [l for l in settings_seg.label_colors_paper.keys() if l in mapping_size.label_names()]
        labels_size = [settings_seg.labels_paper_renaming.get(l, l) for l in labels_size]
        labels_size = ["\\oname{" + l + "}" for l in labels_size]

        df_organ = sqldf(f"""
            SELECT label_name, COUNT(DISTINCT subject_name) AS n_pigs, COUNT(DISTINCT timestamp) AS n_images
            FROM df
            WHERE label_valid = 1 AND label_index != {mapping.name_to_index("background")}
            GROUP BY label_name
        """)

        df_background = sqldf(f"""
            SELECT timestamp, CAST(SUM(n_pixels) AS FLOAT) / {dataset_settings.pixels_image()} AS pixel_ratio
            FROM df
            WHERE label_index = {mapping.name_to_index("background")}
            GROUP BY timestamp
        """)
        assert len(df_background) == df["timestamp"].nunique(), "Every image must have background"

        df_invalid = sqldf(f"""
            SELECT timestamp, CAST(SUM(n_pixels) AS FLOAT) / {dataset_settings.pixels_image()} AS pixel_ratio
            FROM df
            WHERE label_valid = 0
            GROUP BY timestamp
        """)

        # Neighbour information about the gallbladder
        df_labels = sqldf(f"""
            SELECT timestamp, GROUP_CONCAT(label_name, ',') AS all_labels
            FROM df
            WHERE label_valid = 1 AND label_index != {mapping.name_to_index("background")}
            GROUP BY timestamp
        """)
        counts = dict.fromkeys(mapping.label_names(), 0)
        for i, row in df_labels.iterrows():
            labels = row["all_labels"].split(",")
            if "gallbladder" in labels:
                for label in labels:
                    counts[label] += 1

        df_counts = pd.DataFrame(counts.items(), columns=["label_name", "count"])
        df_counts.sort_values(by="count", ascending=False, inplace=True, ignore_index=True)
        top3_labels = df_counts.query('label_name != "gallbladder"')["label_name"].values[:3]
        top3_counts = df_counts.query('label_name != "gallbladder"')["count"].values[:3]

        self.vars["varTotalImages"] = f"{df['timestamp'].nunique()}"
        self.vars["varTotalPigs"] = f"{df['subject_name'].nunique()}"
        self.vars["varTotalTrainingPigs"] = str(df.query('set_type == "train"')["subject_name"].nunique())
        self.vars["varTotalTrainingImages"] = str(df.query('set_type == "train"')["timestamp"].nunique())
        self.vars["varMaxTrainingPigsDatasetSize"] = str(df.query('set_type == "train"')["subject_name"].nunique() - 1)
        self.vars["varTotalClassesDatasetSize"] = self.num_mapping[len(labels_size)]
        self.vars["varClassesDatasetSize"] = ", ".join(labels_size[:-1]) + f" and {labels_size[-1]}"
        self.vars["varTotalOrganClasses"] = f"{len([l for l in mapping.label_names() if l != 'background'])}"
        self.vars["varTotalClasses"] = f"{len(mapping)}"

        self.vars["varMinPigsOrgan"] = f"{df_organ['n_pigs'].min()}"
        self.vars["varMaxPigsOrgan"] = f"{df_organ['n_pigs'].max()}"
        self.vars["varMinImagesOrgan"] = f"{df_organ['n_images'].min()}"
        self.vars["varMaxImagesOrgan"] = f"{df_organ['n_images'].max()}"

        self.vars["varInvalidImages"] = f"{len(df_invalid)}"
        self.vars["varBackgroundPixelRatio"] = (
            "\\SI{"
            + f"{100 * df_background['pixel_ratio'].mean():0.0f}"
            + "}{\\percent} ("
            + "\\ac{sd} \\SI{"
            + f" {100 * df_background['pixel_ratio'].std():0.0f}"
            + "}{\\percent})"
        )
        self.vars["varInvalidPixelRatio"] = (
            "\\SI{"
            + f"{100 * df_invalid['pixel_ratio'].mean():0.0f}"
            + "}{\\percent} ("
            + "\\ac{sd} \\SI{"
            + f" {100 * df_invalid['pixel_ratio'].std():0.0f}"
            + "}{\\percent})"
        )

        self.vars["varGallbladderImages"] = str(df_counts.query('label_name == "gallbladder"')["count"].item())
        self.vars["varGallbladderNeighbourLabels"] = f"{top3_labels[0]}, {top3_labels[1]} and {top3_labels[2]}"
        self.vars["varGallbladderNeighbourCounts"] = f"{top3_counts[0]}, {top3_counts[1]} and {top3_counts[2]}"

    def add_epoch_table(self) -> None:
        table = """\\begin{tabular}{llll}
\\toprule
model & \\# pixels & epoch size & batch size \\\\
\\midrule
"""

        dataset_settings = DatasetSettings(settings.data_dirs.semantic)
        pixels_mapping = {
            "image": dataset_settings.pixels_image(),
            "patch_64": 64 * 64,
            "patch_32": 32 * 32,
            "pixel": 1,
        }

        for idx in reversed(self.df_runs.index):
            row = self.df_runs.loc[idx]
            # for i, row in reversed(self.df_runs.iterrows()):
            config = Config(settings.training_dir / row["model"] / row["run_hsi"] / "config.json")

            if row["name"] == "superpixel_classification":
                n_superpixels = pixels_mapping["image"] / config["input/superpixels/n_segments"]
                # Round to the next hundred
                n_pixels = "$\\approx " + str(int(round(n_superpixels / 100.0)) * 100) + "$"
            else:
                n_pixels = "\\num{" + str(pixels_mapping[row["name"]]) + "}"

            name = row["name"].replace("superpixel_classification", "superpixel").replace("_", "\\_")

            table += (
                name
                + " & "
                + n_pixels
                + " & \\num{"
                + str(config["input/epoch_size"])
                + "} & \\num{"
                + str(config["dataloader_kwargs/batch_size"])
                + "} \\\\\n"
            )

        table += "\\bottomrule\n"
        table += "\\end{tabular}"

        self.vars["varEpochTable"] = table

    def add_improvement_percent(self) -> None:
        for model in self.df_test["model_name"].unique():
            rgb = self.df_test.query('model_name == @model and model_type == "rgb"')["dice_metric_image"].mean()
            hsi = self.df_test.query('model_name == @model and model_type == "hsi"')["dice_metric_image"].mean()
            name = self._model_to_name(model)

            self.vars[f"var{name}Improvement"] = "\\SI{" + f"{(hsi / rgb - 1) * 100:0.1f}" + "}{\\percent}"

    def add_model_results(self) -> None:
        df_test = self.df_test.query('model_type == "hsi"')
        for model in df_test["model_name"].unique():
            name = self._model_to_name(model)

            for short, long in self.metric_mapping.items():
                values = df_test.query("model_name == @model")[long]
                self.vars[f"varPerformance{name}{short}"] = (
                    f"{values.mean():0.2f} (" + "\\ac{sd}" + f" {values.std():0.2f})"
                )

    def add_confusion_matrix_values(self) -> None:
        cm_stacked = np.stack(
            self.df_test.query('model_name == "image" and model_type == "hsi"')["confusion_matrix"].values
        )
        cm_rel, _ = normalize_grouped_cm(cm_stacked)
        mapping = settings_seg.label_mapping

        # Best classes
        cm_thresh_good = 0.95
        self.vars["varCMThreshold"] = "\\SI{" + f"{cm_thresh_good * 100:.0f}" + "}{\\percent}"
        self.vars["varCMTotalClassesAboveThreshold"] = sum(np.diag(cm_rel) > cm_thresh_good)

        # We describe the vena case in more detail
        major_vein_index = mapping.name_to_index("major_vein")
        major_vein_cm = cm_rel[major_vein_index, :].copy()
        major_vein_cm[major_vein_index] = 0
        df_stats = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json")
        df_major_vein = df_stats.query('label_name == "major_vein"')

        self.vars["varCMVenaCavaSensitivity"] = (
            "\\SI{" + f"{cm_rel[major_vein_index, major_vein_index] * 100:0.1f}" + "}{\\percent}"
        )
        self.vars["varCMVenaCavaMaxConfusion"] = "\\SI{" + f"{np.max(major_vein_cm) * 100:0.1f}" + "}{\\percent}"
        self.vars["varCMVenaCavaMaxConfusionClass"] = mapping.index_to_name(np.argmax(major_vein_cm))
        self.vars["varCMVenaCavaTotalImages"] = df_major_vein["timestamp"].nunique()
        self.vars["varCMVenaCavaPixels"] = (
            "\\SI{"
            + f"{df_major_vein['n_pixels'].mean():.0f}"
            + "}{\\px} (\\ac{sd} \\SI{"
            + f"{df_major_vein['n_pixels'].std():.0f}"
            + "}{\\px})"
        )

    def add_rater_info(self) -> None:
        for rater, name in [("semantic#inter1", "Inter"), ("semantic#intra1", "Intra")]:
            metrics, stats = rater_evaluation(rater)

            self.vars[f"varRater{name}Additional"] = len(stats["additional_labels"])
            self.vars[f"varRater{name}Missing"] = len(stats["missing_labels"])
            self.vars[f"varRater{name}TotalMaskDiffPixels"] = "\\SI{" + f"{sum(stats['mask_differences'])}" + "}{\\px}"
            self.vars[f"varRater{name}TotalMaskDiffImages"] = len(stats["mask_differences"])

            for short, long in self.metric_mapping.items():
                self.vars[f"varRater{name}{short}"] = (
                    f"{np.mean(metrics[long]):0.2f} (" + "\\ac{sd}" + f" {np.std(metrics[long]):0.2f})"
                )

        self.vars["varRaterTotalImages"] = metrics["timestamp"].nunique()

    def add_seed_variation(self) -> None:
        runs = sorted(
            settings.training_dir.glob(f"image/{settings_seg.seed_experiment_timestamp}_generated_default_seed=*")
        )

        rows = []
        for run_dir in runs:
            match = re.search(r"seed=(\d+)", run_dir.name)
            assert match is not None
            seed = int(match.group(1))

            current_row = {"seed": seed}
            agg = MetricAggregation(run_dir / "test_table.pkl.xz", metrics=list(self.metric_mapping.values()))
            df = agg.grouped_metrics(mode="image_level")
            for metric in self.metric_mapping.values():
                current_row[metric] = df[metric].mean()
            rows.append(current_row)

        df = pd.DataFrame(rows)
        for short, long in self.metric_mapping.items():
            value_min = df[long].min()
            value_max = df[long].max()

            self.vars[f"varSeedVariationTest{short}"] = f"[{value_min:0.3f}; {value_max:0.3f}]"

    def add_runtime_stats(self) -> None:
        hours = []

        for i, row_runs in self.df_runs.iterrows():
            for model_type in ["rgb", "param", "hsi"]:
                run_dir = settings.training_dir / row_runs["model"] / row_runs[f"run_{model_type}"]
                hours += utilization_table(run_dir)["hours"].values.tolist()

        self.vars["varTrainingTimeTotal"] = "\\SI{" + f"{sum(hours):0.0f}" + "}{\\hour}"

    def add_spx_results(self) -> None:
        from htc.models.data.run_pig_dataset import test_set  # noqa: F401

        df = pd.read_pickle(settings.results_dir / "superpixel_gt" / "spxs_predictions.pkl.xz")
        df_grouped = sqldf("""
            SELECT subject_name, AVG(dice) AS DSC, AVG(asd) AS ASD, AVG(nsd) AS NSD
            FROM df
            GROUP BY subject_name
        """).query("subject_name in @test_set")

        for short in self.metric_mapping.keys():
            self.vars[f"varSpxLimit{short}"] = (
                f"{df_grouped[short].mean():0.2f} (" + "\\ac{sd}" + f" {df_grouped[short].std():0.2f})"
            )

    def add_image_example_values(self) -> None:
        # Collect all dice values
        rows = []

        for i, row_runs in self.df_runs.iterrows():
            for model_type in ["rgb", "param", "hsi"]:
                run_dir = settings.training_dir / row_runs["model"] / row_runs[f"run_{model_type}"]
                df_test = pd.read_pickle(run_dir / "test_table.pkl.xz")
                df_test["subject_name"], df_test["timestamp"] = zip(
                    *df_test["image_name"].map(lambda x: x.split("#")), strict=True
                )

                for j, row_test in df_test.iterrows():
                    rows.append([
                        row_runs["name"],
                        model_type,
                        row_test["subject_name"],
                        row_test["timestamp"],
                        row_test["dice_metric_image"],
                        row_test["surface_distance_metric_image"],
                        row_test[settings_seg.nsd_aggregation],
                    ])

        df = pd.DataFrame(rows, columns=["name", "model_type", "subject_name", "timestamp", "dice", "asd", "nsd"])

        # Average dice across models
        df_dice = sqldf("""
            SELECT subject_name, timestamp, model_type, AVG(dice) AS dice
            FROM df
            WHERE model_type = 'hsi'
            GROUP BY subject_name, timestamp, model_type
            ORDER BY dice
        """)

        # Extract information for pixel model
        img = df_dice.iloc[round(0.95 * len(df_dice))]
        entry = df.query('model_type == "hsi" and timestamp == @img.timestamp and name == "pixel"')

        self.vars["varImageExamplesPixelDSC"] = f"{entry['dice'].item():0.2f}"
        self.vars["varImageExamplesPixelASD"] = f"{entry['asd'].item():0.2f}"

    def add_threshold_info(self) -> None:
        df_thresholds = pd.read_csv(settings_seg.nsd_tolerances_path)
        used_aggregation = settings_seg.nsd_aggregation.split("_")[-1]

        skin_factor = (
            df_thresholds.query('label_name == "skin"')["tolerance_mean_std"].item()
            / df_thresholds.query('label_name == "skin"')["tolerance_mean"].item()
        )

        self.vars["varThresholdsAggregation"] = used_aggregation
        self.vars["varThresholdsAggregationLow"] = (
            "\\SI{" + str(int(np.ceil(df_thresholds["tolerance_mean"].max() / 10.0)) * 10) + "}{\\px}"
        )
        self.vars["varThresholdsAggregationHigh"] = (
            "\\SI{" + str(int(np.floor(df_thresholds["tolerance_q95"].max() / 10.0)) * 10) + "}{\\px}"
        )
        self.vars["varThresholdsAggregationSkinFactor"] = f"{skin_factor:0.1f}"

    def add_lr_info(self) -> None:
        df_runs_fixed = pd.read_pickle(settings.results_dir / "challengeR/lrs/runs_fixed.pkl.xz")
        df_runs_best = pd.read_pickle(settings.results_dir / "challengeR/lrs/runs_best.pkl.xz")

        df_dice_fixed = (
            df_runs_fixed.query('metric == "dice_metric_image"')
            .groupby("model_id", as_index=False)["metric_value"]
            .mean()
            .sort_values("model_id")
        )
        df_dice_best = (
            df_runs_best.query('metric == "dice_metric_image"')
            .groupby("model_id", as_index=False)["metric_value"]
            .mean()
            .sort_values("model_id")
        )

        max_diff = (df_dice_fixed["metric_value"] - df_dice_best["metric_value"]).max()
        digits = int(np.ceil(np.abs(np.log10(max_diff))))

        self.vars["varLrDSCDiff"] = round(
            max_diff, digits
        )  # Round to the nearest higher value (e.g. 0.006796514707378742 --> 0.007)
        self.vars["varLrDefault"] = settings_seg.lr_default
        self.vars["varLrHigher"] = settings_seg.lr_higher
        self.vars["varLrLower"] = settings_seg.lr_lower

    def add_inference_time(self) -> None:
        df = pd.read_pickle(settings.results_dir / "misc/inference_times.pkl.xz")
        times = df["time [ms]"].values
        self.vars["varInferenceTime"] = "\\SI{" + f"{np.mean(times):0.0f}" + "}{\\ms}"

    def export_tex(self) -> None:
        tex_str = ""
        for key, value in self.vars.items():
            tex_str += "\\newcommand{" + f"\\{key}" + "}{" + f"{value}\\xspace" + "}\n"

        tex_str += "\n".join(self.commands)

        with (settings_seg.paper_dir / "generated_vars.tex").open("w") as f:
            f.write(tex_str)


if __name__ == "__main__":
    generator = VariableGeneration()

    generator.add_model_info()
    generator.add_dataset_statistics()
    generator.add_epoch_table()
    generator.add_improvement_percent()
    generator.add_rater_info()
    generator.add_seed_variation()
    generator.add_runtime_stats()
    generator.add_spx_results()
    generator.add_model_results()
    generator.add_confusion_matrix_values()
    generator.add_image_example_values()
    generator.add_threshold_info()
    generator.add_lr_info()
    generator.add_inference_time()

    generator.export_tex()
