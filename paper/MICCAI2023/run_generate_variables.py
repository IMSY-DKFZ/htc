# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc.models.data.DataSpecification import DataSpecification
from htc.settings_seg import settings_seg
from htc.utils.helper_functions import basic_statistics
from htc_projects.context.manipulated_datasets.utils import compare_performance
from htc_projects.context.models.context_evaluation import best_run_data
from htc_projects.context.settings_context import settings_context


class VariableGeneration:
    def __init__(self):
        self.df_best = best_run_data(test=True)

        self.vars = {}
        self.commands = []

    def add_dataset_statistics(self) -> None:
        mapping = settings_seg.label_mapping

        df_old = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json")

        df_glove = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2_glove.json")
        df_glove = df_glove[~pd.isna(df_glove["set_type"])]
        assert len(df_glove) < len(df_old)
        specs_glove = DataSpecification("pigs_semantic-only_5foldsV2_glove.json")
        specs_glove.activate_test_set()
        names_no_glove = [p.image_name() for p in specs_glove.paths("^test$")]
        names_glove = [p.image_name() for p in specs_glove.paths("^test_ood$")]

        df_isolation_real = basic_statistics(
            "2021_02_05_Tivita_multiorgan_masks", annotation_name="semantic#annotator5"
        )
        df_combined = pd.concat([df_old, df_isolation_real])

        self.vars["varTotalImages"] = f'{df_combined["timestamp"].nunique()}'
        self.vars["varTotalPigs"] = f'{df_combined["subject_name"].nunique()}'

        self.vars["varTotalImagesOld"] = f'{df_old["timestamp"].nunique()}'
        self.vars["varTotalPigsOld"] = f'{df_old["subject_name"].nunique()}'

        self.vars["varTotalImagesIsolationReal"] = f'{df_isolation_real["timestamp"].nunique()}'
        self.vars["varTotalPigsIsolationReal"] = f'{df_isolation_real["subject_name"].nunique()}'

        glove_images = df_old.query("label_name == 'glove'")["image_name"].nunique()
        self.vars["varTotalImagesGlove"] = str(glove_images)
        self.vars["varTotalImagesNoGlove"] = str(df_old["timestamp"].nunique() - glove_images)

        self.vars["varTotalTrainingImagesGlove"] = str(df_glove.query('set_type == "train"')["timestamp"].nunique())
        self.vars["varTotalTrainingPigsGlove"] = str(df_glove.query('set_type == "train"')["subject_name"].nunique())
        self.vars["varTotalTestImagesNoGlove"] = str(
            df_glove.query("image_name in @names_no_glove")["timestamp"].nunique()
        )
        self.vars["varTotalTestPigsNoGlove"] = str(
            df_glove.query("image_name in @names_no_glove")["subject_name"].nunique()
        )
        self.vars["varTotalTestImagesGlove"] = str(df_glove.query("image_name in @names_glove")["timestamp"].nunique())
        self.vars["varTotalTestPigsGlove"] = str(df_glove.query("image_name in @names_glove")["subject_name"].nunique())

        self.vars["varTotalTrainingPigs"] = str(df_old.query('set_type == "train"')["subject_name"].nunique())
        self.vars["varTotalTrainingImages"] = str(df_old.query('set_type == "train"')["timestamp"].nunique())

        self.vars["varTotalTestPigsInDistribution"] = str(df_old.query('set_type == "test"')["subject_name"].nunique())
        self.vars["varTotalTestImagesInDistribution"] = str(df_old.query('set_type == "test"')["timestamp"].nunique())

        self.vars["varTotalClasses"] = f"{len(mapping)}"
        self.vars["varTotalOrganClasses"] = f'{len([l for l in mapping.label_names() if l != "background"])}'

    def add_model_results(self) -> None:
        df_agg = (
            self.df_best.groupby(["network", "dataset", "modality"], as_index=False)[
                ["dice_metric", settings_seg.nsd_aggregation_short]
            ]
            .agg(["mean", "std"])
            .reset_index()
        )

        # Convert multi-index to single-index
        df_agg = df_agg.set_axis(df_agg.columns.map("_".join).str.removesuffix("_"), axis=1)

        # RGB vs. HSI performance
        df_datasets_modality = (
            df_agg.query("network == 'organ_transplantation'")
            .groupby("modality")["dice_metric_mean"]
            .agg(["mean", "std"])
        )
        self.vars["varMeanDatasetsRGB"] = (
            f"{df_datasets_modality.loc['RGB']['mean']:.2f} ("
            + "\\ac{sd}"
            + f" {df_datasets_modality.loc['RGB']['std']:.2f})"
        )
        self.vars["varMeanDatasetsHSI"] = (
            f"{df_datasets_modality.loc['HSI']['mean']:.2f} ("
            + "\\ac{sd}"
            + f" {df_datasets_modality.loc['HSI']['std']:.2f})"
        )

        baselineHSI_indist = df_agg.query("dataset == 'semantic' & network == 'baseline' & modality == 'HSI'")
        assert len(baselineHSI_indist) == 1
        self.vars["varDSCBaselineHSIInDistribution"] = (
            f"{baselineHSI_indist['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {baselineHSI_indist['dice_metric_std'].values[0]:.2f})"
        )

        baselineHSI_OOD = df_agg.query("dataset == 'masks_isolation' & network == 'baseline' & modality == 'HSI'")
        assert len(baselineHSI_OOD) == 1
        self.vars["varDSCBaselineHSIOOD"] = (
            f"{baselineHSI_OOD['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {baselineHSI_OOD['dice_metric_std'].values[0]:.2f})"
        )

        organtransHSI_OOD = df_agg.query(
            "dataset == 'masks_isolation' & network == 'organ_transplantation' & modality == 'HSI'"
        )
        assert len(organtransHSI_OOD) == 1
        self.vars["varDSCOrganTransHSIOOD"] = (
            f"{organtransHSI_OOD['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {organtransHSI_OOD['dice_metric_std'].values[0]:.2f})"
        )

        baselineRGB_indist = df_agg.query("dataset == 'semantic' & network == 'baseline' & modality == 'RGB'")
        assert len(baselineRGB_indist) == 1
        self.vars["varDSCBaselineRGBInDistribution"] = (
            f"{baselineRGB_indist['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {baselineRGB_indist['dice_metric_std'].values[0]:.2f})"
        )

        baselineRGB_OOD = df_agg.query("dataset == 'masks_isolation' & network == 'baseline' & modality == 'RGB'")
        assert len(baselineRGB_OOD) == 1
        self.vars["varDSCBaselineRGBOOD"] = (
            f"{baselineRGB_OOD['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {baselineRGB_OOD['dice_metric_std'].values[0]:.2f})"
        )

        organtransRGB_OOD = df_agg.query(
            "dataset == 'masks_isolation' & network == 'organ_transplantation' & modality == 'RGB'"
        )
        assert len(organtransRGB_OOD) == 1
        self.vars["varDSCOrganTransRGBOOD"] = (
            f"{organtransRGB_OOD['dice_metric_mean'].values[0]:.2f} ("
            + "\\ac{sd}"
            + f" {organtransRGB_OOD['dice_metric_std'].values[0]:.2f})"
        )

        # Max % drop per modality across all datasets
        performances = []
        for modality in ["HSI", "RGB"]:
            row_baseline_semantic = df_agg.query(
                "network == 'baseline' and modality == @modality and dataset == 'semantic'"
            )
            assert len(row_baseline_semantic) == 1

            # Different baseline for the glove experiment
            row_baseline_glove = df_agg.query(
                "network == 'baseline' and modality == @modality and dataset == 'no-glove'"
            )
            assert len(row_baseline_glove) == 1

            for dataset in settings_context.task_name_mapping.keys():
                row_in_distribution = row_baseline_glove if "glove" in dataset else row_baseline_semantic
                row_baseline = df_agg.query("network == 'baseline' and modality == @modality and dataset == @dataset")
                row_context = df_agg.query(
                    "network == 'organ_transplantation' and modality == @modality and dataset == @dataset"
                )

                if len(row_baseline) == 1:
                    performances.append({
                        "modality": modality,
                        "dataset": dataset,
                        # Drop for each dataset to the corresponding in-distribution dataset
                        "drop_DSC": (
                            row_in_distribution["dice_metric_mean"].item() - row_baseline["dice_metric_mean"].item()
                        )
                        / row_in_distribution["dice_metric_mean"].item(),
                        # Improvement from the baseline to the context model
                        "improvement_DSC": (
                            row_context["dice_metric_mean"].item() - row_baseline["dice_metric_mean"].item()
                        )
                        / row_baseline["dice_metric_mean"].item(),
                        "improvement_NSD": (
                            row_context[f"{settings_seg.nsd_aggregation_short}_mean"].item()
                            - row_baseline[f"{settings_seg.nsd_aggregation_short}_mean"].item()
                        )
                        / row_baseline[f"{settings_seg.nsd_aggregation_short}_mean"].item(),
                    })

        df_performance = pd.DataFrame(performances)

        # Improvements for the isolation datasets
        for modality in ["RGB", "HSI"]:
            improvement_isolation_0 = df_performance.query("modality == @modality and dataset == 'isolation_0'")[
                "improvement_DSC"
            ].item()
            improvement_isolation_cloth = df_performance.query(
                "modality == @modality and dataset == 'isolation_cloth'"
            )["improvement_DSC"].item()
            improvement_isolation_real = df_performance.query("modality == @modality and dataset == 'masks_isolation'")[
                "improvement_DSC"
            ].item()
            self.vars[f"var{modality}ImprovementIsolationZeroClothMean"] = (
                "\\SI{" + f"{(improvement_isolation_0 + improvement_isolation_cloth) * 0.5 * 100:0.0f}" + "}{\\percent}"
            )
            self.vars[f"var{modality}ImprovementIsolationReal"] = (
                "\\SI{" + f"{improvement_isolation_real * 100:0.0f}" + "}{\\percent}"
            )

        df_drops_range = (
            df_performance.query("drop_DSC > 0")
            .groupby(["modality"])
            .agg(
                min_drop_DSC=pd.NamedAgg(column="drop_DSC", aggfunc="min"),
                max_drop_DSC=pd.NamedAgg(column="drop_DSC", aggfunc="max"),
            )
        )
        self.vars["varHSIDropRange"] = (
            "\\SIrange{"
            + f"{df_drops_range.loc['HSI', 'min_drop_DSC'].item() * 100:0.0f}"
            + "}{"
            + f"{df_drops_range.loc['HSI', 'max_drop_DSC'].item() * 100:0.0f}"
            + "}{\\percent}"
        )
        self.vars["varRGBDropRange"] = (
            "\\SIrange{"
            + f"{df_drops_range.loc['RGB', 'min_drop_DSC'].item() * 100:0.0f}"
            + "}{"
            + f"{df_drops_range.loc['RGB', 'max_drop_DSC'].item() * 100:0.0f}"
            + "}{\\percent}"
        )

        # Improvement range only for the OOD datasets, i.e. the datasets where we have a drop
        df_improvements_range = (
            df_performance.query("drop_DSC > 0")
            .groupby(["modality"])
            .agg(
                min_improvement_DSC=pd.NamedAgg(column="improvement_DSC", aggfunc="min"),
                max_improvement_DSC=pd.NamedAgg(column="improvement_DSC", aggfunc="max"),
                min_improvement_NSD=pd.NamedAgg(column="improvement_NSD", aggfunc="min"),
                max_improvement_NSD=pd.NamedAgg(column="improvement_NSD", aggfunc="max"),
            )
        )
        for metric in ["DSC", "NSD"]:
            self.vars[f"varHSIImprovementRange{metric}"] = (
                "\\SIrange{"
                + f"{df_improvements_range.loc['HSI', f'min_improvement_{metric}'].item() * 100:0.0f}"
                + "}{"
                + f"{df_improvements_range.loc['HSI', f'max_improvement_{metric}'].item() * 100:0.0f}"
                + "}{\\percent}"
            )
            self.vars[f"varRGBImprovementRange{metric}"] = (
                "\\SIrange{"
                + f"{df_improvements_range.loc['RGB', f'min_improvement_{metric}'].item() * 100:0.0f}"
                + "}{"
                + f"{df_improvements_range.loc['RGB', f'max_improvement_{metric}'].item() * 100:0.0f}"
                + "}{\\percent}"
            )

        self.vars["varHSIImprovementMax"] = (
            "\\SI{" + f"{df_improvements_range.loc['HSI', 'max_improvement_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )
        self.vars["varRGBImprovementMax"] = (
            "\\SI{" + f"{df_improvements_range.loc['RGB', 'max_improvement_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )

        df_drops_max = df_performance.groupby(["modality"]).max("drop_DSC")
        self.vars["varHSIMaxDrop"] = (
            "\\SI{" + f"{df_drops_max.loc['HSI', 'drop_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )
        self.vars["varRGBMaxDrop"] = (
            "\\SI{" + f"{df_drops_max.loc['RGB', 'drop_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )

        # Load cm
        experiment_name = "organ_removal_0"
        experiment_dir = (
            settings_context.results_dir
            / "neighbour_analysis"
            / experiment_name
            / "image/2022-02-03_22-58-44_generated_default_model_comparison"
        )
        confusion_matrix, confusion_matrix_baseline, column_names = compare_performance(
            experiment_name=experiment_name, experiment_dir=experiment_dir, return_baseline_cm=True
        )

        # We are interested in the liver gallbladder case
        liver_index = column_names.index("liver")
        gallbladder_index = column_names.index("gallbladder")
        assert confusion_matrix[liver_index, gallbladder_index] < 0

        # All drop values for the gallbladder (across all removals)
        rows = []
        for l, label_name in enumerate(column_names):
            if label_name != "gallbladder":
                rows.append({
                    "label_name": label_name,
                    "drop_DSC": (
                        abs(confusion_matrix[l, gallbladder_index]) / confusion_matrix_baseline[l, gallbladder_index]
                    ).item(),
                })

        df_removal = pd.DataFrame(rows)
        df_removal = df_removal.sort_values(by="drop_DSC", ascending=False).reset_index()

        self.vars["varRemovalLiverGallbladderDrop"] = (
            "\\SI{" + f"{df_removal.iloc[0]['drop_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )
        self.vars["varRemovalRemainingAverageDrop"] = (
            "\\SI{" + f"{df_removal.iloc[1:]['drop_DSC'].mean() * 100:0.0f}" + "}{\\percent}"
        )

        # Organ improvements for the occlusion scenario (HSI)
        df_glove_baseline = self.df_best.query(
            "modality == 'HSI' and dataset == 'glove' and network == 'baseline'"
        ).reset_index(drop=True)
        df_glove_context = self.df_best.query(
            "modality == 'HSI' and dataset == 'glove' and network == 'organ_transplantation'"
        ).reset_index(drop=True)
        assert np.all(df_glove_baseline["label_name"].values == df_glove_context["label_name"].values)
        assert len(df_glove_baseline) == df_glove_baseline["label_name"].nunique()
        df_glove_best = pd.DataFrame({
            "label_name": df_glove_baseline["label_name"],
            "improvement": (df_glove_context["dice_metric"] - df_glove_baseline["dice_metric"])
            / df_glove_baseline["dice_metric"],
        })
        df_glove_best.sort_values("improvement", ascending=False, inplace=True)

        top_n = 3
        for i in range(top_n):
            self.vars[f"varHSIImprovementOcclusion{df_glove_best.iloc[i].label_name.capitalize()}"] = (
                "\\SI{" + f"{df_glove_best.iloc[i].improvement * 100:0.0f}" + "}{\\percent}"
            )

    def export_tex(self) -> None:
        tex_str = ""
        for key, value in self.vars.items():
            tex_str += "\\newcommand{" + f"\\{key}" + "}{" + f"{value}\\xspace" + "}\n"

        tex_str += "\n".join(self.commands)

        with (settings_context.paper_dir / "generated_vars.tex").open("w") as f:
            f.write(tex_str)


if __name__ == "__main__":
    generator = VariableGeneration()

    generator.add_dataset_statistics()
    generator.add_model_results()

    generator.export_tex()
