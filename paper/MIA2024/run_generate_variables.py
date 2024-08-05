# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.context.models.context_evaluation import baseline_granularity_comparison
from htc.context.neighbour.find_normalized_neighbour_matrix import find_normalized_neighbour_matrix
from htc.context.settings_context import settings_context
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.settings_seg import settings_seg
from htc.utils.Config import Config
from htc.utils.helper_functions import sort_labels, sort_labels_cm


class VariableGeneration:
    def __init__(self):
        self.vars = {}
        self.commands = []

    def neighbor_matrix(self) -> None:
        specs = DataSpecification("pigs_semantic-only_5foldsV2.json")
        specs.activate_test_set()
        config = Config({"input/no_features": True, "label_mapping": settings_seg.label_mapping})
        dataset = DatasetImage(specs.paths("test"), train=False, config=config)

        number_of_classes = len(settings_seg.label_mapping)
        labels = sort_labels(settings_seg.labels)
        normalized_neighbor_matrix = find_normalized_neighbour_matrix(dataset, number_of_classes) * 100
        normalized_neighbor_matrix = sort_labels_cm(
            normalized_neighbor_matrix,
            cm_order=settings_seg.labels,
            target_order=labels,
        )
        normalized_neighbor_matrix = normalized_neighbor_matrix.T

        value = normalized_neighbor_matrix[labels.index("liver"), labels.index("gallbladder")]
        self.vars["varGallbladderLiverNeigbor"] = "\\SI{" + f"{value:0.1f}" + "}{\\percent}"
        value = normalized_neighbor_matrix[labels.index("peritoneum"), labels.index("major_vein")]
        self.vars["varMajorVeinPeritoneum"] = "\\SI{" + f"{value:0.1f}" + "}{\\percent}"

    def granularity_scores(self) -> None:
        df = baseline_granularity_comparison(
            baseline_timestamp=settings_seg.model_comparison_timestamp,
            glove_runs_hsi=settings_context.glove_runs_granularities,
            glove_runs_rgb=settings_context.glove_runs_granularities_rgb,
        )
        df = df.groupby(["network", "dataset", "modality"]).agg(dice_metric=("dice_metric", "mean"))

        relative_changes = []
        for (network, dataset, modality), row in df.iterrows():
            baseline_dataset = "glove" if dataset == "no-glove" else "semantic"
            baseline_value = df.loc[(network, baseline_dataset, modality), "dice_metric"]

            relative_changes.append(abs((row["dice_metric"] - baseline_value) / baseline_value))

        df["relative_change_DSC"] = relative_changes
        df = df.reset_index()

        df_change = (
            df.query("dataset not in ['semantic', 'no-glove'] and network == 'image'")
            .groupby("modality")
            .agg(relative_change_DSC=("relative_change_DSC", "mean"))
        )

        self.vars["varHSIDropAverage"] = (
            "\\SI{" + f"{df_change.loc['HSI', 'relative_change_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )
        self.vars["varRGBDropAverage"] = (
            "\\SI{" + f"{df_change.loc['RGB', 'relative_change_DSC'].item() * 100:0.0f}" + "}{\\percent}"
        )

    def export_tex(self) -> None:
        tex_str = ""
        for key, value in self.vars.items():
            tex_str += "\\newcommand{" + f"\\{key}" + "}{" + f"{value}\\xspace" + "}\n"

        tex_str += "\n".join(self.commands)

        with (settings_context.paper_extended_dir / "generated_vars_extended.tex").open("w") as f:
            f.write(tex_str)


if __name__ == "__main__":
    generator = VariableGeneration()

    generator.neighbor_matrix()
    generator.granularity_scores()

    generator.export_tex()
