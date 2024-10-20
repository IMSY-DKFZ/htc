# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc_projects.species.settings_species import settings_species
from htc_projects.species.species_evaluation import baseline_performance
from htc_projects.species.tables import paper_table


class VariableGeneration:
    def __init__(self):
        self.df = paper_table()

        self.vars = {}
        self.commands = []

    def add_dataset_statistics(self) -> None:
        self.vars["varTotalImages"] = "\\num{" + f'{self.df["image_name"].nunique()}' + "}"
        self.vars["varTotalSubjects"] = "\\num{" + f'{self.df["subject_name"].nunique()}' + "}"

        df_annotation_counts = self.df[["image_name", "annotation_name"]].drop_duplicates(ignore_index=True)
        df_annotation_counts = df_annotation_counts.annotation_name.value_counts(dropna=False)
        assert len(df_annotation_counts) == 3, "There should be three different types of annotations"
        self.vars["varTotalImagesSemantic"] = "\\num{" + f'{df_annotation_counts["semantic#primary"]}' + "}"
        self.vars["varTotalImagesPolygon"] = (
            "\\num{"
            + f'{df_annotation_counts["polygon#annotator1"] + df_annotation_counts["polygon#malperfused"]}'
            + "}"
        )

        df_phys = self.df[self.df.baseline_dataset | self.df.standardized_recordings]
        df_mal = self.df[~(self.df.baseline_dataset | self.df.standardized_recordings)]

        self.vars["varTotalImagesPhys"] = "\\num{" + f'{df_phys["image_name"].nunique()}' + "}"

        for species in settings_species.species_colors.keys():
            df_species = self.df[self.df.species_name == species]
            self.vars[f"varTotalImages{species.capitalize()}"] = (
                "\\num{" + f'{df_species["image_name"].nunique()}' + "}"
            )
            self.vars[f"varTotalSubjects{species.capitalize()}"] = (
                "\\num{" + f'{df_species["subject_name"].nunique()}' + "}"
            )

            self.vars[f"varTotalImagesMal{species.capitalize()}"] = (
                "\\num{" + f'{df_mal[df_mal.species_name == species]["image_name"].nunique()}' + "}"
            )

    def baseline_performance_scores(self) -> None:
        df = baseline_performance()

        df_in = df[df.source_species == df.target_species]
        df_in = df_in.groupby("target_species")["dice_metric"].agg(["mean", "std"])

        for species in settings_species.species_colors.keys():
            self.vars[f"varDSCInSpecies{species.capitalize()}"] = (
                f"{df_in.loc[species, 'mean']:.2f} (" + "\\ac{sd}" + f" {df_in.loc[species, 'std']:.2f})"
            )

        df_out = df[
            (df.source_species != df.target_species)
            & (df.source_species != "human")
            & (df.source_species != "pig-p+rat-p2human")
        ]
        df_out = df_out.groupby(["source_species", "target_species"], as_index=False)["dice_metric"].agg([
            "mean",
            "std",
        ])
        change = []
        for _, row in df_out.iterrows():
            in_species = df_in.loc[row["target_species"], "mean"]
            change.append((row["mean"] - in_species) / in_species)
        df_out["change"] = change
        df_out = df_out.sort_values("change", ascending=False)

        drop_min = df_out.iloc[-1]
        drop_max = df_out.iloc[0]
        self.vars["varHSIDropRanging"] = (
            "\\SI{"
            + f"{drop_max['change'] * 100:0.0f}"
            + "}{\\percent}"
            + f" ({drop_max['source_species']}2{drop_max['target_species']}) to "
            + "\\SI{"
            + f"{drop_min['change'] * 100:0.0f}"
            + "}{\\percent}"
            + f" ({drop_min['source_species']}2{drop_min['target_species']})"
        )

    def export_tex(self) -> None:
        tex_str = ""
        for key, value in self.vars.items():
            tex_str += "\\newcommand{" + f"\\{key}" + "}{" + f"{value}\\xspace" + "}\n"

        tex_str += "\n".join(self.commands)

        with (settings_species.paper_dir / "generated_vars.tex").open("w") as f:
            f.write(tex_str)


if __name__ == "__main__":
    generator = VariableGeneration()

    generator.add_dataset_statistics()
    generator.baseline_performance_scores()

    generator.export_tex()
