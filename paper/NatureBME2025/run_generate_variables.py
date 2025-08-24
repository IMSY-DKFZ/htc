# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc_projects.species.settings_species import settings_species
from htc_projects.species.species_evaluation import TrainingDistanceComputation, baseline_performance, icg_performance
from htc_projects.species.tables import paper_table


class VariableGeneration:
    def __init__(self):
        self.df = paper_table()

        self.vars = {}
        self.commands = []

    def add_dataset_statistics(self) -> None:
        self.vars["varTotalImages"] = "\\num{" + f"{self.df['image_name'].nunique()}" + "}"
        self.vars["varTotalSubjects"] = "\\num{" + f"{self.df['subject_name'].nunique()}" + "}"

        df_annotation_counts = self.df[["image_name", "annotation_name"]].drop_duplicates(ignore_index=True)
        df_annotation_counts = df_annotation_counts.annotation_name.value_counts(dropna=False)
        assert len(df_annotation_counts) == 4, "There should be three different types of annotations"
        self.vars["varTotalImagesSemantic"] = (
            "\\num{"
            + f"{df_annotation_counts['semantic#primary'] + df_annotation_counts['semantic#reannotation']}"
            + "}"
        )
        self.vars["varTotalImagesPolygon"] = (
            "\\num{"
            + f"{df_annotation_counts['polygon#annotator1'] + df_annotation_counts['polygon#malperfused']}"
            + "}"
        )

        df_phys = self.df[~self.df["perfusion_state"].isin(["malperfused", "icg"])]
        df_mal = self.df[self.df["perfusion_state"] == "malperfused"]
        df_icg = self.df[self.df["perfusion_state"] == "icg"]

        df_icg_subjects = df_icg.groupby("species_name")["subject_name"].nunique()
        self.vars["varTotalSubjectsICGPig"] = "\\num{" + f"{df_icg_subjects.pig.item()}" + "}"
        self.vars["varTotalSubjectsICGRat"] = "\\num{" + f"{df_icg_subjects.rat.item()}" + "}"

        for species in settings_species.species_colors.keys():
            df_species = self.df[self.df.species_name == species]
            self.vars[f"varTotalImages{species.capitalize()}"] = (
                "\\num{" + f"{df_species['image_name'].nunique()}" + "}"
            )
            self.vars[f"varTotalSubjects{species.capitalize()}"] = (
                "\\num{" + f"{df_species['subject_name'].nunique()}" + "}"
            )

            self.vars[f"varTotalImagesPhys{species.capitalize()}"] = (
                "\\num{" + f"{df_phys[df_phys.species_name == species]['image_name'].nunique()}" + "}"
            )
            self.vars[f"varTotalImagesMal{species.capitalize()}"] = (
                "\\num{" + f"{df_mal[df_mal.species_name == species]['image_name'].nunique()}" + "}"
            )
            self.vars[f"varTotalImagesICG{species.capitalize()}"] = (
                "\\num{" + f"{df_icg[df_icg.species_name == species]['image_name'].nunique()}" + "}"
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

    def distance_improvements(self) -> None:
        dc = TrainingDistanceComputation(param_name="median_sto2")

        rows = []
        for target_species, source_species in [("pig", "rat"), ("rat", "pig"), ("human", "rat"), ("human", "pig")]:
            label_diffs = []
            for label in settings_species.malperfused_labels_extended:
                _, df = dc.compute_distances(target_species, source_species, label)
                assert set(df["network"]) == {"in-species", "xeno-learning"}

                df_baseline = df[df.network == "in-species"]
                df_xeno = df[df.network == "xeno-learning"]
                assert (df_baseline.image_name == df_xeno.image_name).all(), "The same images should be compared"
                diffs = (df_baseline.euclidean_distance - df_xeno.euclidean_distance) / df_baseline.euclidean_distance
                label_diffs.append(diffs.mean())

            rows.append({
                "source_species": source_species,
                "target_species": target_species,
                "improvements": label_diffs,
            })

        df = pd.DataFrame(rows)
        for _, row in df.iterrows():
            self.vars[
                f"varDistanceImprovement{row['source_species'].capitalize()}To{row['target_species'].capitalize()}"
            ] = (
                "\\SI{"
                + f"{np.mean(row['improvements']) * 100:0.1f}"
                + "}{\\percent} ("
                + "\\ac{sd} \\SI{"
                + f" {100 * np.std(row['improvements']):0.0f}"
                + "}{\\percent})"
            )

    def icg_performance_scores(self) -> None:
        df = icg_performance()

        rows = {
            "improvement": [],
            "label_name": [],
            "species_name": [],
        }
        for species in df["species"].unique():
            df_baseline = df[(df.species == species) & (df.network.str.startswith("baseline"))]
            df_xeno = df[(df.species == species) & (df.network.str.startswith("projected"))]
            assert (df_baseline.subject_name.values == df_xeno.subject_name.values).all(), (
                "The same subjects should be compared"
            )

            diff = df_xeno.dice_metric.values - df_baseline.dice_metric.values
            rows["improvement"].extend(diff)
            rows["label_name"].extend(df_baseline.label_name)
            rows["species_name"].extend([species] * len(diff))

        df_scores = pd.DataFrame(rows)
        df_scores = df_scores.groupby(["species_name", "label_name"], as_index=False)["improvement"].agg([
            "mean",
            "std",
        ])

        for species in df["species"].unique():
            df_species = df_scores[df_scores.species_name == species]

            self.vars[f"varICGImprovementMean{species.capitalize()}"] = (
                f"{df_species['mean'].mean():.2f} (" + "\\ac{sd}" + f" {df_species['std'].mean():.2f})"
            )

            df_species = df_species.sort_values(by="mean", ascending=False)
            self.vars[f"varICGImprovementMax{species.capitalize()}"] = f"{df_species['mean'].max():.2f}"
            self.vars[f"varICGImprovementMaxLabel{species.capitalize()}"] = df_species.iloc[0]["label_name"]

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
    generator.distance_improvements()
    generator.icg_performance_scores()

    generator.export_tex()
