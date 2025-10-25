# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
from rich.progress import track
from scipy import stats

from htc_projects.sepsis_icu.settings_sepsis_icu import settings_sepsis_icu
from htc_projects.sepsis_icu.tables import first_inclusion
from htc_projects.sepsis_icu.visualization_helpers import generate_df, generate_run_data


class VariableGeneration:
    def __init__(self):
        self.targets = ["sepsis", "survival"]
        self.full_targets = [*self.targets, "shock"]
        self.sites = ["palm", "finger"]
        self.table_name = "test_table_new"
        self.model_names = {
            "HSI + clinical data": "HSIPlusClinicalData",
            "clinical data": "ClinicalData",
        }

        self.vars = {}
        self.commands = []

    def add_statistical_testing_table(self) -> None:
        rows = []

        for site in ["palm", "finger"]:
            for target in track(self.targets, description="Statistics table..."):
                df = first_inclusion(target, site)
                target_name = settings_sepsis_icu.task_mapping[target]
                df[target_name] = [settings_sepsis_icu.status_mapping[n] for n in df[target_name]]

                # collect parameter values in dict
                vals = {}

                for param in settings_sepsis_icu.functional_parameter_mapping.keys():
                    vals[param] = {}

                for i, status in enumerate(df[target_name].unique()):
                    df_status = df[df[target_name] == status]
                    for param in settings_sepsis_icu.functional_parameter_mapping.keys():
                        vals[param][i] = df_status[f"median_{param}"].values

                # compute statistics
                for param in settings_sepsis_icu.functional_parameter_mapping.keys():
                    main_idx = 1 if target == "sepsis" else 0
                    other_idx = 0 if target == "sepsis" else 1
                    ttest = stats.ttest_ind(vals[param][other_idx], vals[param][main_idx], equal_var=False)
                    ci = ttest.confidence_interval(confidence_level=0.95)

                    p_value = "\\num{" + f"{ttest.pvalue:.1E}".lower() + "}"
                    self.vars[
                        f"varPValue{site.capitalize()}{target.capitalize()}{settings_sepsis_icu.functional_parameter_mapping[param].split(' ')[0].capitalize()}"
                    ] = p_value
                    rows.append([
                        site,
                        f"{target} status",
                        settings_sepsis_icu.functional_parameter_mapping[param],
                        p_value,
                        round(ttest.df),
                        f"{ttest.statistic:.2f}",
                        f"$[{ci.low:.2f}; {ci.high:.2f}]$",
                    ])

        df = pd.DataFrame(
            rows,
            columns=[
                "site",
                "target",
                "functional parameter",
                "$p$-value",
                "\\acs*{dof}",
                "$t$-statistic",
                "\\SI{95}{\\percent} \\acs*{ci}",
            ],
        )

        self.vars["varStatisticsTable"] = df.to_latex(index=False, column_format="lllrrrr")

    def add_descriptive_statistics(self) -> None:
        n_subjects_all = first_inclusion()["subject_name"].nunique()
        self.vars["varTotalSubjects"] = n_subjects_all

        # proportion of all patients with shock
        df = first_inclusion("shock", "palm")
        n_shock = df[df["shock"]].subject_name.nunique()
        self.vars["varPatientsShock"] = (
            "\\SI{"
            + f"{n_shock / df.subject_name.nunique() * 100:0.0f}"
            + "}{\\percent} ($"
            + str(n_shock)
            + "/"
            + str(df.subject_name.nunique())
            + "$)"
        )

        for target in track(self.targets, description="Descriptive statistics..."):
            df = first_inclusion(target, "palm")

            if target == "sepsis":
                df_sepsis = df[df["sepsis_status"] == "sepsis"]
                df_sepsis["focus"] = ["unknown_focus" if f == "no_focus" else f for f in df_sepsis["focus"]]
                df_sepsis["focus"] = ["unknown_focus" if f == "unkown_focus" else f for f in df_sepsis["focus"]]
                focus_values = df_sepsis.focus.value_counts() / len(df_sepsis)
                for name, value in focus_values.items():
                    name = "".join([x.capitalize() for x in name.split("_")])
                    self.vars[f"varFocusRelative{name}"] = f"\\SI{{{value * 100:.0f}}}{{\\percent}}"

                # proportion of sepsis patients with shock
                n_septic_shock = df_sepsis[df_sepsis["septic_shock"]].subject_name.nunique()
                self.vars["varPatientsSepsisSepticShock"] = (
                    "\\SI{"
                    + f"{n_septic_shock / n_shock * 100:0.0f}"
                    + "}{\\percent} ($"
                    + str(n_septic_shock)
                    + "/"
                    + str(n_shock)
                    + "$)"
                )

                # mortality rate of sepsis patients with shock
                n_septic_shock_dead = len(
                    df_sepsis[df_sepsis["septic_shock"] & ~df_sepsis["survival_30_days_post_inclusion"]]
                )
                self.vars["varMortalityRateSepsisSepticShock"] = (
                    "\\SI{"
                    + f"{n_septic_shock_dead / n_septic_shock * 100:0.0f}"
                    + "}{\\percent} ($"
                    + str(n_septic_shock_dead)
                    + "/"
                    + str(n_septic_shock)
                    + "$)"
                )

            n_subjects_total = df["subject_name"].nunique()
            self.vars[f"varTotal{target.capitalize()}Task"] = n_subjects_total
            self.vars[f"varTotalExclusion{target.capitalize()}Task"] = n_subjects_all - n_subjects_total
            self.vars[f"varRelativeExclusion{target.capitalize()}Task"] = (
                "\\SI{" + f"{(n_subjects_all - n_subjects_total) / n_subjects_all * 100:0.0f}" + "}{\\percent}"
            )
            self.vars[f"varRelativeInclusion{target.capitalize()}Task"] = (
                "\\SI{" + f"{n_subjects_total / n_subjects_all * 100:0.0f}" + "}{\\percent}"
            )

            for target_value in df[settings_sepsis_icu.task_mapping[target]].unique():
                df_target = df[df[settings_sepsis_icu.task_mapping[target]] == target_value]
                target_name = settings_sepsis_icu.status_mapping[target_value]

                n_subjects = df_target["subject_name"].nunique()

                latex_name = target_name.capitalize().replace(" ", "")
                self.vars[f"varTotal{latex_name}"] = n_subjects
                self.vars[f"varRelative{latex_name}"] = (
                    "\\SI{" + f"{n_subjects / n_subjects_total * 100:0.0f}" + "}{\\percent}"
                )

                if "sepsis" in target_name:
                    n_subjects_dead = len(df_target[~df_target["survival_30_days_post_inclusion"]])
                    self.vars[f"varMortalityRate{latex_name}"] = (
                        "\\SI{"
                        + f"{n_subjects_dead / n_subjects * 100:0.0f}"
                        + "}{\\percent} ($"
                        + str(n_subjects_dead)
                        + "/"
                        + str(n_subjects)
                        + "$)"
                    )

    def add_descriptive_table(self) -> None:
        def _meta_set_table(meta_set_name: str, include_group_headings: bool, include_subjects_row: bool) -> str:
            meta_names = []
            meta_set_indices = {}
            for set_name in meta_set_name.split("+"):
                # We need the starting positions later to insert group headings
                meta_set_indices[len(meta_names)] = settings_sepsis_icu.metadata_groups_renaming.get(set_name, set_name)

                for meta_name in settings_sepsis_icu.metadata_selection_comprehensive[set_name]:
                    if meta_name in ["milrinone_dose"]:
                        continue

                    meta_names.append(meta_name)

            attribute_names = ["number of subjects"] if include_subjects_row else []
            for meta_name in meta_names:
                name = settings_sepsis_icu.metadata_paper_renaming.get(meta_name, meta_name).replace("_", " ")
                if meta_name in settings_sepsis_icu.metadata_units:
                    name += " [\\si{" + settings_sepsis_icu.metadata_units[meta_name] + "}]"
                elif settings_sepsis_icu.metadata_scales[meta_name] == "boolean":
                    name += " [\\si{\\percent}]"
                attribute_names.append(name)

            rows = {"attribute": attribute_names}
            rows |= {s: [] for s in settings_sepsis_icu.status_mapping.values()}

            for target in track(self.targets, description="Descriptive table..."):
                df = first_inclusion(target, "palm")

                for target_value in df[settings_sepsis_icu.task_mapping[target]].unique():
                    df_target = df[df[settings_sepsis_icu.task_mapping[target]] == target_value]
                    target_name = settings_sepsis_icu.status_mapping[target_value]

                    if include_subjects_row:
                        n_subjects = df_target["subject_name"].nunique()
                        rows[target_name].append(n_subjects)

                    for meta_name in meta_names:
                        meta_values = df_target[meta_name]

                        if settings_sepsis_icu.metadata_scales[meta_name] == "boolean":
                            rows[target_name].append(f"\\num{{{meta_values.sum() / len(meta_values) * 100:0.0f}}}")
                        elif settings_sepsis_icu.metadata_scales[meta_name] == "nominal":
                            counts = [f"{y} {x}" for x, y in meta_values.value_counts().items()]
                            rows[target_name].append("\\newline".join(counts))
                        else:
                            # \num automatically formats the scientific notation
                            value = "\\num{" + f"{meta_values.mean():.1E}".lower() + "}"
                            variation = "\\num{" + f"{meta_values.std():.1E}".lower() + "}"
                            rows[target_name].append(f"{value} ({variation})")

            df = pd.DataFrame(rows)
            latex = df.to_latex(index=False, column_format=">{\\raggedright}p{2.5cm}XXXX")

            # tabularx allows to insert newlines in the cells
            latex = latex.replace("\\begin{tabular}", "\\begin{tabularx}{\\textwidth}")
            latex = latex.replace("\\end{tabular}", "\\end{tabularx}")

            if include_group_headings:
                latex = latex.splitlines()

                lines_new = []
                top_rows = 5
                for i, line in enumerate(latex):
                    if i - top_rows in meta_set_indices:
                        lines_new.append("\\multicolumn{5}{c}{\\textbf{" + meta_set_indices[i - top_rows] + "}} \\\\")
                    lines_new.append(line)

                latex = "\n".join(lines_new)

            return latex

        meta_set_name = "demographic+vital+BGA+diagnosis+ventilation+catecholamines"
        self.vars["varDescriptiveTableOneHrs"] = _meta_set_table(
            meta_set_name, include_group_headings=True, include_subjects_row=True
        )

        meta_set_name = "lab"
        self.vars["varDescriptiveTableTenHrs"] = _meta_set_table(
            meta_set_name, include_group_headings=False, include_subjects_row=False
        )

    def add_performance_scores(self) -> None:
        # Modality and measurement site
        max_improvement = -1
        for target in track(self.full_targets, description="Baseline performance scores..."):
            for site in self.sites:
                target_runs = [
                    f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image_nested-*-4_seed-*-2",
                    f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image_rgb_nested-*-4_seed-*-2",
                ]
                run_data = generate_run_data(target, target_runs, self.table_name)
                df_agg = generate_df(run_data, target)[1]

                assert df_agg["model"].nunique() == len(df_agg)
                for _, row in df_agg.iterrows():
                    self.vars[f"var{target.capitalize()}Performance{site.capitalize()}{row['model']}"] = (
                        f"{row['median_AUROC']:.2f} (\\SI{{95}}{{\\percent}} #1{{ci}}"
                        f" [{row['percentile_025_AUROC']:.2f}; {row['percentile_975_AUROC']:.2f}])"
                    )

                hsi_score = df_agg[df_agg["model"] == "HSI"].iloc[0]["median_AUROC"]
                rgb_score = df_agg[df_agg["model"] == "RGB"].iloc[0]["median_AUROC"]
                improvement = (hsi_score - rgb_score) / rgb_score
                if improvement > max_improvement:
                    max_improvement = improvement  # also includes shock

        self.vars["varMaxRGBHSIImprovement"] = "\\SI{" + f"{max_improvement * 100:.0f}" + "}{\\percent}"

        # Features
        for target in track(self.targets, description="Feature performance scores..."):
            for time_name, timedelta in [("One", 1), ("Ten", 10)]:
                df_agg = pd.read_pickle(
                    settings_sepsis_icu.results_dir / f"{target}_metadata_adding_rf_importances_{timedelta}hrs_agg.pkl"
                )

                if target == "sepsis":
                    self.vars[f"varTotalFeatures{time_name}Hrs"] = df_agg["n_features"].max()

                for features_name, n_features in [
                    ("PlusOne", 1),
                    ("PlusTwo", 2),
                    ("PlusThree", 3),
                    ("PlusAll", df_agg["n_features"].max()),
                ]:
                    df_agg_n = df_agg[df_agg["n_features"] == n_features]
                    assert df_agg_n["model"].nunique() == len(df_agg_n)
                    for _, row in df_agg_n.iterrows():
                        self.vars[
                            f"var{target.capitalize()}{self.model_names[row['model']]}Performance{time_name}Hrs{features_name}"
                        ] = (
                            f"{row['median_AUROC']:.2f} (\\SI{{95}}{{\\percent}} #1{{ci}}"
                            f" [{row['percentile_025_AUROC']:.2f}; {row['percentile_975_AUROC']:.2f}])"
                        )

        self.vars["varTotalFeaturesLab"] = len(settings_sepsis_icu.metadata_selection_comprehensive["lab"])
        assert (
            self.vars["varTotalFeaturesTenHrs"] - self.vars["varTotalFeaturesOneHrs"]
            == self.vars["varTotalFeaturesLab"]
        )

        # Septic Shock Diagnosis
        target = "shock"
        subgroup = ["sepsis"]
        for site in self.sites:
            target_run = [
                f"image/{settings_sepsis_icu.model_timestamp}_{target}-inclusion_{site}_image_nested-*-4_seed-*-2",
            ]
            run_data = generate_run_data(target, target_run, "test_table_new", subgroup_selection=subgroup)
            _, df_agg, df = generate_df(run_data, target, subgroup=subgroup[0])
            for _, row in df_agg.iterrows():
                self.vars[f"varSepticShockPerformance{site.capitalize()}{row['model']}"] = (
                    f"{row['median_AUROC']:.2f} (\\SI{{95}}{{\\percent}} #1{{ci}}"
                    f" [{row['percentile_025_AUROC']:.2f}; {row['percentile_975_AUROC']:.2f}])"
                )

    def export_tex(self) -> None:
        tex_str = ""
        for key, value in self.vars.items():
            if "Performance" in key:
                # Allow to manually set the acro options for the performance scores (e.g., for abstracts)
                tex_str += "\\newcommand{" + f"\\{key}" + "}[1][\\ac]{" + f"{value}\\xspace" + "}\n"
            else:
                tex_str += "\\newcommand{" + f"\\{key}" + "}{" + f"{value}\\xspace" + "}\n"

        tex_str += "\n".join(self.commands)

        with (settings_sepsis_icu.paper_dir / "generated_vars.tex").open("w") as f:
            f.write(tex_str)


if __name__ == "__main__":
    generator = VariableGeneration()

    generator.add_statistical_testing_table()
    generator.add_descriptive_statistics()
    generator.add_descriptive_table()
    generator.add_performance_scores()

    generator.export_tex()
