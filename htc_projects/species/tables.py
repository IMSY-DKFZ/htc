# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from htc import median_table
from htc.models.data.DataSpecification import DataSpecification
from htc.utils.helper_functions import add_times_table, sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc_projects.atlas.tables import standardized_recordings
from htc_projects.rat.tables import standardized_recordings_rat
from htc_projects.species.settings_species import settings_species


def baseline_table(label_mapping: LabelMapping = None) -> pd.DataFrame:
    """
    Load the physiological median spectra data from all species (species_name column).

    The same images as used for training are loaded here.

    Args:
        label_mapping: Optional label mapping passed on to the median_table function.

    Returns: Median spectra with species information.
    """
    spec_pig = DataSpecification("pig_semantic-only_5folds_nested-0-2_mapping-12_seed-0.json")
    spec_rat = DataSpecification("rat_semantic-only_5folds_nested-0-2_mapping-12_seed-0.json")
    spec_human = DataSpecification("human_semantic-only_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0.json")

    dfs = []
    for spec, species_name in [(spec_pig, "pig"), (spec_rat, "rat"), (spec_human, "human")]:
        spec.activate_test_set()
        if label_mapping is not None:
            # Remove images which do not contain any of the requested labels (e.g., background may not be needed)
            paths = [
                p
                for p in spec.paths()
                if len(set(p.annotated_labels()).intersection(label_mapping.label_names(all_names=True))) > 0
            ]
        else:
            paths = spec.paths()

        df = median_table(paths=paths, label_mapping=label_mapping, keep_mapped_columns=False)
        df["species_name"] = species_name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return sort_labels(df)


def ischemic_table(label_mapping: LabelMapping = None) -> pd.DataFrame:
    """
    Load ischemic median spectra data from all pig and rat species (species_name column). This includes a diff_spectrum column that contains the difference between the normalized median spectrum for each row and the physiological spectrum.

    The physiological data is loaded as well for each species so that this table is the most comprehensive overview of used images.

    Args:
        label_mapping: Optional label mapping passed on to the median_table function.

    Returns: The ischemic median spectra data.
    """

    def _add_diff_spectra(
        df: pd.DataFrame, base_spectra: pd.DataFrame, base_spectra_global: pd.DataFrame = None
    ) -> None:
        diff_spectra = []
        for i, row in df.iterrows():
            if (row["subject_name"], row["label_name"]) in base_spectra.index:
                diff_spectra.append(
                    row["median_normalized_spectrum"]
                    - base_spectra.loc[(row["subject_name"], row["label_name"])]["median_normalized_spectrum"]
                )
            else:
                assert base_spectra_global is not None, (
                    "base_spectra_global is missing but there is no base spectrum for subject"
                    f" {row['subject_name']} and label {row['label_name']}"
                )
                diff_spectra.append(
                    row["median_normalized_spectrum"]
                    - base_spectra_global.loc[row["label_name"]]["median_normalized_spectrum"]
                )
        df["diff_spectrum"] = diff_spectra

    dfk = median_table(
        dataset_name="2023_04_22_Tivita_multiorgan_kidney",
        annotation_name="semantic#primary",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    dfk = dfk.query("label_name == 'kidney'").copy()
    dfk["species_name"] = "pig"

    dfk["phase_type"] = dfk["phase_type"].astype("category")
    dfk["phase_type"] = dfk["phase_type"].cat.set_categories(["physiological", "ischemia", "avascular", "stasis"])
    dfk = sort_labels(dfk, sorting_cols=["label_name", "phase_type"])
    add_times_table(dfk, groups=["subject_name"])
    dfk.reset_index(drop=True, inplace=True)

    base_spectra = (
        dfk.query("phase_type == 'physiological'")
        .groupby(["subject_name", "label_name"])
        .agg(
            median_normalized_spectrum=pd.NamedAgg(
                column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
            ),
        )
    )
    _add_diff_spectra(dfk, base_spectra)

    dfa = median_table(
        dataset_name="2023_12_05_Tivita_multiorgan_aortic_clamping",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    add_times_table(dfa, groups=["subject_name"])
    dfa["species_name"] = "pig"

    base_spectra = (
        dfa.query("phase_type == 'physiological'")
        .groupby(["subject_name", "label_name"])
        .agg(
            median_normalized_spectrum=pd.NamedAgg(
                column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
            ),
        )
    )

    # Some labels like unclear_organic are not available for all subjects
    base_spectra_global = base_spectra.groupby(["label_name"]).agg(
        median_normalized_spectrum=pd.NamedAgg(
            column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
        ),
    )
    _add_diff_spectra(dfa, base_spectra, base_spectra_global=base_spectra_global)

    dfr = median_table(
        dataset_name="2023_12_07_Tivita_multiorgan_rat#perfusion_experiments",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    add_times_table(dfr, groups=["subject_name"])
    dfr["species_name"] = "rat"

    base_spectra = (
        dfr.query("phase_type == 'physiological'")
        .groupby(["subject_name", "label_name"])
        .agg(
            median_normalized_spectrum=pd.NamedAgg(
                column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
            ),
        )
    )
    _add_diff_spectra(dfr, base_spectra)

    df_baseline = baseline_table(label_mapping=label_mapping)

    dfh_phys = df_baseline[df_baseline["species_name"] == "human"].copy()
    dfh_phys["perfusion_state"] = "physiological"
    dfh_phys["baseline_dataset"] = True
    base_spectra = dfh_phys.groupby(["subject_name", "label_name"]).agg(
        median_normalized_spectrum=pd.NamedAgg(
            column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
        ),
    )
    base_spectra_global = base_spectra.groupby(["label_name"]).agg(
        median_normalized_spectrum=pd.NamedAgg(
            column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
        ),
    )
    dfh_mal = median_table(
        dataset_name="2021_07_26_Tivita_multiorgan_human",
        annotation_name="polygon#malperfused",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )

    # There is unavoidable overlap between the physiological and malperfused images for humans but this is only for the non-target labels
    n_target_labels = dfh_mal.label_name.isin(settings_species.malperfused_labels).sum()
    dfh_mal = dfh_mal[~dfh_mal.image_name.isin(dfh_phys.image_name)].copy()
    assert (
        dfh_mal.label_name.isin(settings_species.malperfused_labels).sum() == n_target_labels
    ), "The number of target labels must not change"

    dfh_mal["species_name"] = "human"
    dfh_mal["perfusion_state"] = "malperfused"
    _add_diff_spectra(dfh_mal, base_spectra, base_spectra_global)

    # Every semantically annotated image which has not been used previously
    dfh_unclear = median_table(
        dataset_name="2021_07_26_Tivita_multiorgan_human",
        annotation_name="semantic#primary",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    dfh_unclear = dfh_unclear[
        ~(dfh_unclear.image_name.isin(dfh_phys.image_name) | dfh_unclear.image_name.isin(dfh_mal.image_name))
    ].copy()
    assert set(settings_species.malperfused_kidney_subjects).issubset(
        set(dfh_unclear.subject_name)
    ), "All manually excluded kidney subjects must be unclear"
    dfh_unclear["species_name"] = "human"
    dfh_unclear["perfusion_state"] = "unclear"

    # Also add the other physiological images we have for pig and rat
    df_baseline_rest = df_baseline[df_baseline["species_name"] != "human"].copy()
    df_baseline_rest["phase_type"] = "physiological"
    df_baseline_rest["baseline_dataset"] = True

    df = pd.concat([dfk, dfa, dfr, dfh_phys, dfh_mal, dfh_unclear, df_baseline_rest], ignore_index=True)
    with pd.option_context("future.no_silent_downcasting", True):
        df["baseline_dataset"] = df["baseline_dataset"].fillna(False).astype(bool)

    # There may be multiple label names which get mapped to background and hence would have the same image_name and label_name in this table
    assert (
        df[df.label_name != "background"].set_index(["image_name", "label_name"]).index.is_unique
    ), "There must be no overlap between the different tables"

    return df


def ischemic_clear_table(label_mapping: LabelMapping = None) -> pd.DataFrame:
    """
    Subset of the ischemic table which only includes clear perfusion states.

    Args:
        label_mapping: Optional label mapping passed on to the median_table function.

    Returns: The ischemic median spectra data with only clear perfusion states.
    """
    df = ischemic_table(label_mapping=label_mapping)

    # Select for each species only the clear perfusion states
    df_rat = df.query("species_name == 'rat'").copy()
    df_rat = df_rat[
        (pd.isna(df_rat.reperfused) | (df_rat.reperfused == False))  # noqa: E712
        & (pd.isna(df_rat.phase_time) | df_rat.phase_time != 0)
    ]
    assert not pd.isna(df_rat["phase_type"]).any(), "All rat images must have a phase type"
    df_rat["perfusion_state"] = df_rat["phase_type"].map({
        "physiological": "physiological",
        "ischemia": "malperfused",
        "stasis": "malperfused",
        "avascular": "malperfused",
    })

    df_pig = df.query("species_name == 'pig'").copy()
    assert not pd.isna(df_pig["phase_type"]).any(), "All pig images must have a phase type"
    df_pig["perfusion_state"] = df_pig["phase_type"].map({
        "physiological": "physiological",
        "ischemia": "malperfused",
        "stasis": "malperfused",
        "avascular": "malperfused",
    })

    df_human = df.query("species_name == 'human'").copy()
    df_human = df_human[df_human["perfusion_state"] != "unclear"]

    return pd.concat([df_human, df_rat, df_pig], ignore_index=True)


def paper_table() -> pd.DataFrame:
    """
    Complete table which includes all data which is used in the paper.

    Returns: Median spectra table from all images.
    """
    # Figure 7 shows all baseline data, Figure 8 all kidney data
    df = ischemic_table(settings_species.label_mapping)
    df = df[df.baseline_dataset | df.label_name.isin(settings_species.malperfused_labels)].copy()
    df["standardized_recordings"] = False

    # LMM data
    label_mapping = settings_species.label_mapping_organs
    df_pig = standardized_recordings()
    df_pig["species_name"] = "pig"
    df_pig["baseline_dataset"] = False
    df_pig["standardized_recordings"] = True
    df_pig = df_pig.query("label_name in @label_mapping.label_names()").copy()
    df_rat = standardized_recordings_rat(label_mapping=label_mapping, Camera_CamID="0202-00118")
    df_rat["species_name"] = "rat"
    df_rat["baseline_dataset"] = False
    df_rat["standardized_recordings"] = True

    names = set(df.image_name)
    df_pig = df_pig[~df_pig.image_name.isin(names)]
    df_rat = df_rat[~df_rat.image_name.isin(names)]

    return pd.concat([df, df_pig, df_rat], ignore_index=True)
