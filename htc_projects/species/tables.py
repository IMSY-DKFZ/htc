# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
from functools import partial

import numpy as np
import pandas as pd

from htc import median_table
from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.DomainMapper import DomainMapper
from htc.utils.helper_functions import add_times_table, sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.paths import filter_labels, filter_min_labels
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
    spec_pig = DataSpecification(settings_species.spec_names["pig"].replace("nested-*", "nested-0"))
    spec_rat = DataSpecification(settings_species.spec_names["rat"].replace("nested-*", "nested-0"))
    spec_human = DataSpecification(settings_species.spec_names["human"].replace("nested-*", "nested-0"))

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
    Load ischemic median spectra data from all species (species_name column). Physiological baseline images are included as well (baseline_dataset column) so that this table is the most comprehensive overview of available images. Only the ICG images are not included (`icg_table()` can be used for this).

    The `perfusion_state` column is available for all images and indicates whether an image is considered physiological or malperfused (or unclear). For animal data, a `phase_type` column is available which additionally indicates the type of ischemia.

    The diff_spectrum column contains the difference between the normalized median spectrum for each row and the physiological spectrum.

    Args:
        label_mapping: Optional label mapping passed on to the median_table function.

    Returns: The ischemic median spectra data.
    """

    def _add_diff_spectra(df: pd.DataFrame) -> None:
        if "phase_type" in df.columns:
            df_base = df[df["phase_type"] == "physiological"]
        elif "perfusion_state" in df.columns:
            df_base = df[df["perfusion_state"] == "physiological"]
        else:
            raise ValueError("Either phase_type or perfusion_state must be present in the dataframe")

        assert len(df_base) > 0, "There must be at least one physiological image"

        base_spectra = df_base.groupby(["subject_name", "label_name"]).agg(
            median_normalized_spectrum=pd.NamedAgg(
                column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
            ),
        )

        # Some labels like unclear_organic are not available for all subjects
        base_spectra_global = base_spectra.groupby(["label_name"]).agg(
            median_normalized_spectrum=pd.NamedAgg(
                column="median_normalized_spectrum", aggfunc=lambda c: np.mean(np.stack(c), axis=0)
            ),
        )

        diff_spectra = []
        for _, row in df.iterrows():
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

    phase_type_mapping = {
        "physiological": "physiological",
        "ischemia": "malperfused",
        "stasis": "malperfused",
        "avascular": "malperfused",
    }

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
    dfk["perfusion_state"] = dfk["phase_type"].map(phase_type_mapping)
    dfk = sort_labels(dfk, sorting_cols=["label_name", "phase_type"])
    add_times_table(dfk, groups=["subject_name"])
    dfk.reset_index(drop=True, inplace=True)

    _add_diff_spectra(dfk)

    dfa = median_table(
        dataset_name="2023_12_05_Tivita_multiorgan_aortic_clamping",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    add_times_table(dfa, groups=["subject_name"])
    dfa["species_name"] = "pig"
    dfa["perfusion_state"] = dfa["phase_type"].map(phase_type_mapping)

    _add_diff_spectra(dfa)

    dfr = median_table(
        dataset_name="2023_12_07_Tivita_multiorgan_rat#perfusion_experiments",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    add_times_table(dfr, groups=["subject_name"])
    dfr["species_name"] = "rat"
    dfr["perfusion_state"] = dfr["phase_type"].map(phase_type_mapping)

    _add_diff_spectra(dfr)

    df_baseline = baseline_table(label_mapping=label_mapping)
    df_baseline["perfusion_state"] = "physiological"
    df_baseline["baseline_dataset"] = True

    dfh_phys = df_baseline[df_baseline["species_name"] == "human"].copy()
    dfh_mal = median_table(
        dataset_name="2021_07_26_Tivita_multiorgan_human",
        annotation_name="polygon#malperfused",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )

    # There is unavoidable overlap between the physiological and malperfused images for humans but this is only for the non-target labels
    n_target_labels = dfh_mal.label_name.isin(settings_species.malperfused_labels).sum()
    dfh_mal = dfh_mal[~dfh_mal.image_name.isin(dfh_phys.image_name)].copy()
    assert dfh_mal.label_name.isin(settings_species.malperfused_labels).sum() == n_target_labels, (
        "The number of target labels must not change"
    )

    dfh_mal["species_name"] = "human"
    dfh_mal["perfusion_state"] = "malperfused"

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
    assert set(settings_species.malperfused_kidney_subjects).issubset(set(dfh_unclear.subject_name)), (
        "All manually excluded kidney subjects must be unclear"
    )
    dfh_unclear["species_name"] = "human"
    dfh_unclear["perfusion_state"] = "unclear"

    dfh = pd.concat([dfh_phys, dfh_mal, dfh_unclear], ignore_index=True)
    _add_diff_spectra(dfh)

    # Also add the other physiological images we have for pig and rat
    df_baseline_rest = df_baseline[df_baseline["species_name"] != "human"].copy()
    df_baseline_rest["phase_type"] = "physiological"
    _add_diff_spectra(df_baseline_rest)

    df = pd.concat([dfk, dfa, dfr, dfh, df_baseline_rest], ignore_index=True)
    with pd.option_context("future.no_silent_downcasting", True):
        df["baseline_dataset"] = df["baseline_dataset"].fillna(False).astype(bool)

    # Exclude images which are not suitable for our analysis
    df = df[~df.image_name.isin(settings_species.excluded_images)].copy()

    # There may be multiple label names which get mapped to background and hence would have the same image_name and label_name in this table
    assert df[df.label_name != "background"].set_index(["image_name", "label_name"]).index.is_unique, (
        "There must be no overlap between the different tables"
    )
    assert not pd.isna(df["perfusion_state"]).any(), "All images must have the perfusion_state column"

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


def icg_table(label_mapping: LabelMapping = None) -> pd.DataFrame:
    """
    Load the ICG median spectra data from the pig and rat species (species_name column).

    The resulting table contains a perfusion_state columns which indicates whether the image is subject to ICG or whether it is considered a physiological image (physiological reference images from the same time series are only available for rats).

    Args:
        label_mapping: Optional label mapping passed on to the median_table function.

    Returns: The ICG median spectra data.
    """
    # Don't include data from subjects which are also part of the training
    train_subjects = set(
        baseline_table(label_mapping=label_mapping).query("species_name == 'pig'").subject_name.unique()
    )
    filter_subjects = lambda p: p.subject_name not in train_subjects

    filters = [filter_subjects]
    if label_mapping is not None:
        filters.append(partial(filter_labels, mapping=label_mapping))
    else:
        filters.append(filter_min_labels)

    paths = list(DataPath.iterate(settings.data_dirs.icg_pig, filters=filters))
    df_pig = median_table(paths=paths, label_mapping=label_mapping, keep_mapped_columns=False)
    df_pig["species_name"] = "pig"
    df_pig["perfusion_state"] = "icg"

    assert not pd.isna(df_pig["icg"]).any()

    # Expand icg column
    df_icg = pd.json_normalize(df_pig["icg"])
    df_icg.columns = [f"icg_{col}" for col in df_icg.columns]

    # The pig data is already confirmed to be clear cases with ICG
    df_icg["icg_clear"] = True
    df_pig = pd.concat([df_pig, df_icg], axis=1)

    df_rat = median_table(
        dataset_name="2023_12_07_Tivita_multiorgan_rat#ICG_experiments",
        label_mapping=label_mapping,
        keep_mapped_columns=False,
    )
    df_icg = pd.json_normalize(df_rat["icg"])
    df_icg.columns = [f"icg_{col}" for col in df_icg.columns]

    # The rat data contains time series ICG data and we consider 10 minutes after the ICG injection as clear cases
    df_icg["icg_clear"] = df_icg.icg_seconds_after_dose.apply(lambda x: np.min(x) <= 600)
    df_rat = pd.concat([df_rat, df_icg], axis=1)

    df_rat["species_name"] = "rat"

    states = []
    for _, row in df_rat.iterrows():
        if pd.isna(row["icg"]):
            states.append("physiological")
        else:
            states.append("icg")

    df_rat["perfusion_state"] = states

    df = pd.concat([df_pig, df_rat], ignore_index=True)
    df["icg_last_injection"] = df.icg_seconds_after_dose.apply(np.min)
    df["icg_series"] = df.icg_seconds_after_dose.apply(lambda x: x if type(x) == float else len(x))
    df["baseline_dataset"] = False
    df = df.convert_dtypes()

    assert not df[df.perfusion_state == "physiological"].icg_clear.any(), (
        "Physiological data must never be clear ICG data"
    )

    return df


def paper_table() -> pd.DataFrame:
    """
    Complete table which includes all data which is used in the paper.

    Returns: Median spectra table from all images.
    """
    # All baseline data from the networks + the malperfused data + the ICG data
    df_mal = ischemic_table(settings_species.label_mapping)
    df_mal = df_mal[
        df_mal.baseline_dataset | df_mal.label_name.isin(settings_species.malperfused_labels_extended)
    ].copy()
    df_mal["standardized_recordings"] = False

    df_icg = icg_table()
    df_icg = df_icg[(df_icg["perfusion_state"] == "icg") & (df_icg.label_name.isin(settings_species.icg_labels))]
    df_icg["standardized_recordings"] = False

    # LMM data (extended data figure)
    label_mapping = settings_species.label_mapping_organs
    df_pig = standardized_recordings()
    df_pig["species_name"] = "pig"
    df_pig["baseline_dataset"] = False
    df_pig["standardized_recordings"] = True
    df_pig["perfusion_state"] = "physiological"
    df_pig = df_pig.query("label_name in @label_mapping.label_names()").copy()

    df_rat = standardized_recordings_rat(label_mapping=label_mapping, Camera_CamID="0202-00118")
    df_rat["species_name"] = "rat"
    df_rat["baseline_dataset"] = False
    df_rat["standardized_recordings"] = True
    df_rat["perfusion_state"] = "physiological"

    names = set(df_mal.image_name).union(set(df_icg.image_name))
    df_pig = df_pig[~df_pig.image_name.isin(names)]
    df_rat = df_rat[~df_rat.image_name.isin(names)]
    names_standardized = set(df_pig.image_name).union(set(df_rat.image_name))

    df = pd.concat([df_mal, df_icg, df_pig, df_rat], ignore_index=True)
    names = set(df.image_name)

    # Make sure we include the images used for the projection matrices
    json_files = sorted((settings.results_dir / "projection_matrices").glob("*.json"))
    assert len(json_files) > 0, "There must be at least one projection matrix file"
    projection_paths = {"physiological": set(), "icg": set(), "malperfused": set()}
    for json_file in json_files:
        with json_file.open() as f:
            data = json.load(f)

        for values in data.values():
            for v in values:
                projection_paths["physiological"].add(DataPath.from_image_name(v["name_physiological"]))
                if "ICG" in json_file.name:
                    projection_paths["icg"].add(DataPath.from_image_name(v["name_ischemic"]))
                elif "malperfusion" in json_file.name:
                    projection_paths["malperfused"].add(DataPath.from_image_name(v["name_ischemic"]))
                else:
                    raise ValueError(f"Unknown ischemic state: {json_file.name}")

    df_projections = []
    for state, paths in projection_paths.items():
        if len(paths) == 0:
            continue

        paths = sorted(paths)
        df_projection = median_table(
            paths=paths, label_mapping=settings_species.label_mapping, keep_mapped_columns=False
        )
        assert not df_projection.image_name.isin(names_standardized).any(), (
            "The projection images should not overlap with the standardized recordings"
        )

        df_projection = df_projection[~df_projection.image_name.isin(names)]
        df_projection["species_name"] = df_projection.image_name.apply(
            DomainMapper(paths, target_domain="species_index").domain_name
        )
        df_projection["standardized_recordings"] = False
        df_projection["baseline_dataset"] = False
        df_projection["perfusion_state"] = state

        df_projections.append(df_projection)

    df = pd.concat([df, *df_projections], ignore_index=True)
    df = df.convert_dtypes()

    return df
