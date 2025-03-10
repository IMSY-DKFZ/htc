# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from functools import partial

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.paths import filter_labels
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import baseline_table, ischemic_clear_table, ischemic_table


def test_ischemic_table() -> None:
    mapping = settings_species.label_mapping_organs
    mapping_filter = partial(filter_labels, mapping=mapping)
    df_baseline = baseline_table(label_mapping=mapping)
    df_ischemic = ischemic_table(label_mapping=mapping)
    assert set(df_baseline["image_name"]).issubset(set(df_ischemic["image_name"]))

    # Check that all relevant image are used for all species
    paths = list(
        DataPath.iterate(settings.data_dirs.semantic, annotation_name="semantic#primary", filters=[mapping_filter])
    )
    assert set(df_baseline.query("species_name == 'pig'")["image_name"]) == {p.image_name() for p in paths}

    paths += list(
        DataPath.iterate(settings.data_dirs.kidney, annotation_name="semantic#primary", filters=[mapping_filter])
    )
    paths += list(
        DataPath.iterate(
            settings.data_dirs.kidney / "overlap", annotation_name="semantic#primary", filters=[mapping_filter]
        )
    )
    paths += list(
        DataPath.iterate(
            settings.data_dirs.aortic_clamping, annotation_name="semantic#primary", filters=[mapping_filter]
        )
    )
    assert set(df_ischemic.query("species_name == 'pig'")["image_name"]) == {p.image_name() for p in paths}

    paths = list(DataPath.iterate(settings.data_dirs.rat, annotation_name="semantic#primary", filters=[mapping_filter]))
    assert set(df_baseline.query("species_name == 'rat'")["image_name"]) == {p.image_name() for p in paths}

    paths += list(DataPath.iterate(settings.data_dirs.rat / "perfusion_experiments"))
    assert set(df_ischemic.query("species_name == 'rat'")["image_name"]) == {p.image_name() for p in paths}

    excluded_images = set(settings_species.excluded_images)
    filter_excluded = lambda p: p.image_name() not in excluded_images
    paths = list(
        DataPath.iterate(
            settings.data_dirs.human, annotation_name="semantic#primary", filters=[mapping_filter, filter_excluded]
        )
    )
    paths += list(
        DataPath.iterate(
            settings.data_dirs.human, annotation_name="polygon#malperfused", filters=[mapping_filter, filter_excluded]
        )
    )
    assert set(df_ischemic.query("species_name == 'human'")["image_name"]) == {p.image_name() for p in paths}


def test_ischemic_clear_table() -> None:
    df_baseline = baseline_table()
    df_ischemic = ischemic_table()
    df_ischemic_clear = ischemic_clear_table()

    assert set(df_baseline["image_name"]).issubset(set(df_ischemic_clear["image_name"]))
    assert set(df_ischemic_clear["image_name"]).issubset(set(df_ischemic["image_name"]))

    df_ischemic_baseline = df_ischemic[df_ischemic["baseline_dataset"]][["image_name", "annotation_name", "label_name"]]
    df_ischemic_baseline = df_ischemic_baseline.sort_values(
        by=["image_name", "annotation_name", "label_name"], ignore_index=True
    )
    df_baseline_main_cols = df_baseline[["image_name", "annotation_name", "label_name"]].sort_values(
        by=["image_name", "annotation_name", "label_name"], ignore_index=True
    )
    assert len(df_ischemic[df_ischemic["baseline_dataset"]]) == len(df_baseline)
    assert (df_ischemic_baseline.values == df_baseline_main_cols.values).all()

    species = df_ischemic_clear["species_name"].unique()
    assert set(species) == {"pig", "rat", "human"}
    for species_name in species:
        n_clear = df_ischemic_clear[df_ischemic_clear["species_name"] == species_name]["image_name"].nunique()
        n_all = df_ischemic[df_ischemic["species_name"] == species_name]["image_name"].nunique()

        if species_name == "pig":
            assert n_all >= n_clear > 0
        else:
            assert n_all > n_clear > 0
