# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch

from htc.models.data.DataSpecification import DataSpecification
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import basic_statistics, get_nsd_thresholds, median_table, sort_labels
from htc.utils.LabelMapping import LabelMapping


@pytest.mark.skipif(not settings_seg.nsd_tolerances_path.exists(), reason="Precomputed NSD values are not available")
def test_get_nsd_thresholds() -> None:
    mapping = LabelMapping({"heart": 2, "colon": 0, "omentum": 1}, last_valid_label_index=2)
    tolerances = get_nsd_thresholds(mapping)
    # Values need to be updated when changing the values in nsd_tolerances_path
    assert tolerances == pytest.approx([5.319908, 9.685355, 12.253547])

    tolerances = get_nsd_thresholds(mapping, aggregation_method="median")
    assert tolerances == pytest.approx([0.785714, 1.6897757, 2])


def test_basic_statistics() -> None:
    df = basic_statistics("2021_02_05_Tivita_multiorgan_semantic", "pigs_semantic-only_5foldsV2.json")
    image_names = {p.image_name() for p in DataPath.iterate(settings.data_dirs.semantic)}
    assert set(df["image_name"]) == image_names

    df_mapped = basic_statistics(
        "2021_02_05_Tivita_multiorgan_semantic",
        "pigs_semantic-only_5foldsV2.json",
        label_mapping=settings_seg.label_mapping,
    )
    assert set(df_mapped["image_name"]) == image_names

    df_mapped_single = df_mapped.query("image_name == 'P041#2019_12_14_12_00_16'")
    assert (
        df_mapped_single["label_name"] == ["background", "colon", "small_bowel", "bladder", "fat_subcutaneous"]
    ).all()
    # Background is composed of blue_cloth and metal
    assert (df_mapped_single["n_pixels"] == [158624 + 162, 67779, 65634, 10594, 4251]).all()

    df_mapped2 = basic_statistics(
        image_names=df_mapped["image_name"].tolist(),
        spec="pigs_semantic-only_5foldsV2.json",
        label_mapping=settings_seg.label_mapping,
    )
    assert_frame_equal(df_mapped, df_mapped2, check_categorical=False, check_dtype=False)


def test_sort_labels() -> None:
    assert sort_labels(["colon", "stomach"]) == ["stomach", "colon"]
    assert (sort_labels(np.array(["colon", "stomach"])) == np.array(["stomach", "colon"])).all()
    assert sort_labels({"colon": 1, "stomach": 2}) == {"stomach": 2, "colon": 1}
    assert sort_labels(["colon", "stomach"], label_ordering=["colon", "stomach"]) == ["colon", "stomach"]

    df_unsorted = pd.DataFrame({
        "label_name": ["unknown_organ", "colon", "stomach", "stomach", "stomach"],
        "image_name": ["img1", "img2", "img4", "img3", "img3"],
        "annotation_name": ["annotator1", "annotator1", "annotator1", "annotator2", "annotator1"],
    })
    df_sorted = pd.DataFrame({
        "label_name": ["stomach", "stomach", "stomach", "colon", "unknown_organ"],
        "image_name": ["img3", "img3", "img4", "img2", "img1"],
        "annotation_name": ["annotator1", "annotator2", "annotator1", "annotator1", "annotator1"],
    })
    assert_frame_equal(df_sorted, sort_labels(df_unsorted))

    df_sorted2 = pd.DataFrame({
        "label_name": ["stomach", "stomach", "stomach", "colon", "unknown_organ"],
        "image_name": ["img3", "img4", "img3", "img2", "img1"],
        "annotation_name": ["annotator1", "annotator1", "annotator2", "annotator1", "annotator1"],
    })
    assert_frame_equal(
        df_sorted2, sort_labels(df_unsorted, sorting_cols=["label_name", "annotation_name", "image_name"])
    )

    unsorted_labels = ["x", "a", "g", "f", "q"]
    sorted_labels = ["a", "f", "g", "q", "x"]
    df_unsorted = pd.DataFrame({"label_name": unsorted_labels})
    assert sort_labels(unsorted_labels) == sorted_labels
    assert sort_labels(dict.fromkeys(unsorted_labels, 1)) == dict.fromkeys(sorted_labels, 1)
    assert_frame_equal(sort_labels(df_unsorted), pd.DataFrame({"label_name": sorted_labels}))


class TestMedianTable:
    def test_default(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        df_example1 = pd.DataFrame(
            [
                ["i1", "bladder", 0, 0],
                ["i2", "lung", 1, 1],
                ["i3", "bladder", 0, 2],
                ["i4", "bladder", 0, 3],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )

        df_example2_a1 = pd.DataFrame(
            [
                ["i1#overlap", "bladder", 0, 4],
                ["i5", "bladder", 0, 5],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )
        df_example2_a2 = pd.DataFrame(
            [
                ["i1#overlap", "bladder", 0, 6],
                ["i5", "bladder", 0, 7],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )

        target_dir = tmp_path / "tables"
        target_dir.mkdir(exist_ok=True, parents=True)

        # Median spectra tables are usually sorted
        df_example1 = sort_labels(df_example1)
        df_example2_a1 = sort_labels(df_example2_a1)

        df_example1.to_feather(
            target_dir / "2021_02_05_Tivita_multiorgan_semantic@median_spectra@semantic#primary.feather"
        )
        df_example2_a1.to_feather(
            target_dir / "2021_02_05_Tivita_multiorgan_masks@median_spectra@polygon#annotator1.feather"
        )
        df_example2_a2.to_feather(
            target_dir / "2021_02_05_Tivita_multiorgan_masks@median_spectra@polygon#annotator2.feather"
        )

        monkeypatch.setattr(settings, "_intermediates_dir_all", tmp_path)

        df = median_table(image_names=["i1"])
        assert_frame_equal(
            df,
            pd.DataFrame(
                [["i1", "bladder", 0, 0, "semantic#primary"]],
                columns=["image_name", "label_name", "label_index", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

        df = median_table(image_names=["i1", "i1"])
        assert_frame_equal(
            df,
            pd.DataFrame(
                [["i1", "bladder", 0, 0, "semantic#primary"]],
                columns=["image_name", "label_name", "label_index", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

        df = median_table(image_names=["i2", "i5"])
        assert_frame_equal(
            df,
            pd.DataFrame(
                [
                    ["i5", "bladder", 5, "polygon#annotator1"],
                    ["i2", "lung", 1, "semantic#primary"],
                ],
                columns=["image_name", "label_name", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

        # Annotation name as part of the image name
        df = median_table(
            image_names=[
                "i1",
                "i1#overlap@polygon#annotator1&polygon#annotator2",
                "i5@polygon#annotator1",
                "i5@polygon#annotator2",
            ]
        )
        assert_frame_equal(
            df,
            pd.DataFrame(
                [
                    ["i1", "bladder", 0, "semantic#primary"],
                    ["i1#overlap", "bladder", 4, "polygon#annotator1"],
                    ["i1#overlap", "bladder", 6, "polygon#annotator2"],
                    ["i5", "bladder", 5, "polygon#annotator1"],
                    ["i5", "bladder", 7, "polygon#annotator2"],
                ],
                columns=["image_name", "label_name", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

        # No annotation name if not available
        df_example2_a1.to_feather(target_dir / "2021_03_30_Tivita_studies@median_spectra.feather")
        df = median_table(dataset_name="2021_03_30_Tivita_studies")
        assert_frame_equal(
            df,
            pd.DataFrame(
                [["i1#overlap", "bladder", 0, 4], ["i5", "bladder", 0, 5]],
                columns=["image_name", "label_name", "label_index", "test_value"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

    def test_table_name(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        df_example1 = pd.DataFrame(
            [
                ["i1", "bladder", 0, 0],
                ["i3", "bladder", 0, 2],
                ["i4", "bladder", 0, 3],
                ["i2", "lung", 1, 1],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )

        df_example2 = pd.DataFrame(
            [
                ["i5", "bladder", 0, 4],
                ["i6", "bladder", 0, 5],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )
        df_example2_recalibrated = pd.DataFrame(
            [
                ["i5", "bladder", 0, 6],
                ["i6", "bladder", 0, 7],
            ],
            columns=["image_name", "label_name", "label_index", "test_value"],
        )

        target_dir = tmp_path / "tables"
        target_dir.mkdir(exist_ok=True, parents=True)

        df_example1.to_feather(
            target_dir / "2021_02_05_Tivita_multiorgan_semantic@median_spectra@semantic#primary.feather"
        )
        df_example2.to_feather(
            target_dir
            / "2021_02_05_Tivita_multiorgan_semantic#context_experiments@median_spectra@semantic#primary.feather"
        )
        df_example2_recalibrated.to_feather(
            target_dir
            / "2021_02_05_Tivita_multiorgan_semantic#context_experiments@recalibrated@median_spectra@semantic#primary.feather"
        )

        monkeypatch.setattr(settings, "_intermediates_dir_all", tmp_path)

        df_example1["annotation_name"] = "semantic#primary"
        df_example2["annotation_name"] = "semantic#primary"
        df_example2_recalibrated["annotation_name"] = "semantic#primary"

        assert_frame_equal(df_example1, median_table("2021_02_05_Tivita_multiorgan_semantic"))
        assert_frame_equal(df_example2, median_table("2021_02_05_Tivita_multiorgan_semantic#context_experiments"))
        assert_frame_equal(
            df_example2_recalibrated,
            median_table(
                dataset_name="2021_02_05_Tivita_multiorgan_semantic#context_experiments", table_name="recalibrated"
            ),
        )

        assert_frame_equal(
            median_table(image_names=["i5"]),
            pd.DataFrame(
                [["i5", "bladder", 0, 4, "semantic#primary"]],
                columns=["image_name", "label_name", "label_index", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )
        assert_frame_equal(
            median_table(image_names=["i5"], table_name="recalibrated"),
            pd.DataFrame(
                [["i5", "bladder", 0, 6, "semantic#primary"]],
                columns=["image_name", "label_name", "label_index", "test_value", "annotation_name"],
            ),
            check_dtype=False,
            check_categorical=False,
        )

    def test_semantic(self) -> None:
        df = median_table(image_names=["P062#2020_05_15_22_02_05"], label_mapping=settings_seg.label_mapping)
        assert_frame_equal(df, sort_labels(df))
        assert df["annotation_name"].unique().item() == "semantic#primary"
        assert settings_seg.label_mapping.is_index_valid(df["label_index_mapped"]).all()
        assert sorted(df["label_name_mapped"].tolist()) == sorted([
            "fat_subcutaneous",
            "lung",
            "heart",
            "skin",
            "muscle",
            "background",
            "background",
            "background",
            "background",
            "background",
            "background",
        ])
        assert sorted(df["label_name"].tolist()) == sorted([
            "fat_subcutaneous",
            "lung",
            "heart",
            "skin",
            "muscle",
            "blue_cloth",
            "white_compress",
            "metal",
            "anorganic_artifact",
            "foil",
            "glove",
        ])

        df2 = median_table(
            dataset_name="2021_02_05_Tivita_multiorgan_semantic", label_mapping=settings_seg.label_mapping
        )
        assert df2["annotation_name"].unique().item() == "semantic#primary"
        assert_frame_equal(
            df2.query("image_name == 'P062#2020_05_15_22_02_05'").reset_index(drop=True),
            df,
            check_categorical=False,
            check_dtype=False,
        )
        assert_frame_equal(df2, sort_labels(df2))

        df3 = median_table(
            dataset_name="2021_02_05_Tivita_multiorgan_semantic", annotation_name=["semantic#intra1", "semantic#inter1"]
        )
        assert_frame_equal(df3, sort_labels(df3))

        df_mapped = median_table(
            image_names=["P062#2020_05_15_22_02_05"],
            label_mapping=settings_seg.label_mapping,
            keep_mapped_columns=False,
        )
        assert len(df) == len(df_mapped)
        assert "label_name_mapped" not in df_mapped.columns
        assert "label_index_mapped" not in df_mapped.columns
        assert (df["label_name_mapped"] == df_mapped["label_name"]).all()
        assert (df["label_index_mapped"] == df_mapped["label_index"]).all()

    def test_annotation_names(self) -> None:
        df = median_table(image_names=["P091#2021_04_24_12_02_50@polygon#annotator1&polygon#annotator2"])
        assert df["annotation_name"].unique().tolist() == ["polygon#annotator1", "polygon#annotator2"]
        assert df["image_name"].unique().tolist() == ["P091#2021_04_24_12_02_50"]

        df2 = median_table(
            paths=[DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator1&polygon#annotator2")]
        )
        assert_frame_equal(df, df2)

    def test_mapping(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        df_example1 = pd.DataFrame(
            [
                ["i1", "sepsis", 0],
                ["i2", "no_sepsis", 1],
                ["i3", "name1", 2],
            ],
            columns=["image_name", "sepsis_status", "test_value"],
        )
        df_example2 = pd.DataFrame(
            [
                ["i4", "sepsis", 10],
                ["i5", "healthy", 20],
            ],
            columns=["image_name", "health_status", "test_value"],
        )

        tmp_dataset = tmp_path / "Tivita_dataset_test"
        tmp_dataset.mkdir(exist_ok=True, parents=True)
        tmp_data = tmp_dataset / "data"
        tmp_data.mkdir(exist_ok=True, parents=True)
        tmp_intermediates = tmp_dataset / "intermediates"
        tmp_intermediates.mkdir(exist_ok=True, parents=True)
        tmp_tables = tmp_intermediates / "tables"
        tmp_tables.mkdir(exist_ok=True, parents=True)

        df_example1.to_feather(tmp_tables / "Tivita_dataset_test@median_spectra@semantic#primary.feather")
        df_example2.to_feather(tmp_tables / "Tivita_dataset_test2@median_spectra@semantic#primary.feather")

        monkeypatch.setattr(settings, "_datasets", None)
        monkeypatch.setattr(settings, "_intermediates_dir_all", tmp_intermediates)
        monkeypatch.setenv("PATH_Tivita_dataset_test", str(tmp_dataset))

        df_example1["annotation_name"] = "semantic#primary"

        df_example1_mapped = df_example1.copy()
        df_example1_mapped["sepsis_status_index"] = [0, 1, 2]

        # Check single column mapping
        sepsis_status_mapping = LabelMapping({"sepsis": 0, "no_sepsis": 1, "healthy": 1}, last_valid_label_index=1)
        assert_frame_equal(
            df_example1_mapped,
            median_table("Tivita_dataset_test", additional_mappings={"sepsis_status": sepsis_status_mapping}),
        )

        # Check image_labels column with one dimension
        df_combined = median_table(
            image_names=["i1", "i2", "i4", "i5"],
            image_labels_column=[
                {"meta_attributes": ["sepsis_status", "health_status"], "image_label_mapping": sepsis_status_mapping},
            ],
        )
        df_combined_true = pd.DataFrame(
            [
                ["i1", np.nan, 0, "semantic#primary", "sepsis", 0],
                ["i2", np.nan, 1, "semantic#primary", "no_sepsis", 1],
                ["i4", "sepsis", 10, "semantic#primary", np.nan, 0],
                ["i5", "healthy", 20, "semantic#primary", np.nan, 1],
            ],
            columns=["image_name", "health_status", "test_value", "annotation_name", "sepsis_status", "image_labels"],
        )
        assert_frame_equal(df_combined, df_combined_true, check_categorical=False, check_dtype=False)

        # Check image_labels column with two dimensions
        df_combined = median_table(
            image_names=["i1", "i2", "i4", "i5"],
            image_labels_column=[
                {"meta_attributes": ["sepsis_status", "health_status"], "image_label_mapping": sepsis_status_mapping},
                {
                    "meta_attributes": ["image_name"],
                    "image_label_mapping": LabelMapping({"i1": 0, "i2": 0, "i4": 1, "i5": 1}),
                },
            ],
        )
        df_combined_true = pd.DataFrame(
            [
                ["i1", np.nan, 0, "semantic#primary", "sepsis", [0, 0]],
                ["i2", np.nan, 1, "semantic#primary", "no_sepsis", [1, 0]],
                ["i4", "sepsis", 10, "semantic#primary", np.nan, [0, 1]],
                ["i5", "healthy", 20, "semantic#primary", np.nan, [1, 1]],
            ],
            columns=["image_name", "health_status", "test_value", "annotation_name", "sepsis_status", "image_labels"],
        )
        assert_frame_equal(df_combined, df_combined_true, check_categorical=False, check_dtype=False)

    def test_dataset_concatenation(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        df1 = median_table(dataset_name="2022_10_24_Tivita_sepsis_ICU#calibrations")
        df2 = median_table(dataset_name="2022_10_24_Tivita_sepsis_ICU#subjects")
        df = median_table(dataset_name="2022_10_24_Tivita_sepsis_ICU")

        assert_frame_equal(sort_labels(pd.concat([df1, df2])), df)

    def test_config(self) -> None:
        config = Config({
            "input/data_spec": "htc/models/data/pigs_semantic-only_5foldsV2.json",
        })
        spec = DataSpecification.from_config(config)
        df = median_table(config=config)
        assert sorted([p.image_name() for p in spec.paths()]) == sorted(df["image_name"].unique().tolist())
