# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import LogCaptureFixture, MonkeyPatch

from htc.dataset_preparation.DatasetGenerator import DatasetGenerator
from htc.models.data.run_pig_dataset import filter_test, filter_train, test_set
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DataPathMultiorgan import DataPathMultiorgan
from htc.tivita.DataPathTissueAtlas import DataPathTissueAtlas
from htc.tivita.DataPathTivita import DataPathTivita
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.blosc_compression import decompress_file
from htc.utils.Config import Config
from htc.utils.Datasets import Datasets
from htc.utils.helper_functions import median_table
from htc_projects.camera.calibration.CalibrationSwap import CalibrationFiles, CalibrationSwap
from htc_projects.camera.colorchecker.ColorcheckerTransform import ColorcheckerTransform


class TestDataPath:
    def test_multiorgan(self) -> None:
        path_multi = DataPath.from_image_name("P043#2019_12_20_12_38_35")

        assert isinstance(path_multi, DataPathMultiorgan)
        assert path_multi.subject_name == "P043"
        assert path_multi.timestamp == "2019_12_20_12_38_35"
        assert path_multi.image_name() == "P043#2019_12_20_12_38_35"
        datetime = path_multi.datetime()
        assert datetime.year == 2019
        assert datetime.month == 12
        assert datetime.day == 20
        assert datetime.hour == 12
        assert datetime.minute == 38
        assert datetime.second == 35

    def test_single(self) -> None:
        atlas_dir = (
            settings.datasets.network_data
            / "2020_07_23_hyperspectral_MIC_organ_database/data/Catalogization_tissue_atlas"
        )
        path_single = next(iter(DataPath.iterate(atlas_dir)))

        assert isinstance(path_single, DataPathTissueAtlas)
        assert path_single.subject_name == "P002"
        assert path_single.timestamp == "2018_08_06_11_30_26"
        assert path_single.image_name() == "stomach#P002_OP002_2018_08_06_Experiment1#2018_08_06_11_30_26"
        assert path_single.organ == "stomach"
        assert path_single.subject_folder == "P002_OP002_2018_08_06_Experiment1"
        assert path_single.organ_folder == "Cat_0001_stomach"

    def test_sorted_train(self) -> None:
        files_multi = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))
        files_multi2 = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train]))

        assert [f() for f in files_multi] == [f() for f in files_multi2]
        assert all(f.subject_name not in test_set for f in files_multi)

    def test_fallback(self) -> None:
        path1 = next(DataPath.iterate(settings.data_dirs.semantic))
        path2 = next(DataPath.iterate(settings.data_dirs.semantic / "subjects"))
        path3 = next(DataPath.iterate(settings.data_dirs.semantic / "subjects" / "P041"))
        path4 = next(DataPath.iterate(settings.data_dirs.semantic / "subjects" / "P041" / "2019_12_14_12_00_16"))

        assert isinstance(path1, DataPathMultiorgan)
        assert isinstance(path2, DataPathTivita)
        assert isinstance(path3, DataPathTivita)
        assert isinstance(path4, DataPathTivita)

    @pytest.mark.parametrize(
        "dataset_name", ["2021_02_05_Tivita_multiorgan_semantic", "2021_02_05_Tivita_multiorgan_masks"]
    )
    def test_dataset_settings(self, dataset_name: str) -> None:
        data_iter = DataPath.iterate(settings.data_dirs[dataset_name], filters=[filter_train])
        path = next(data_iter)
        path2 = next(data_iter)

        assert path.dataset_settings["shape"] == (480, 640, 100)
        assert "label_mapping" in path.dataset_settings
        assert path.dataset_settings == path2.dataset_settings

    def test_from_image_name(self) -> None:
        path = DataPath.from_image_name("P043#2019_12_20_12_38_35")

        assert path.subject_name == "P043"
        assert path.timestamp == "2019_12_20_12_38_35"
        assert path().exists()
        assert len(list(path().glob("*.png"))) > 0

        # We can access every image also via its name
        for p in DataPath.iterate(settings.data_dirs.semantic):
            assert p == DataPath.from_image_name(p.image_name())

    @pytest.mark.parametrize(
        "dataset_name", ["2021_02_05_Tivita_multiorgan_semantic", "2021_02_05_Tivita_multiorgan_masks"]
    )
    def test_overlap(self, dataset_name: str) -> None:
        paths = list(DataPath.iterate(settings.data_dirs[dataset_name]))
        assert all(not p.is_overlap for p in paths)

    def test_overlap_reading(self) -> None:
        path = next(DataPath.iterate(settings.data_dirs.masks / "overlap"))
        assert path.is_overlap
        assert path.image_name().endswith("overlap")
        assert "2021_02_05_Tivita_multiorgan_semantic" in str(path.cube_path())
        assert "2021_02_05_Tivita_multiorgan_semantic" in str(path.camera_meta_path())
        assert "2021_02_05_Tivita_multiorgan_semantic" in str(path.rgb_path_reconstructed())
        assert "2021_02_05_Tivita_multiorgan_masks" in str(path.segmentation_path())

        path2 = next(DataPath.iterate(settings.data_dirs.masks))
        assert not path2.is_overlap
        assert not path2.image_name().endswith("overlap")
        assert "2021_02_05_Tivita_multiorgan_masks" in str(path2.cube_path())
        assert "2021_02_05_Tivita_multiorgan_masks" in str(path2.camera_meta_path())
        assert "2021_02_05_Tivita_multiorgan_masks" in str(path2.rgb_path_reconstructed())
        assert "2021_02_05_Tivita_multiorgan_masks" in str(path2.segmentation_path())

    def test_filtering(self) -> None:
        paths_all = DataPath.iterate(settings.data_dirs.semantic)
        paths_train = DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train])
        paths_test = DataPath.iterate(settings.data_dirs.semantic, filters=[filter_test])
        filter_2019 = lambda p: p.timestamp.startswith("2019")
        paths_2019 = DataPath.iterate(settings.data_dirs.semantic, filters=[filter_2019])
        paths_train_2019 = DataPath.iterate(settings.data_dirs.semantic, filters=[filter_train, filter_2019])

        assert sorted(p.image_name() for p in paths_all) == sorted(
            [p.image_name() for p in paths_train] + [p.image_name() for p in paths_test]
        )
        assert all(p.subject_name not in test_set for p in paths_train)
        assert all(p.timestamp.startswith("2019") for p in paths_2019)
        assert all(p.timestamp.startswith("2019") and p.subject_name not in test_set for p in paths_train_2019)

    def test_path_creation(self) -> None:
        path = DataPath(settings.data_dirs.studies / "2021_03_30_straylight/Tivita/colorchecker/2021_03_30_13_54_53")
        assert path.cube_path().exists()
        assert path.timestamp == "2021_03_30_13_54_53"
        assert "shape" in path.dataset_settings

    def test_path_cache(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        # Create temporary data dirs to check that the cache is build only twice (local and network)
        tmp_local = tmp_path / "local"
        tmp_network = tmp_path / "network"
        tmp_network_data = tmp_network / "Biophotonics/Data"
        dirs = [
            tmp_local / "dataset1/data/subjects/subject1/img1",
            tmp_local / "dataset1/data/subjects/subject2/img2",
            tmp_network_data / "dataset2/data/subjects/subject3/img3",
            tmp_network_data / "dataset2/data/subjects/subject3/img4",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        tmp_data_dirs = Datasets(network_dir=tmp_network)

        # If an environment variable is set, it is considered local
        monkeypatch.setenv("PATH_Tivita_multiorgan_dataset1", str(tmp_local / "dataset1"))

        for dataset in [tmp_local / "dataset1", tmp_network_data / "dataset2"]:
            dsettings_content = {
                "dataset_name": dataset.name,
                "data_path_class": "htc.tivita.DataPathMultiorgan>DataPathMultiorgan",
            }
            (dataset / "data" / "dataset_settings.json").write_text(json.dumps(dsettings_content))

            f = DatasetGenerator(output_path=dataset)
            f.meta_table()
            assert (dataset / "intermediates" / "tables" / f"{dataset.name}@meta.feather").exists()

            tmp_data_dirs.add_dir(f"PATH_Tivita_multiorgan_{dataset.name}", dataset.name)

        monkeypatch.setattr(settings, "_datasets", tmp_data_dirs)

        # Start with an empty cache for this test (e.g. local cache may already be filled by other tests)
        monkeypatch.setattr(DataPath, "_local_meta_cache", None)
        monkeypatch.setattr(DataPath, "_network_meta_cache", None)
        monkeypatch.setattr(DataPath, "_meta_labels_cache", {})
        monkeypatch.setattr(DataPath, "_data_paths_cache", {})

        n_calls = 0

        def call_count(func):
            def _call_count(*args, **kwargs):
                nonlocal n_calls
                n_calls += 1
                return func(*args, **kwargs)

            return _call_count

        DataPath._build_cache = call_count(DataPath._build_cache)

        assert DataPath.from_image_name("subject1#img1")() == tmp_local / "dataset1/data/subjects/subject1/img1"
        assert n_calls == 1
        assert DataPath.from_image_name("subject2#img2")() == tmp_local / "dataset1/data/subjects/subject2/img2"
        assert n_calls == 1

        assert DataPath.from_image_name("subject3#img3")() == tmp_network_data / "dataset2/data/subjects/subject3/img3"
        assert n_calls == 2
        assert DataPath.from_image_name("subject3#img4")() == tmp_network_data / "dataset2/data/subjects/subject3/img4"
        assert n_calls == 2

        assert DataPath.from_image_name("subject2#img2")() == tmp_local / "dataset1/data/subjects/subject2/img2"
        assert n_calls == 2

    def test_meta(self) -> None:
        path = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        assert path.meta("Camera_CamID") == "0102-00085"
        assert "Camera_CamID" in path.meta().keys()

        t = ColorcheckerTransform()
        for camera_name, path in t.checkerboard_paths.items():
            assert path.meta("camera_name") == camera_name

        path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        meta = path.meta()
        assert "label_meta" in meta
        assert "camera_name" in meta
        assert meta["label_meta"]["omentum"]["situs"] == 2

    def test_meta_labels(self) -> None:
        DataPath._meta_labels_cache = {}
        path = DataPath.from_image_name("P091#2021_04_24_12_02_50")
        meta_labels = path.read_annotation_meta()
        assert meta_labels["label_meta"]["omentum"]["situs"] == path.meta("label_meta/omentum/situs") == 2
        assert len(DataPath._meta_labels_cache) == 1

    def test_annotation_name(self) -> None:
        data = decompress_file(settings.intermediates_dir_all / "segmentations/P091#2021_04_24_12_02_50.blosc")
        path = DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator2")
        assert np.all(path.read_segmentation() == data["polygon#annotator2"])
        assert np.all(path.read_segmentation("polygon#annotator1") == data["polygon#annotator1"])

        path = DataPath.from_image_name("P091#2021_04_24_12_02_50@polygon#annotator1&polygon#annotator2")
        seg = path.read_segmentation()
        assert type(seg) == dict and len(seg) == 2
        assert np.all(seg["polygon#annotator1"] == data["polygon#annotator1"]) and np.all(
            seg["polygon#annotator2"] == data["polygon#annotator2"]
        )

        seg1 = path.read_segmentation(["polygon#annotator1", "polygon#annotator2"])
        seg2 = path.read_segmentation("polygon#annotator1&polygon#annotator2")
        assert seg1.keys() == seg2.keys()
        for v1, v2 in zip(seg1.values(), seg2.values(), strict=True):
            assert np.all(v1 == v2)

        for i, path in enumerate(DataPath.iterate(settings.data_dirs.masks, annotation_name="polygon#annotator1")):
            if i == 2:
                break

            data = decompress_file(settings.intermediates_dir_all / f"segmentations/{path.image_name()}.blosc")
            assert np.all(path.read_segmentation() == data["polygon#annotator1"])

    @pytest.mark.parametrize("image_name", ["P058#2020_05_13_18_09_26", "S001#2022_10_24_13_49_45"])
    def test_params(self, image_name: str, monkeypatch: MonkeyPatch, check_sepsis_data_accessible: Callable) -> None:
        if image_name == "S001#2022_10_24_13_49_45":
            check_sepsis_data_accessible()
            monkeypatch.delattr("htc.tivita.functions_official.calc_sto2")

        path = DataPath.from_image_name(image_name)
        cube = path.read_cube()

        parameter_names = ["StO2", "NIR", "TWI", "OHI", "TLI", "THI"]
        sample = DatasetImage(
            [path],
            train=False,
            config=Config({
                "input/preprocessing": "parameter_images",
                "input/parameter_names": parameter_names,
            }),
        )[0]

        n_calls = 0

        def call_count(func):
            def _call_count(*args, **kwargs):
                nonlocal n_calls
                n_calls += 1
                return func(*args, **kwargs)

            return _call_count

        monkeypatch.setattr(path, "_load_precomputed_parameters", call_count(path._load_precomputed_parameters))

        # Compute parameter images again
        computed = []
        for name in parameter_names:
            func = getattr(path, f"compute_{name.lower()}")
            computed.append(func(cube))
        assert n_calls == 0

        # Load precomputed images and check that they can be loaded and are equal to the computed ones
        monkeypatch.delattr("htc.tivita.functions_official.detect_background")
        for i, name in enumerate(parameter_names):
            func = getattr(path, f"compute_{name.lower()}")
            precomputed = func()

            assert np.allclose(computed[i].data, precomputed.data, atol=1e-05)
            assert np.all(computed[i].mask == precomputed.mask)
            assert computed[i].fill_value == precomputed.fill_value
            assert np.allclose(sample["features"][..., i].numpy(), precomputed.data, atol=1e-05)
            assert n_calls == i + 1

    def test_sto2_version(self, check_sepsis_data_accessible: Callable) -> None:
        path = DataPath.from_image_name("P058#2020_05_13_18_09_26")
        assert np.allclose(path.compute_sto2(), path.compute_sto2(version="calc_sto2"))

        check_sepsis_data_accessible()
        path = DataPath.from_image_name("S001#2022_10_24_13_49_45")
        assert np.allclose(path.compute_sto2(), path.compute_sto2(version="calc_sto2_2_helper"))

    def test_is_cube_valid(self, tmp_path: Path, caplog: LogCaptureFixture) -> None:
        def write_cube(cube: np.ndarray) -> DataPath:
            # Revert the reading process
            cube = np.swapaxes(cube, 0, 1)
            shape = np.array(cube.shape).astype(">i")
            cube = np.flip(cube, axis=1)
            cube = cube.flatten()
            cube = cube.astype(">f")

            tmp_img_dir = tmp_path / "img"
            tmp_img_dir.mkdir(parents=True, exist_ok=True)

            with (tmp_img_dir / "img_SpecCube.dat").open("wb") as f:
                f.write(shape.tobytes())
                f.write(cube.tobytes())

            dsettings = DatasetSettings({"shape": (480, 640, 100)})
            return DataPath(tmp_img_dir, dataset_settings=dsettings)

        # Valid
        cube = np.ones((480, 640, 100), dtype=np.float32)
        path = write_cube(cube)
        assert np.all(path.read_cube() == 1)
        assert path.is_cube_valid()
        assert len(caplog.records) == 0

        # Warning due to zero value
        cube[0, 0, 0] = 0
        path = write_cube(cube)
        cube_read = path.read_cube()
        assert cube_read[0, 0, 0] == 0 and np.all(
            np.unique(cube_read, return_counts=True)[1] == np.array([1, 30719999])
        )
        assert path.is_cube_valid()
        assert (
            len(caplog.records) == 1
            and "img has 1 zero values" in caplog.records[0].msg
            and caplog.records[0].levelno == logging.WARNING
        )

        # Invalid, zero pixel
        cube = np.ones((480, 640, 100), dtype=np.float32)
        cube[0, 1, :] = 0
        path = write_cube(cube)
        cube_read = path.read_cube()
        assert np.all(cube_read[0, 1, :] == 0) and np.all(
            np.unique(cube_read, return_counts=True)[1] == np.array([100, 30719900])
        )
        assert not path.is_cube_valid()
        assert (
            len(caplog.records) == 2
            and "img has 1 zero pixels" in caplog.records[1].msg
            and caplog.records[1].levelno == logging.ERROR
        )

        # Invalid all values negative
        cube = np.ones((480, 640, 100), dtype=np.float32) * -1
        path = write_cube(cube)
        cube_read = path.read_cube()
        assert np.all(cube_read == -1)
        assert not path.is_cube_valid()
        assert (
            len(caplog.records) == 3
            and "img contains only negative values" in caplog.records[2].msg
            and caplog.records[2].levelno == logging.ERROR
        )

        # Warning, negative pixels
        cube = np.ones((480, 640, 100), dtype=np.float32)
        cube[0, 1, :] = -1
        path = write_cube(cube)
        cube_read = path.read_cube()
        assert np.all(cube_read[0, 1, :] == -1) and np.all(
            np.unique(cube_read, return_counts=True)[1] == np.array([100, 30719900])
        )
        assert path.is_cube_valid()
        assert (
            len(caplog.records) == 4
            and "img contains 1 negative pixels" in caplog.records[3].msg
            and caplog.records[3].levelno == logging.WARNING
        )

        # Invalid, wrong size
        cube = np.ones((40, 60, 20), dtype=np.float32)
        path = write_cube(cube)
        assert not path.is_cube_valid()
        assert (
            len(caplog.records) == 5
            and "not have the correct shape" in caplog.records[4].msg
            and caplog.records[4].levelno == logging.ERROR
        )

        cube = np.ones((480, 640, 100), dtype=np.float32)
        cube[0, 1, :] = np.nan
        path = write_cube(cube)
        cube_read = path.read_cube()
        assert not path.is_cube_valid()
        assert (
            len(caplog.records) == 6
            and "contains invalid values (nan/inf) " in caplog.records[5].msg
            and caplog.records[5].levelno == logging.ERROR
        )

    def test_read_colorchecker_mask(self) -> None:
        t = ColorcheckerTransform()
        path = t.checkerboard_paths["0102-00085_correct-1"]
        cc_mask = path.read_colorchecker_mask(return_spectra=True, normalization=1)
        assert np.all(
            np.median(cc_mask["spectra"][0, 0], axis=0)
            == cc_mask["median_table"].query("row == 0 and col == 0")["median_normalized_spectrum"].item()
        )

        # test MITK mask reading:
        path = DataPath(
            settings.data_dirs.studies / "2023_02_09_colorchecker_MIC1_TivitaMini/cc_black/2023_02_08_11_04_11"
        )
        cc_mask = path.read_colorchecker_mask(return_spectra=True, normalization=1)
        assert all(np.unique(cc_mask["mask"]) == np.array([0, 24], dtype=np.uint8))

    def test_read_cube_raw(self) -> None:
        path = DataPath(
            settings.data_dirs.rat
            / "straylight_experiments/calibration_white/0202-00118/OR-situs+ceiling/2023_11_15_11_59_30"
        )
        cube = path.read_cube_raw()
        cube_calibrated = path.read_cube()
        assert np.all(cube > cube_calibrated)

        # test whether manual handover of calibration files yields same result
        path_dark = settings.data_dirs.studies / "white_balances/0202-00118/2023_11_14/DarkPattern.dpic"
        path_white = settings.data_dirs.studies / "white_balances/0202-00118/2023_11_14/EqBinWhiteCube.dat"
        calibration_original = CalibrationFiles(
            dark=path_dark,
            white=path_white,
            cam_id="0202-00118",
            timestamp_dark=pd.Timestamp("2023-11-14"),
            timestamp_white=pd.Timestamp("2023-11-14"),
        )
        calibration_original = CalibrationSwap()._cached_calibration_data(calibration_original)
        cube2 = path.read_cube_raw(calibration_original=calibration_original)
        assert np.all(cube == cube2)

    def test_compute_oversaturation_mask(self) -> None:
        path = DataPath(
            settings.data_dirs.rat
            / "straylight_experiments/calibration_white/0202-00118/OR-situs+ceiling/2023_11_15_11_59_30"
        )
        oversaturation_mask = path.compute_oversaturation_mask()
        assert (
            np.sum(oversaturation_mask) == np.shape(oversaturation_mask)[0] * np.shape(oversaturation_mask)[1]
        )  # under this straylight condition, all pixels of the white tile image are oversaturated

        path = DataPath(
            settings.data_dirs.rat
            / "straylight_experiments/calibration_white/0202-00118/no_straylight/2023_11_14_08_52_15"
        )
        oversaturation_mask = path.compute_oversaturation_mask(
            threshold=1020
        )  # threshold is set to higher than 1000 as a camera calibration has very recently been performed
        assert (
            np.sum(oversaturation_mask) == 0
        )  # as no straylight is present, no pixels of the white tile image are oversaturated

    def test_from_tables(self) -> None:
        df = pd.DataFrame({
            "image_name": ["P041#2019_12_14_13_33_30", "P041#2019_12_14_13_33_30"],
            "annotation_name": ["semantic#primary", "semantic#primary"],
        })
        paths = DataPath.from_table(df)
        assert len(paths) == 1
        assert paths[0].image_name_annotations() == "P041#2019_12_14_13_33_30@semantic#primary"

        df = pd.DataFrame({
            "image_name": ["P041#2019_12_14_13_33_30", "P041#2019_12_14_13_33_30"],
            "annotation_name": ["semantic#primary", "semantic#inter1"],
        })
        paths = DataPath.from_table(df)
        assert len(paths) == 1
        assert paths[0].image_name_annotations() == "P041#2019_12_14_13_33_30@semantic#primary&semantic#inter1"

        df = pd.DataFrame({
            "image_name": ["P041#2019_12_14_13_33_30", "P043#2019_12_20_10_08_40"],
            "annotation_name": ["semantic#primary", "semantic#inter1"],
        })
        paths = DataPath.from_table(df)
        assert len(paths) == 2
        assert paths[0].image_name_annotations() == "P041#2019_12_14_13_33_30@semantic#primary"
        assert paths[1].image_name_annotations() == "P043#2019_12_20_10_08_40@semantic#inter1"

    def test_path_links(self) -> None:
        paths = list(DataPath.iterate(settings.data_dirs.haemorrhage_pig))
        names = {p.image_name_annotations() for p in paths}

        links_file = settings.data_dirs.haemorrhage_pig / "path_links.json"
        with links_file.open() as f:
            link_data = json.load(f)

        image_names = []
        for links in link_data.values():
            image_names += links

        assert len(set(image_names)) == len(image_names)
        assert set(image_names).issubset(names)

        df = median_table(dataset_name="2022_03_08_Tivita_haemorrhage_pig")
        assert df["image_name"].nunique() == len(names)

        names_table = df["image_name"] + "@" + df["annotation_name"]
        assert set(names_table) == names

    def test_dataset_or_data_dir(self) -> None:
        paths_data = list(DataPath.iterate(settings.data_dirs.semantic))
        paths_dataset = list(DataPath.iterate(settings.datasets.semantic["path_dataset"]))

        assert type(paths_data[0]) == type(paths_dataset[0]) == DataPathMultiorgan
        assert paths_data == paths_dataset
