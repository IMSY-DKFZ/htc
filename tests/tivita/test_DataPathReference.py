# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pytest import MonkeyPatch

from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config

# from htc.utils.Datasets import Datasets


class TestDataPathReference:
    # def test_reference(self, monkeypatch: MonkeyPatch) -> None:
    #     paths = list(DataPath.iterate(settings.data_dirs.unsorted))
    #     for p in paths:
    #         assert p().exists(), p
    #         assert p.dataset_name not in p.dataset_settings["excluded_datasets"]
    #         assert all(e not in str(p()) for e in p.dataset_settings["excluded_datasets"])
    #         assert all(e not in p.image_name() for e in p.dataset_settings["excluded_datasets"])

    #     path = DataPath.from_image_name("ref#2021_03_30_Tivita_studies#2021_03_30_14_41_34")
    #     assert path.dataset_name == "2021_03_30_Tivita_studies"
    #     assert path.cube_path().exists()

    #     # We can still access the images by name even if the data directory is not available anymore (because it is cached)
    #     data_dir_tmp = Datasets(network_dir=settings.datasets.network_dir)
    #     monkeypatch.setattr(settings, "_datasets", data_dir_tmp)
    #     assert settings.data_dirs.unsorted is None
    #     assert DataPath.from_image_name("ref#2021_03_30_Tivita_studies#2021_03_30_14_44_01").cube_path().exists()

    def test_intermediates_only(self, monkeypatch: MonkeyPatch) -> None:
        monkeypatch.setattr(settings.datasets, "network_dir", None)
        path = DataPath.from_image_name("ref#2020_07_23_hyperspectral_MIC_organ_database#2020_02_20_18_29_29")
        assert path.timestamp == "2020_02_20_18_29_29"
        sample = DatasetImage([path], train=False, config=Config({"input/preprocessing": "L1"}))[0]
        assert sample["features"].shape == (480, 640, 100)

    def test_rgb_reconstructed(self, monkeypatch: MonkeyPatch) -> None:
        path = DataPath.from_image_name("ref#2021_03_30_Tivita_studies#2021_03_30_14_41_34")
        assert path.intermediates_dir is not None
        assert path.rgb_path_reconstructed().is_relative_to(path.data_dir)
        rgb1 = path.read_rgb_reconstructed()

        monkeypatch.setattr(settings.datasets, "network_dir", None)
        DataPath._data_paths_cache = {}
        path = DataPath.from_image_name("ref#2021_03_30_Tivita_studies#2021_03_30_14_41_34")
        assert path.rgb_path_reconstructed().is_relative_to(path.intermediates_dir)
        rgb2 = path.read_rgb_reconstructed()
        assert (rgb1 == rgb2).all()
