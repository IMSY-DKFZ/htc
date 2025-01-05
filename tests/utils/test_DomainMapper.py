# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper
from htc.utils.helper_functions import median_table
from htc_projects.camera.settings_camera import settings_camera


class TestDomainMapper:
    def test_cam_name(self) -> None:
        config = Config({
            "input/data_spec": "data/pigs_masks_loocv_4cam.json",
            "input/target_domain": ["camera_index"],
            "input/domain_loss_weighting": {"camera_index": 1},
        })

        domain_mapper = DomainMapper.from_config(config)["camera_index"]
        data_spec = DataSpecification.from_config(config)

        assert set(settings_camera.cameras).issubset(domain_mapper.domains), (
            "Not all cameras found in the domains of the DomainMapper"
        )
        assert len(domain_mapper) == len(settings_camera.cameras)

        df = median_table(dataset_name="2021_02_05_Tivita_multiorgan_masks")
        for path in data_spec.paths():
            image_name = f"{path.subject_name}#{path.timestamp}"

            if image_name in df["image_name"].to_list():
                correct = df.query("image_name == @image_name")["camera_name"].values[0]
                out = domain_mapper.domain_name(image_name)
                assert correct == out, (
                    f"The domain name for image_name: {image_name} is incorrectly specified "
                    f"as {out} but it should be {correct}"
                )

    def test_subject_name(self) -> None:
        config = Config({
            "input/data_spec": "data/pigs_masks_loocv_4cam.json",
            "input/target_domain": ["subject_index"],
            "input/domain_loss_weighting": {"subject_index": 1},
        })

        domain_mapper = DomainMapper.from_config(config)["subject_index"]
        data_spec = DataSpecification.from_config(config)
        pigs = [path.subject_name for path in data_spec.paths()]

        assert set(pigs).issubset(domain_mapper.domains), "Not all pigs found in the domains of the DomainMapper"

        for path in data_spec.paths():
            image_name = path.image_name()
            correct = path.subject_name
            out = domain_mapper.domain_name(image_name)
            assert correct == out, (
                f"The domain name for image_name: {image_name} is incorrectly specified "
                f"as {out} but it should be {correct}"
            )

    def test_species_index(self) -> None:
        paths = [
            DataPath.from_image_name("P043#2019_12_20_12_38_35"),
            DataPath.from_image_name("R002#2023_09_19_10_14_28#0202-00118"),
        ]
        domain_mapper = DomainMapper(paths_or_specs=paths, target_domain="species_index")

        assert domain_mapper.domain_name("P043#2019_12_20_12_38_35") == "pig"
        assert domain_mapper.domain_index("P043#2019_12_20_12_38_35") == 0
        assert domain_mapper.domain_name("R002#2023_09_19_10_14_28#0202-00118") == "rat"
        assert domain_mapper.domain_index("R002#2023_09_19_10_14_28#0202-00118") == 1

    def test_from_config(self) -> None:
        config = Config({
            "input/data_spec": "pigs_semantic-only_5foldsV2.json",
            "input/target_domain": ["subject_index"],
        })
        domain_mapper = DomainMapper.from_config(config)["subject_index"]
        assert domain_mapper.index_to_domain(0) == "P041"
        assert type(config["domain_mappings"]) == dict
        assert isinstance(config["domain_mappings"]["subject_index"], DomainMapper)

    def test_shuffle_domains(self) -> None:
        config = Config({
            "input/data_spec": "data/pigs_masks_loocv_4cam.json",
            "input/target_domain": ["subject_index"],
            "input/shuffle_domains": True,
            "input/domain_loss_weighting": {"subject_index": 1},
        })

        domain_mapper = DomainMapper.from_config(config)["subject_index"]

        d1 = domain_mapper.domain_name("P041#2019_12_14_10_51_18")

        assert d1 != "P041", "The domains seem to have not been shuffled"
