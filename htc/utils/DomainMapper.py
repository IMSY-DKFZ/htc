# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import random
from typing import Any, Union

import numpy as np
import pandas as pd
from typing_extensions import Self

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import median_table


class DomainMapper:
    def __init__(
        self, paths_or_specs: Union[list[DataPath], DataSpecification], target_domain: str, shuffle_domains: str = False
    ):
        """
        Helper class to handle the mapping of paths to the corresponding domain value based on a pool of paths. Domain mapper also works with using multiple domains simultaneously as well.

        >>> paths = [DataPath.from_image_name('P043#2019_12_20_12_38_35'), DataPath.from_image_name('P084#2021_03_21_21_14_20')]
        >>> mapper = DomainMapper(paths_or_specs=paths, target_domain='subject_index')
        >>> mapper.domain_name('P043#2019_12_20_12_38_35')
        'P043'
        >>> mapper.domain_index('P043#2019_12_20_12_38_35')
        0
        >>> mapper.domain_index('P084#2021_03_21_21_14_20')
        1

        Args:
            paths_or_specs: List of paths to be used as pool (either directly from the list or from the spec).
            target_domain: The domain which should be extracted from the paths.
            shuffle_domains: If True, domains are assigned randomly. May be used for debugging purposes to check whether the domain has an effect.
        """
        self.paths_or_specs = paths_or_specs
        self.target_domain = target_domain
        self.shuffle_domains = shuffle_domains

        self.domains, self.domain_mapping, self.domain_colors = self._init_attributes()
        self.img_per_domain = {
            domain: np.sum(np.array(list(self.domain_mapping.values())) == domain) for domain in self.domains
        }

    def __len__(self):
        """
        Returns: The number of unique domain ids (e.g. number of cameras).
        """
        return len(self.domains)

    def _init_attributes(self) -> tuple[Union[list, list[str]], dict, Any]:
        paths = None

        if isinstance(self.paths_or_specs, DataSpecification):
            paths = self.paths_or_specs.paths()
        elif isinstance(self.paths_or_specs, list):
            paths = self.paths_or_specs
        else:
            ValueError(f"Unknown data type for parameter paths_or_specs: {type(self.paths_or_specs)}")

        dataset = {x.subject_name for x in paths}
        dataset = sorted(dataset)

        domain_colors = None
        if "camera_index" == self.target_domain:
            domains, domain_mapping = self._cam_domains(dataset, paths)
            try:
                from htc.cameras.settings_cam import settings_cam

                domain_colors = settings_cam.camera_name_colors
            except ImportError:
                pass
        elif "subject_index" == self.target_domain:
            domains, domain_mapping = self._pig_domains(dataset, paths)
        elif "species_index" == self.target_domain:
            domains, domain_mapping = self._species_domains(paths)
            try:
                from htc.human.settings_human import settings_human

                domain_colors = settings_human.species_colors
            except ImportError:
                pass
        elif "no_domain" == self.target_domain:
            domains = ["None"]
            domain_mapping = {x.image_name(): domains[0] for x in paths}
        else:
            raise ValueError(f"Currently using {self.target_domain} as a target domain for training is not supported.")

        if domain_colors is None:
            domain_colors = {domain: "#FFFFFF" for domain in domains}

        # baseline case where the domain ids are shuffled, to check if the domain task is acting as a regularization
        if self.shuffle_domains:
            mappings = list(domain_mapping.values())
            random.Random(0).shuffle(mappings)  # Does not affect global seed
            domain_mapping = dict(zip(list(domain_mapping.keys()), mappings))

        return domains, domain_mapping, domain_colors

    @staticmethod
    def _cam_domains(dataset: list[str], paths: list[DataPath]) -> tuple[list, dict]:
        df = median_table(image_names=[x.image_name() for x in paths])

        df_train = pd.DataFrame(dataset, columns=["subject_name"])
        df_train = df_train.merge(df[["subject_name", "image_name", "camera_name"]], on="subject_name", how="left")

        assert df_train.shape[0] != 0, (
            "No camera information found for the pig ids. Please make sure that "
            "the current data specification file is specified in the configs. Note "
            "the camera information is only available for Tivita masks dataset."
        )

        df_train["camera_name"] = df_train["camera_name"].astype("category")

        domains = df_train["camera_name"].cat.categories.to_list()
        domain_mapping = df_train.set_index("image_name").to_dict()["camera_name"]

        return domains, domain_mapping

    @staticmethod
    def _pig_domains(dataset: list[str], paths: list[DataPath]) -> tuple[list, dict]:
        return dataset, {x.image_name(): x.subject_name for x in paths}

    @staticmethod
    def _species_domains(paths: list[DataPath]) -> tuple[list, dict]:
        domains = set()
        domain_mapping = {}
        for p in paths:
            if p.subject_name.startswith("SPACE_"):
                domain_mapping[p.image_name()] = "human"
                domains.add("human")
            elif p.subject_name.startswith("P"):
                domain_mapping[p.image_name()] = "pig"
                domains.add("pig")
            elif p.subject_name.startswith("R"):
                domain_mapping[p.image_name()] = "rat"
                domains.add("rat")
            else:
                raise ValueError(f"Unknown species for path: {p}")

        return sorted(domains), domain_mapping

    def domain_name(self, image_name: str) -> str:
        """
        Returns: Domain name (e.g. P001) for the given image_name.
        """
        return self.domain_mapping[image_name]

    def domain_index(self, image_name: str) -> int:
        """
        Returns: Index of the domain (e.g. 0 for P001 if P001 is the first pig seen by this class) for the given image_name.
        """
        try:
            return self.domains.index(self.domain_name(image_name))
        except KeyError:
            return 0

    def index_to_domain(self, i: int) -> str:
        return self.domains[i]

    def index_to_color(self, i: int) -> str:
        return self.domain_colors[self.index_to_domain(i)]

    def to_json(self) -> dict:
        """
        Returns: Basic properties of the mapping for future reference.
        """
        return {"target_domain": self.target_domain, "domains": self.domains}

    @classmethod
    def from_config(cls, config: Config, target_domains: list[str] = None) -> dict[str, Self]:
        if target_domains is None:
            target_domains = config.get("input/target_domain", ["camera_index"])
        else:
            target_domains = target_domains
        shuffle_domains = config.get("input/shuffle_domains", False)

        if "domain_mappings" in config:
            mappings = config["domain_mappings"]
            type_check = [isinstance(mappings[target_domain], DomainMapper) for target_domain in target_domains]

            if all(type_check):
                return mappings

        mappings = {}

        data_spec = DataSpecification.from_config(config)
        for target_domain in target_domains:
            mappings[target_domain] = cls(data_spec, target_domain=target_domain, shuffle_domains=shuffle_domains)

        config["domain_mappings"] = mappings  # Cache for future use

        return mappings
