# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc_projects.sepsis_icu.models.DatasetMeta import DatasetMeta


class TestDatasetMeta:
    def test_basics(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        path = DataPath.from_image_name("S438#2023_10_02_19_23_03")
        config = Config({
            "task": "classification",
            "input/image_labels": [
                {
                    "meta_attributes": ["sepsis_status"],
                    "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>sepsis_label_mapping",
                }
            ],
            "input/meta/attributes": [{"name": "age"}, {"name": "sex", "mapping": {"male": 0, "female": 1}}],
        })
        sample = DatasetMeta([path], train=False, config=config)[0]
        assert sample["image_name"] == "S438#2023_10_02_19_23_03"
        assert sample["image_labels"].item() == 0

        meta = sample["meta"].tolist()
        assert len(meta) == 2
        assert meta[0] == path.meta("age")
        assert meta[1] == config["input/meta/attributes"][1]["mapping"][path.meta("sex")]
