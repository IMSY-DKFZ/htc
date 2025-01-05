# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
from lightning import seed_everything

from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.hyper_diva.DatasetPixelStreamAverage import DatasetPixelStreamAverage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper


class TestDatasetPixelStreamAverage:
    def test_no_averaging(self) -> None:
        seed_everything(settings.default_seed, workers=True)

        config = Config({
            "input/epoch_size": "5 images",
            "input/averaging": 1,
            "input/target_domain": ["subject_index"],
            "input/data_spec": "data/pigs_semantic-only_5foldsV2.json",
            "input/n_channels": 100,
            "dataloader_kwargs/num_workers": 2,
            "dataloader_kwargs/batch_size": 4,
        })
        domain_mapper = DomainMapper.from_config(config)["subject_index"]

        paths = [
            DataPath.from_image_name("P043#2019_12_20_12_38_35"),
            DataPath.from_image_name("P059#2020_05_14_12_50_10"),
        ]
        dataset = DatasetPixelStreamAverage(paths, train=True, config=config)
        dataloader = StreamDataLoader(dataset, config)

        sample = next(iter(dataloader))
        assert (
            len([
                dataset.paths[i]
                for i in sample["image_index"]
                if dataset.paths[i].image_name() == "P043#2019_12_20_12_38_35"
            ])
            == 2
        )
        assert (
            len([
                dataset.paths[i]
                for i in sample["image_index"]
                if dataset.paths[i].image_name() == "P059#2020_05_14_12_50_10"
            ])
            == 2
        )

        d1 = domain_mapper.domain_index("P043#2019_12_20_12_38_35")
        d2 = domain_mapper.domain_index("P059#2020_05_14_12_50_10")
        assert len(sample["subject_index"].unique()) == 2
        assert torch.all(
            sample["subject_index"].cpu().sort()[0] == torch.tensor([d1, d1, d2, d2], dtype=torch.int64).sort()[0]
        )
