# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
import pytest
from lightning import seed_everything

from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.data.DataSpecification import DataSpecification
from htc.models.pixel.DatasetPixelStream import DatasetPixelStream
from htc.settings import settings
from htc.utils.Config import Config


@pytest.mark.serial
class TestHTCDatasetStream:
    def test_data_sampling(self, check_human_data_accessible: Callable) -> None:
        check_human_data_accessible()
        seed_everything(settings.default_seed, workers=True)

        # defining the data sampling config
        config = Config({
            "label_mapping": "htc_projects.species.settings_species>label_mapping",
            "input/epoch_size": "50 images",
            "input/target_domain": ["subject_name"],
            "input/n_channels": 3,
            "dataloader_kwargs/batch_size": 15000,
            "dataloader_kwargs/num_workers": 2,
            "input/data_spec": "human_semantic-only+pig-p+rat-p_physiological-kidney_5folds_nested-0-2_mapping-12_seed-0",
            "input/dataset_sampling": {
                "2023_12_07_Tivita_multiorgan_rat": 1,
                "2021_02_05_Tivita_multiorgan_semantic": 1,
                "2021_07_26_Tivita_multiorgan_human": 3,
            },
        })

        # setting the number of batches to be collected from the data loader
        n_batches = 200
        dataset_paths_sampled = {k: set() for k in config["input/dataset_sampling"].keys()}

        data_specs = DataSpecification.from_config(config)

        dataset = DatasetPixelStream(data_specs.paths(), train=False, config=config)
        dataloader = StreamDataLoader(dataset, config)

        # normalize the data sampling values
        config["input/dataset_sampling"] = {
            k: v / np.sum(list(config["input/dataset_sampling"].values()))
            for k, v in config["input/dataset_sampling"].items()
        }

        # iterate through the dataloader and add the paths to the sets
        d_iter = iter(dataloader)
        for i in range(n_batches):
            batch = next(d_iter)
            for x in np.unique(batch["image_index"].cpu().numpy()):
                dataset_paths_sampled[dataset.paths[x].dataset_settings["dataset_name"]].add(x)

        # normalized the number of paths collected for each dataset
        dataset_paths_sampled = {
            k: len(v) / np.sum([len(v) for v in dataset_paths_sampled.values()])
            for k, v in dataset_paths_sampled.items()
        }

        # check if the sampled ratio falls in the ballpark (+- 20%) of the given ratio in configs
        for k, v in config["input/dataset_sampling"].items():
            assert (v + 0.2) > dataset_paths_sampled[k] > (v - 0.2), (
                "The ratios in the datasets are not close to the ratios specified in the configs i.e. within +- 0.2 of"
                " the ratios in the configs"
            )
