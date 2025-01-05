# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import inspect
import re
from collections.abc import Callable

import pytest
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.sampler import RandomSampler

from htc.models.common.HierarchicalSampler import HierarchicalSampler
from htc.models.common.StreamDataLoader import StreamDataLoader
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.models.image.DatasetImageBatch import DatasetImageBatch
from htc.models.image.DatasetImageStream import DatasetImageStream
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config


@pytest.mark.serial
class TestSharedMemoryDatasetMixin:
    @pytest.mark.parametrize("SamplerClass, adjusted_epoch_size", [(HierarchicalSampler, 100), (RandomSampler, 99)])
    def test_sampler(self, SamplerClass: type[Sampler], adjusted_epoch_size: int) -> None:
        config = Config({
            "input/no_labels": True,
            "input/no_features": True,
            "input/data_spec": "pigs_masks_loocv_4cam.json",
            "input/n_channels": 3,
            "input/epoch_size": 99,
            "input/target_domain": ["camera_index"],
            "dataloader_kwargs/batch_size": 10,
            "dataloader_kwargs/num_workers": 2,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])
        if "config" in inspect.signature(SamplerClass.__init__).parameters:
            sampler = SamplerClass(paths, config)
        else:
            sampler = SamplerClass(paths, replacement=True, num_samples=adjusted_epoch_size)

        seed_everything(settings.default_seed, workers=True)
        list(sampler)  # Called twice per default
        path_indices = sorted(sampler)

        def sample_from_dataset(DatasetClass: type, single_mode: bool) -> tuple[list[int], list[int]]:
            seed_everything(settings.default_seed, workers=True)

            dataset = DatasetClass(paths, train=False, config=config, sampler=sampler)
            dataloader = StreamDataLoader(dataset, config)
            assert dataloader.single_mode == single_mode
            assert len(dataloader) == 10
            assert config["input/epoch_size"] == adjusted_epoch_size == len(sampler)

            # Same indices as from the sampler
            sampled_indices = []
            for batch in dataloader:
                sampled_indices += batch["image_index"].tolist()
            sampled_indices = sorted(sampled_indices)
            assert path_indices == sampled_indices
            assert len(sampled_indices) == config["input/epoch_size"]

            # Different indices in the next epoch
            sampled_indices2 = []
            for batch in dataloader:
                sampled_indices2 += batch["image_index"].tolist()
            sampled_indices2 = sorted(sampled_indices2)
            assert sampled_indices2 != path_indices
            assert len(sampled_indices2) == config["input/epoch_size"]

            return sampled_indices, sampled_indices2

        indices_batch1, indices_batch2 = sample_from_dataset(DatasetImageBatch, True)
        indices_stream1, indices_stream2 = sample_from_dataset(DatasetImageStream, False)

        assert indices_batch1 == indices_stream1
        assert indices_batch2 == indices_stream2

    @pytest.mark.parametrize(
        "batch_size, num_workers, n_paths, last_batch_size, n_batches",
        [(2, 2, 10, 2, 5), (3, 1, 10, 1, 4), (3, 3, 2, 2, 1), (3, 1, 2, 2, 1), (2, 2, 3, 1, 2)],
    )
    def test_sampler_list(
        self, batch_size: int, num_workers: int, n_paths: int, last_batch_size: int, n_batches: int
    ) -> None:
        config = Config({
            "input/data_spec": "pigs_masks_loocv_4cam.json",
            "input/n_channels": 3,
            "dataloader_kwargs/batch_size": batch_size,
            "dataloader_kwargs/num_workers": num_workers,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])[:n_paths]
        config["input/epoch_size"] = len(paths)
        path_indices = sorted(range(len(paths)))

        def sample_from_dataset(DatasetClass: type, single_mode: bool) -> tuple[list[int], list[torch.Tensor], int]:
            dataloader = DatasetClass.batched_iteration(paths, config)
            assert dataloader.single_mode == single_mode
            assert len(dataloader) == n_batches

            # Same indices as defined in the list
            sampled_indices = []
            for batch in dataloader:
                sampled_indices += batch["image_index"].tolist()
            assert len(batch["image_index"]) == last_batch_size
            sampled_indices = sorted(sampled_indices)
            assert path_indices == sampled_indices

            # A list stays constant across epochs
            for _ in range(3):
                sampled_indices = []
                features = []
                for batch in dataloader:
                    sampled_indices += batch["image_index"].tolist()
                    features.append(batch["features"].cpu())
                assert len(batch["image_index"]) == last_batch_size
                sampled_indices, sort_indices = torch.sort(torch.tensor(sampled_indices))
                sampled_indices = sampled_indices.tolist()
                assert path_indices == sampled_indices

            features = torch.cat(features)[sort_indices]
            return sampled_indices, features, len(dataloader)

        indices_batch, features_batch, dlen_batch = sample_from_dataset(DatasetImageBatch, True)
        indices_stream, features_stream, dlen_stream = sample_from_dataset(DatasetImageStream, False)

        assert indices_batch == indices_stream
        assert torch.allclose(features_batch, features_stream, rtol=1e-03, atol=1e-05)
        assert dlen_batch == dlen_stream

        assert config["input/epoch_size"] == len(paths), (
            "The epoch size should not be changed if we have an explicit sampler"
        )

        # A non-stream-based dataloader must yield the same images
        dataset_classic = DatasetImage(paths, train=False, config=config)
        dataloader_classic = DataLoader(dataset_classic, sampler=path_indices, **config["dataloader_kwargs"])
        assert len(dataloader_classic) == dlen_batch

        classic_indices = []
        classic_features = []
        for batch in dataloader_classic:
            classic_indices += batch["image_index"].tolist()
            classic_features.append(batch["features"])
        assert len(batch["image_index"]) == last_batch_size
        classic_indices, sort_indices = torch.sort(torch.tensor(classic_indices))
        classic_indices = classic_indices.tolist()
        assert classic_indices == path_indices

        classic_features = torch.cat(classic_features)[sort_indices]
        assert torch.allclose(classic_features, features_batch)

    @pytest.mark.parametrize("preprocessing", ["raw16", "L1"])
    def test_features_dtype(self, preprocessing: str, capfd: pytest.CaptureFixture) -> None:
        paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        config = Config({
            "input/n_channels": 100,
            "input/preprocessing": preprocessing,
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        sample = next(iter(dataloader))

        normalization = 1 if preprocessing == "L1" else None
        features = torch.from_numpy(paths[0].read_cube(normalization=normalization))
        features = features.half().cuda().unsqueeze(dim=0)

        assert torch.allclose(sample["features"], features)

        # Check that a warning is emitted if we do something wrong
        config["input/features_dtype"] = "float32"
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        sample = next(iter(dataloader))

        # We need to use capfd here because the warning is emitted in a subprocess
        assert (
            re.search(
                r"WARNING.*The dtype of the loaded data.*\(float16\) does not match", capfd.readouterr().out, re.DOTALL
            )
            is not None
        )

    def test_rgb_reconstructed(self) -> None:
        paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        config = Config({
            "input/n_channels": 3,
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        sample = next(iter(dataloader))

        features = torch.from_numpy(paths[0].read_rgb_reconstructed() / 255)
        features = features.to(dtype=torch.float32, device="cuda").unsqueeze(dim=0)
        assert torch.allclose(sample["features"], features)

    def test_rgb_sensor_aligned(self, check_sepsis_ICU_data_accessible: Callable) -> None:
        check_sepsis_ICU_data_accessible()

        paths = [DataPath.from_image_name("S438#2023_10_02_19_23_03")]
        config = Config({
            "input/n_channels": 3,
            "input/preprocessing": "rgb_sensor_aligned",
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        sample = next(iter(dataloader))

        features = torch.from_numpy(paths[0].align_rgb_sensor().data) / 255
        features = features.cuda().unsqueeze(dim=0)
        assert torch.allclose(sample["features"], features)

    def test_parameter_images(self) -> None:
        paths = [DataPath.from_image_name("P043#2019_12_20_12_38_35")]
        config = Config({
            "input/n_channels": 2,
            "input/preprocessing": "parameter_images",
            "input/parameter_names": ["StO2", "TWI"],
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })
        dataloader = DatasetImageBatch.batched_iteration(paths, config)
        sample = next(iter(dataloader))

        features = torch.stack([
            torch.from_numpy(paths[0].compute_sto2()),
            torch.from_numpy(paths[0].compute_twi()),
        ])
        features = features.cuda().permute(1, 2, 0).unsqueeze(dim=0)
        assert torch.allclose(sample["features"], features)

    def test_image_labels(self, check_sepsis_ICU_data_accessible: Callable) -> None:
        check_sepsis_ICU_data_accessible()

        seed_everything(settings.default_seed, workers=True)
        path = DataPath.from_image_name("S438#2023_10_02_19_23_03")  # No sepsis
        config = Config({
            "task": "classification",
            "input/n_channels": 100,
            "input/no_labels": True,
            "input/image_labels": [
                {
                    "meta_attributes": ["sepsis_status"],
                    "image_label_mapping": {
                        "sepsis": 10,
                        "no_sepsis": 20,
                    },
                }
            ],
            "dataloader_kwargs/num_workers": 1,
            "dataloader_kwargs/batch_size": 1,
        })

        dataloader = DatasetImageBatch.batched_iteration([path], config)
        sample = next(iter(dataloader))
        assert sample["image_labels"] == 20
        assert sample["image_labels"].dtype == torch.int64
        assert sample["image_labels"].is_cuda
