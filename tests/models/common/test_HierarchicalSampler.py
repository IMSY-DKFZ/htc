# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import numpy as np
import pytest
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader

from htc.models.common.HierarchicalSampler import HierarchicalSampler
from htc.models.data.DataSpecification import DataSpecification
from htc.models.image.DatasetImage import DatasetImage
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper
from htc.utils.LabelMapping import LabelMapping


class TestHierarchicalSampler:
    @pytest.mark.parametrize(
        "batch_size, specs_name, hierarchical_sampling",
        [
            # 3 cams
            (5, "pigs_masks_loocv_4cam.json", True),
            (5, "pigs_masks_loocv_4cam.json", "label"),
            (6, "pigs_masks_loocv_4cam.json", True),
            (6, "pigs_masks_loocv_4cam.json", "label"),
            # 4 cams
            (5, "pigs_masks_fold-baseline_4cam.json", True),
            (5, "pigs_masks_fold-baseline_4cam.json", "label"),
            (8, "pigs_masks_fold-baseline_4cam.json", True),
            (8, "pigs_masks_fold-baseline_4cam.json", "label"),
            (8, "pigs_masks_fold-baseline_4cam.json", "label+oversampling"),
        ],
    )
    def test_sampling(self, batch_size: int, specs_name: str, hierarchical_sampling: bool | str) -> None:
        seed_everything(0, workers=True)

        config = Config({
            "label_mapping": "htc_projects.camera.settings_camera>label_mapping",
            "input/data_spec": specs_name,
            "input/epoch_size": 20,
            "input/no_features": True,
            "input/no_labels": type(hierarchical_sampling) == bool,
            "input/target_domain": ["camera_index"],
            "input/hierarchical_sampling": hierarchical_sampling,
            "dataloader_kwargs/batch_size": batch_size,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])
        dataset = DatasetImage(paths, train=False, config=config)

        sampler = HierarchicalSampler(dataset.paths, config)
        assert len(sampler) == config["input/epoch_size"]

        cam_mapper = DomainMapper(paths, target_domain="camera_index")
        dataloader = DataLoader(dataset, sampler=sampler, **config["dataloader_kwargs"])

        image_names = []
        for i, batch in enumerate(dataloader):
            assert len(batch["image_name"]) == config["dataloader_kwargs/batch_size"]

            # With 4 cams and batch size of 5, bootstraps of length 8 are drawn. 3 samples are randomly removed so that 1 camera may be removed completely
            assert len({cam_mapper.domain_index(x) for x in batch["image_name"]}) >= len(cam_mapper) - 1

            image_names += batch["image_name"]

            domains = [cam_mapper.domain_index(x) for x in batch["image_name"]]
            _, counts = np.unique(domains, return_counts=True)
            assert np.all(counts >= 1), "Every domain must be represented at least once"
            if batch_size == 6 or batch_size == 8:
                assert np.all(counts == 2)

            if type(hierarchical_sampling) == str and hierarchical_sampling.startswith("label"):
                domain_labels = {}
                for b, d in enumerate(domains):
                    if d not in domain_labels:
                        domain_labels[d] = set()
                    domain_labels[d].update(batch["labels"][b][batch["valid_pixels"][b]].unique().tolist())

                assert len(set.intersection(*domain_labels.values())) > 0, (
                    "All domains should have at least one label in common"
                )

                if "oversampling" in hierarchical_sampling:
                    assert sampler.sampling_options == ["oversampling"]
                else:
                    assert sampler.sampling_options == []

        assert i == len(dataloader) - 1
        assert len(set(image_names)) / len(image_names) > 0.9, "Different images should have been sampled"

        image_names2 = []
        for i, batch in enumerate(dataloader):
            assert len(batch["image_name"]) == config["dataloader_kwargs/batch_size"]
            assert len({cam_mapper.domain_index(x) for x in batch["image_name"]}) >= len(cam_mapper) - 1

            image_names2 += batch["image_name"]

        assert image_names != image_names2, "Second iteration should yield different images"

    def test_image_label(self, check_sepsis_data_accessible: Callable) -> None:
        check_sepsis_data_accessible()

        seed_everything(0, workers=True)
        config = Config({
            "input/data_spec": "sepsis-inclusion_palm_5folds_test-0.25_seed-0.json",
            "input/preprocessing": "L1_recalibrated_crop_reshape_224+224",
            "input/epoch_size": 8,
            "input/image_labels": [
                {
                    "meta_attributes": ["sepsis_status"],
                    "image_label_mapping": "htc_projects.sepsis_icu.settings_sepsis_icu>sepsis_label_mapping",
                }
            ],
            "input/no_features": True,
            "input/no_labels": True,
            "input/target_domain": ["no_domain"],
            "input/hierarchical_sampling": "image_label+oversampling",
            "dataloader_kwargs/batch_size": 4,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])
        dataset = DatasetImage(paths, train=False, config=config)

        sampler = HierarchicalSampler(dataset.paths, config)
        assert len(sampler) == config["input/epoch_size"]

        dataloader = DataLoader(dataset, sampler=sampler, **config["dataloader_kwargs"])
        batch = next(iter(dataloader))
        assert batch["image_labels"].shape == (4,)
        values, counts = batch["image_labels"].unique(return_counts=True)
        assert torch.all(values == torch.tensor([0, 1], dtype=torch.int64))
        assert torch.all(counts == 2)

    def test_wrong_label_mapping(self) -> None:
        seed_everything(0, workers=True)

        config = Config({
            "label_mapping": LabelMapping({"non_existing_label": 0}),
            "input/data_spec": "rat_semantic-only_5folds_nested-0-2_mapping-12_seed-0",
            "input/epoch_size": 20,
            "input/no_features": True,
            "input/target_domain": ["no_domain"],
            "input/hierarchical_sampling": "label",
            "dataloader_kwargs/batch_size": 8,
        })
        specs = DataSpecification.from_config(config)
        paths = specs.fold_paths(specs.fold_names()[0])

        with pytest.raises(AssertionError, match="problem with the label mapping"):
            HierarchicalSampler(paths, config)
