# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import re
import warnings
from pathlib import Path

import pytest
import torch
from lightning import seed_everything
from pytest import LogCaptureFixture, MonkeyPatch
from torch.utils.data import DataLoader

from htc.models.common.torch_helpers import copy_sample, move_batch_gpu
from htc.models.common.transforms import HTCTransformation
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.settings_seg import settings_seg
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.file_transfer import list_files_s3
from htc.utils.LabelMapping import LabelMapping
from htc_projects.species.species_transforms import ProjectionTransform


@pytest.mark.serial
class TestProjectionTransform:
    @pytest.mark.parametrize(
        "p,label_mode", [(0, "random"), (1, "random"), (1, "label_match"), (1, "label_match_extended")]
    )
    def test_basics(self, p: float, label_mode: str) -> None:
        seed_everything(0, workers=True)

        path_kidney = DataPath.from_image_name("P058#2020_05_13_19_05_34")
        path_kidney2 = DataPath.from_image_name("P062#2020_05_15_20_00_19")
        path_other = DataPath.from_image_name("P059#2020_05_14_13_00_11")
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/preprocessing": "L1",
            "input/features_dtype": "float32",  # Avoid type mismatches in this test
            "input/n_channels": 100,
            "input/transforms_gpu": [
                {
                    "class": "htc_projects.species.species_transforms>ProjectionTransform",
                    "base_name": "weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic",
                    "label_mode": label_mode,
                    "target_labels": ["kidney"] if label_mode == "label_match" else None,
                    "p": p,
                }
            ],
        })

        dataset = DatasetImage([path_kidney, path_kidney2, path_other], train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=3)
        batch = next(iter(dataloader))
        batch = move_batch_gpu(batch)
        batch_copy = copy_sample(batch)

        seed_everything(0, workers=True)  # Reproducible conditions for the label selection in the transform
        aug = HTCTransformation.parse_transforms(
            config["input/transforms_gpu"], config=config, device=batch["features"].device
        )
        batch = HTCTransformation.apply_valid_transforms(batch, aug)

        if label_mode == "label_match":
            assert len(aug[-1].matrices) == len(aug[-1].biases) == 1

        assert torch.all(batch["labels"] == batch_copy["labels"])
        assert batch["features"].dtype == torch.float32
        if label_mode == "label_match" or p == 0:
            assert torch.allclose(batch["features"][2], batch_copy["features"][2])
        else:
            assert not torch.allclose(batch["features"][2], batch_copy["features"][2])

        if p == 0:
            assert torch.allclose(batch["features"][0], batch_copy["features"][0])
            assert torch.allclose(batch["features"][1], batch_copy["features"][1])
        else:
            if label_mode == "label_match":
                kidney_features = lambda batch, b: batch["features"][b][
                    batch["labels"][b] == settings_seg.label_mapping.name_to_index("kidney")
                ]
                non_kidney_features = lambda batch, b: batch["features"][b][
                    batch["labels"][b] != settings_seg.label_mapping.name_to_index("kidney")
                ]
                assert not torch.allclose(kidney_features(batch, 0), kidney_features(batch_copy, 0))
                assert not torch.allclose(kidney_features(batch, 1), kidney_features(batch_copy, 1))
                assert torch.allclose(non_kidney_features(batch, 0), non_kidney_features(batch_copy, 0))
                assert torch.allclose(non_kidney_features(batch, 1), non_kidney_features(batch_copy, 1))
            else:
                # Every spectra should change
                assert not torch.isclose(batch["features"][0], batch_copy["features"][0]).all(dim=-1).any()
                assert not torch.isclose(batch["features"][1], batch_copy["features"][1]).all(dim=-1).any()

        if p == 1:
            seed_everything(0, workers=True)
            config["input/transforms_gpu"][0]["interpolate"] = True
            aug = HTCTransformation.parse_transforms(
                config["input/transforms_gpu"], config=config, device=batch["features"].device
            )
            batch_interp = copy_sample(batch_copy)
            batch_interp = HTCTransformation.apply_valid_transforms(batch_interp, aug)

            assert not torch.allclose(batch_interp["features"][0], batch_copy["features"][0])

            diff_full = torch.abs(batch_copy["features"][0] - batch["features"][0]).sum()
            diff_interp = torch.abs(batch_copy["features"][0] - batch_interp["features"][0]).sum()
            assert diff_interp < diff_full

    def test_public_table(self) -> None:
        doc = ProjectionTransform.__init__.__doc__

        # Check whether the table in the docstring is up-to-date
        match = re.search(r"\s*(\| weights\+bias.*\|)\s+All projection matrices", doc, flags=re.DOTALL)
        assert match is not None
        doc_table = match.group(1)

        names_doc = []
        for line in doc_table.splitlines():
            names_doc.append(line.split("|")[1].strip())

        try:
            files_s3 = list_files_s3("projection_matrices")
        except Exception as e:
            files_s3 = None
            warnings.warn(
                f"Cannot retrieve the file list from the s3 storage (this can happen if the s3 storage is down):\n{e}",
                stacklevel=2,
            )

        if files_s3 is not None:
            names_remote = [Path(f).stem for f in files_s3]
            assert set(names_doc) == set(names_remote) == set(ProjectionTransform.known_projection_matrices.keys())

    def test_download(self, tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture) -> None:
        monkeypatch.setattr(type(settings), "results_dir", tmp_path / "nonexistent")
        monkeypatch.setenv("TORCH_HOME", str(tmp_path))

        mapping = LabelMapping({"colon": 0, "liver": 1})
        ProjectionTransform(
            base_name="weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029",
            device="cpu",
            config=Config({"label_mapping": mapping, "input/n_channels": 100}),
        )
        assert len(caplog.records) == 1 and "Successfully downloaded" in caplog.records[0].msg

        ProjectionTransform(
            base_name="weights+bias_malperfusion_rat_subjects=R017,R019,R025,R029",
            device="cpu",
            config=Config({"label_mapping": mapping, "input/n_channels": 100}),
        )
        assert len(caplog.records) == 1


@pytest.mark.serial
class TestProjectionTransformMultiple:
    def test_multiple(self) -> None:
        seed_everything(0, workers=True)

        path_kidney = DataPath.from_image_name("P058#2020_05_13_19_05_34")
        path_kidney2 = DataPath.from_image_name("P062#2020_05_15_20_00_19")
        config = Config({
            "label_mapping": settings_seg.label_mapping,
            "input/preprocessing": "L1",
            "input/features_dtype": "float32",  # Avoid type mismatches in this test
            "input/n_channels": 100,
            "input/transforms_gpu": [
                {
                    "class": "htc_projects.species.species_transforms>ProjectionTransformMultiple",
                    "projections": [
                        {
                            "base_name": "weights+bias_malperfusion_pig_kidney=P091,P095,P097,P098+aortic",
                            "target_labels": ["kidney"],
                            "p": 1,
                        },
                        {
                            "base_name": "weights+bias_ICG_pig_subjects=P062,P072,P076,P113",
                            "target_labels": ["kidney"],
                            "p": 1,
                        },
                    ],
                    "p": 1,
                }
            ],
        })

        paths = [path_kidney, path_kidney2]
        dataset = DatasetImage(paths, train=True, config=config)
        dataloader = DataLoader(dataset, batch_size=3)
        batch = next(iter(dataloader))
        batch = move_batch_gpu(batch)
        batch_copy = copy_sample(batch)

        aug = HTCTransformation.parse_transforms(
            config["input/transforms_gpu"], config=config, device=batch["features"].device
        )
        batch = HTCTransformation.apply_valid_transforms(batch, aug)

        # The transformation should change every kidney spectra
        kidney_features = lambda batch: batch["features"][
            batch["labels"] == settings_seg.label_mapping.name_to_index("kidney")
        ]
        non_kidney_features = lambda batch: batch["features"][
            batch["labels"] != settings_seg.label_mapping.name_to_index("kidney")
        ]
        assert not torch.allclose(kidney_features(batch), kidney_features(batch_copy))
        assert torch.allclose(non_kidney_features(batch), non_kidney_features(batch_copy))
