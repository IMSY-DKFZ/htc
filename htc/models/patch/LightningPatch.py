# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch
from torch.utils.data.dataloader import DataLoader

from htc.models.common.HierarchicalSampler import HierarchicalSampler
from htc.models.common.HTCDataset import HTCDataset
from htc.models.common.HTCLightning import HTCLightning
from htc.models.image.LightningImage import LightningImage
from htc.models.patch.DatasetPatchImage import DatasetPatchImage
from htc.models.patch.DatasetPatchStream import DatasetPatchStream


class LightningPatch(LightningImage):
    @staticmethod
    def dataset(**kwargs) -> HTCDataset:
        if kwargs["train"]:
            if kwargs["config"]["input/hierarchical_sampling"]:
                kwargs["sampler"] = HierarchicalSampler(
                    kwargs["paths"], kwargs["config"], batch_size=kwargs["config"]["dataloader_kwargs/num_workers"]
                )

            return DatasetPatchStream(**kwargs)
        else:
            return DatasetPatchImage(**kwargs)

    def val_dataloader(self, **kwargs) -> list[DataLoader]:
        # We only evaluate one image at a time (safer)
        return HTCLightning.val_dataloader(self, batch_size=1, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        # We only evaluate one image at a time (safer)
        return HTCLightning.test_dataloader(self, batch_size=1, **kwargs)

    def _predict_images(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # DatasetPatchImage stores the number of patches in the image in the batch dimension so we don't need the real batch dimension
        batch["features"] = batch["features"].squeeze(dim=0)

        predictions = self(batch).permute(0, 2, 3, 1)  # [N, C, H, W] --> [N, H, W, C]
        image_predictions = self.datasets_val[0].reshape_img(predictions, batch)
        image_predictions = image_predictions.permute(2, 0, 1).unsqueeze(dim=0)  # [N, C, H, W]

        return {"class": image_predictions}
