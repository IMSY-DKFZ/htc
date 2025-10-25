# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import torch

from htc.models.common.HTCDataset import HTCDataset


class DatasetMeta(HTCDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.image_labels = torch.stack([self.read_image_labels(path) for path in self.paths])
        self.meta = torch.stack([self.read_meta(path) for path in self.paths])

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = {
            "meta": self.meta[index, :],
            "image_labels": self.image_labels[index],
        }

        if not self.train:
            paths = self.paths[index]
            if isinstance(paths, list):
                sample["image_name"] = [p.image_name() for p in paths]
                sample["image_name_annotations"] = [p.image_name_annotations() for p in paths]
            else:
                sample["image_name"] = paths.image_name()
                sample["image_name_annotations"] = paths.image_name_annotations()
            sample["image_index"] = index

        return sample
