# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import json

from htc.tivita.DataPathMultiorgan import DataPathMultiorgan


class DataPathMultiorganNamed(DataPathMultiorgan):
    def __init__(self, *args, **kwargs):
        """
        This class expects the same data structure as DataPathMultiorgan, but it allows to specify an alternative image name via the metadata.
        """
        super().__init__(*args, **kwargs)
        self._image_name = None
        self._image_name_parts = None

    def image_name(self) -> str:
        if self._image_name is None:
            self._load_image_name()
        return self._image_name

    def image_name_parts(self) -> list[str]:
        if self._image_name_parts is None:
            self._load_image_name()
        return self._image_name_parts

    def _load_image_name(self) -> None:
        with self.annotation_meta_path().open() as f:
            meta_labels = json.load(f)

        self._image_name = meta_labels["image_name"]
        self._image_name_parts = meta_labels["image_name_parts"]
