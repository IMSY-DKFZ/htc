# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.paths import filter_semantic_labels_only


def test_filter_semantic_labels_only() -> None:
    paths_all = list(DataPath.iterate(settings.data_dirs.semantic))
    paths_filter = list(DataPath.iterate(settings.data_dirs.semantic, filters=[filter_semantic_labels_only]))

    assert len(paths_all) == len(paths_filter), (
        "The filter_semantic_labels_only should not have an effect on the semantic dataset"
    )
