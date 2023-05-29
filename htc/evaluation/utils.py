# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union

import pandas as pd

from htc.models.data.DataSpecification import DataSpecification


def split_test_table(
    run_dir: Path, table_name: str = "test_table.pkl.xz", split_tables: bool = True
) -> Union[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Read a test table which contains multiple test sets. This is the default if your data specification has more than one test set but the TestPredictor only creates one resulting test table.

    The default is to get a test table per split:
    >>> from htc.settings import settings
    >>> tables = split_test_table(settings.training_dir / "image" / "2023-02-21_23-14-44_glove_baseline")
    >>> tables.keys()
    dict_keys(['test', 'test_ood'])
    >>> tables["test"]["dataset"].unique().tolist()
    ['test']

    But it is also possible to retrieve only one final test table:
    >>> df = split_test_table(settings.training_dir / "image" / "2023-02-21_23-14-44_glove_baseline", split_tables=False)
    >>> df["dataset"].unique().tolist()
    ['test', 'test_ood']

    Args:
        run_dir: Path to the training directory.
        table_name: Name of the test table to read.
        split_tables: If True, a dictionary with the test table (value) per split (key) is returned. This makes it easier if you want to aggregate the results per test table. If False, one test table is returned (which still contains the "dataset" column).

    Returns: Test table(s) with easy access to the different splits.
    """
    specs = DataSpecification(run_dir / "data.json")
    specs.activate_test_set()
    df = pd.read_pickle(run_dir / table_name).sort_values(by="image_name")

    # All test splits
    split_names = [n for n in specs.split_names() if n.startswith("test")]
    assert len(split_names) > 0, "Cannot find any test splits"

    # We need to map every path to its corresponding split
    name_to_split = {}
    for split in split_names:
        for p in specs.paths(f"^{split}$"):
            assert p.image_name() not in name_to_split, f"The path {p} is part of more than one split"
            name_to_split[p.image_name()] = split

    # Add a column to the dataframe denoting the corresponding split
    assert len(name_to_split) > 0, "Could not find any test paths"
    assert len(name_to_split) == len(specs.paths("^test")), "Could not map every path to a split"
    df["dataset"] = [name_to_split[name] for name in df["image_name"]]
    assert df["dataset"].unique().tolist() == split_names

    if split_tables:
        # Return a dataframe per split (easier for aggregating)
        tables = {split: df[df["dataset"] == split].reset_index(drop=True) for split in split_names}
        return tables
    else:
        return df
