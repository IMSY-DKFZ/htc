# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import pandas as pd
from rich.progress import track

from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.LabelMapping import LabelMapping
from htc_projects.species.apply_transforms_paths import apply_transforms_paths_median
from htc_projects.species.settings_species import settings_species
from htc_projects.species.tables import baseline_table

if __name__ == "__main__":
    # Transform images and compute the median spectra to show that the xeno-learning method works
    labels = settings_species.malperfused_labels
    mapping = LabelMapping({l: i for i, l in enumerate(labels)}, unknown_invalid=True)
    df = baseline_table(mapping)

    dfs = []
    for source_species, target_species in track(
        [("rat", "pig"), ("pig", "rat"), ("rat", "human"), ("pig", "human")], refresh_per_second=1
    ):
        # Same settings as during training
        config = Config({
            "label_mapping": mapping,
            "input/preprocessing": "L1",
            "input/n_channels": 100,
            "input/transforms_gpu": [
                {
                    "class": "htc_projects.species.species_transforms>ProjectionTransform",
                    "base_name": settings_species.species_projection[source_species],
                    "target_labels": labels,
                    "interpolation": True,
                    "p": 0.8,
                }
            ],
            "dataloader_kwargs/num_workers": 2,
            "dataloader_kwargs/batch_size": 8,
        })
        paths = DataPath.from_table(df[df.species_name == target_species])
        df_t = apply_transforms_paths_median(paths, config, epoch_size=100)
        df_t["source_species"] = source_species
        df_t["target_species"] = target_species
        dfs.append(df_t)

    df_t = pd.concat(dfs, ignore_index=True)

    target_dir = settings_species.results_dir / "projections"
    target_dir.mkdir(exist_ok=True, parents=True)
    df_t.to_feather(target_dir / "projections_clear.feather")
