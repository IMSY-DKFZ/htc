# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from htc.data_processing.DatasetIteration import DatasetIteration
from htc.models.image.DatasetImage import DatasetImage
from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.helper_functions import sort_labels
from htc.utils.LabelMapping import LabelMapping
from htc.utils.paths import ParserPreprocessing, filter_min_labels


class MedianSpectra(DatasetIteration):
    def __init__(self, paths: list[DataPath], dataset_name: str, output_dir: Path = None):
        # It does not make any sense to load images where we don't have any labels as we then also can't compute any median spectra
        paths = [p for p in paths if filter_min_labels(p)]
        assert len(paths) > 0, "No paths left for median spectra computation"
        super().__init__(paths)
        self.dataset_name = dataset_name

        if output_dir is None:
            self.output_dir = settings.intermediates_dir_all / "tables"
        else:
            self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)

        config = Config(
            {
                "input/n_channels": 100,
                "label_mapping": None,
                "input/annotation_name": "all",
            }
        )
        self.dataset = DatasetImage(paths, train=False, config=config)

    def compute(self, i: int) -> list[dict]:
        sample = self.dataset[i]
        path = self.dataset.paths[i]
        mapping = LabelMapping.from_path(path)
        annotation_meta = path.read_annotation_meta()
        if annotation_meta is None:
            annotation_meta = {}

        features_normalized = sample["features"] / torch.linalg.norm(sample["features"], ord=1, dim=-1, keepdim=True)
        features_normalized.nan_to_num_()

        cube = sample["features"].numpy()
        sto2_img = path.compute_sto2(cube)
        nir_img = path.compute_nir(cube)
        twi_img = path.compute_twi(cube)
        ohi_img = path.compute_ohi(cube)
        thi_img = path.compute_thi(cube)
        tli_img = path.compute_tli(cube)

        rows = []
        for label_key in sample.keys():
            if not label_key.startswith("labels"):
                continue

            for label_index, counts in zip(*sample[label_key].unique(return_counts=True)):
                label_index = label_index.item()

                # We skip unlabeled pixels as median spectra do not make sense in this case (consist of multiple labels)
                if mapping.is_index_valid(label_index):
                    label_name = mapping.index_to_name(label_index)

                    if sample[label_key].ndim == 3:
                        selection = sample[label_key] == label_index
                        valid_dim = selection.any(dim=-1).any(dim=-1)
                        assert (
                            valid_dim.sum() == 1
                        ), "For multi-layer segmentations, every label must only occur on exactly one layer"
                        selection = selection[valid_dim].squeeze(dim=0)
                    else:
                        selection = sample[label_key] == label_index

                    spectra = sample["features"][selection]
                    spectra_normalized = features_normalized[selection]

                    selected_sto2 = sto2_img[selection]
                    selected_nir = nir_img[selection]
                    selected_twi = twi_img[selection]
                    selected_ohi = ohi_img[selection]
                    selected_thi = thi_img[selection]
                    selected_tli = tli_img[selection]

                    current_row = {"image_name": path.image_name()}
                    current_row |= path.image_name_typed()

                    current_row |= {
                        "label_index": label_index,
                        "label_name": label_name,
                        "median_spectrum": spectra.quantile(q=0.5, dim=0).numpy(),  # Same as np.median
                        "std_spectrum": spectra.std(dim=0).numpy(),
                        "median_normalized_spectrum": spectra_normalized.quantile(q=0.5, dim=0).numpy(),
                        "std_normalized_spectrum": spectra_normalized.std(dim=0).numpy(),
                        "n_pixels": counts.item(),
                        "median_sto2": np.median(selected_sto2.data),
                        "std_sto2": np.std(selected_sto2.data),
                        "median_nir": np.median(selected_nir.data),
                        "std_nir": np.std(selected_nir.data),
                        "median_twi": np.median(selected_twi.data),
                        "std_twi": np.std(selected_twi.data),
                        "median_ohi": np.median(selected_ohi.data),
                        "std_ohi": np.std(selected_ohi.data),
                        "median_thi": np.median(selected_thi.data),
                        "std_thi": np.std(selected_thi.data),
                        "median_tli": np.median(selected_tli.data),
                        "std_tli": np.std(selected_tli.data),
                    }

                    if label_key != "labels":
                        current_row["annotation_name"] = label_key.removeprefix("labels_")

                    # Already add additional meta labels if we have it
                    for k, v in annotation_meta.items():
                        if k == "label_meta":
                            if label_name in v:
                                # Add all the meta information for the current label
                                for label_keys, label_values in v[label_name].items():
                                    current_row[label_keys] = label_values
                        else:
                            # Add all the meta information of the current image
                            current_row[k] = v

                    rows.append(current_row)

        return rows

    def finished(self, results: list[dict]) -> None:
        rows = list(itertools.chain.from_iterable(results))
        df_median = pd.DataFrame(rows)

        # Directly add metadata information to the table as this is often useful to have
        # There is only one meta table for the complete dataset (and not for the subdatasets), so we need to remove the subdataset information for this step
        meta_table_name = re.sub("#.*$", "", self.dataset_name) + "@meta.feather"
        df_meta = pd.read_feather(self.output_dir / meta_table_name)
        df_meta.drop(columns=["annotation_name"], inplace=True)
        df = df_median.merge(df_meta, how="left")

        if "annotation_name" in df:
            # We split the median spectra table by annotators for two reasons:
            # - performance: large tables are slower to read than smaller ones and most of the time we are only interested in the primary annotations anyway
            # - avoid errors: having annotators in the table changes what a row means from (image, label) to (image, label, annotation). This can easily lead to errors if someone reads the table directly and e.g. averages the median spectra by label having accidentally averaged the spectra from all annotators
            # The interface to the median spectra table does not change. Users can still use the median_spectra() function, also to get a table with all annotators when requested (not by default)
            for name in df["annotation_name"].unique():
                df_name = df[df["annotation_name"] == name]
                df_name = df_name.drop(columns="annotation_name")
                df_name = sort_labels(df_name)
                df_name = df_name.reset_index(drop=True)

                target_file = self.output_dir / f"{self.dataset_name}@median_spectra@{name}.feather"
                df_name.to_feather(target_file)
                settings.log.info(f"Wrote median table to {target_file}")
        else:
            # Already sort the table by label name so that the default order is automatically used in plots
            df = sort_labels(df)

            target_file = self.output_dir / f"{self.dataset_name}@median_spectra.feather"
            df.to_feather(target_file)
            settings.log.info(f"Wrote median table to {target_file}")


if __name__ == "__main__":
    prep = ParserPreprocessing(description="Calculate median spectra per pig, organ and image")
    paths = prep.get_paths()
    assert prep.args.spec is None, (
        "Median spectra can only be calculated per dataset and not for a specification file. Otherwise, the label_index"
        " would not be correct (the same label_index can refer to different organs across datasets)."
    )

    settings.log.info(f"Computing median spectra for {len(paths)} paths...")
    MedianSpectra(paths, prep.args.dataset_name, output_dir=prep.args.output_path).run()
