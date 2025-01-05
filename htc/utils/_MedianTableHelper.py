# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from htc.settings import settings
from htc.tivita.DataPath import DataPath
from htc.tivita.DatasetSettings import DatasetSettings
from htc.utils.LabelMapping import LabelMapping


class _MedianTableHelper:
    def __init__(
        self,
        dataset_name: str,
        table_name: str,
        paths: list[DataPath],
        image_names: list[str],
        label_mapping: LabelMapping,
        keep_mapped_columns: bool,
        annotation_name: str | list[str],
        additional_mappings: dict[str, LabelMapping],
        image_labels_column: list[dict[str, list[str] | LabelMapping]],
    ):
        """Internal helper class which is supposed to be used only by the `median_table()` function."""
        self.dataset_name = dataset_name
        self.table_name = table_name
        self.paths = paths
        self.image_names = image_names
        self.label_mapping = label_mapping
        self.keep_mapped_columns = keep_mapped_columns
        self.annotation_name = annotation_name
        self.additional_mappings = additional_mappings
        self.image_labels_column = image_labels_column

        self.tables = self._collect_tables()

    def __call__(self) -> pd.DataFrame:
        if self.dataset_name is not None:
            df = self._read_dataset()
        else:
            df = self._read_images()

        if self.image_labels_column is not None:
            image_labels = []
            for _, row in df.iterrows():
                row_label = []
                # There may be more than one image label to predict (e.g., sepsis_status and shock)
                for level_data in self.image_labels_column:
                    # Multiple attributes can be mapped to the same image label (e.g., sepsis_status (new sepsis study) and health_status (old sepsis study))
                    for attribute in level_data["meta_attributes"]:
                        if attribute in row and not pd.isna(row[attribute]):
                            value = row[attribute]
                            if "image_label_mapping" in level_data:
                                mapping = level_data["image_label_mapping"]
                                value = mapping.name_to_index(value)
                            else:
                                value = int(value)
                            row_label.append(value)
                            break

                assert len(row_label) >= 1, f"Could not map the row\n{row}\nto any image label"
                if len(row_label) == 1:
                    row_label = row_label[0]
                image_labels.append(row_label)

            df["image_labels"] = image_labels

        if self.label_mapping is not None and not self.keep_mapped_columns:
            df = df.drop(columns=["label_index", "label_name"])
            df = df.rename(
                columns={
                    "label_index_mapped": "label_index",
                    "label_name_mapped": "label_name",
                }
            )

        return df

    def _collect_tables(self) -> dict[tuple[str, str], dict[str, Path]]:
        tables = {}
        for path in sorted((settings.intermediates_dir_all / "tables").glob("*median_spectra*.feather")):
            parts = path.stem.split("@")
            assert 2 <= len(parts) <= 4, (
                "Invalid file format for median spectra table (it should be"
                f" dataset_name@table_name@median_spectra@annotation_name.feather with @table_name being optional): {path}"
            )
            if len(parts) == 2:
                dataset_name, table_type = path.stem.split("@")
                annotation_name = None
                table_name = ""
            elif len(parts) == 3:
                dataset_name, table_type, annotation_name = path.stem.split("@")
                table_name = ""
            elif len(parts) == 4:
                dataset_name, table_name, table_type, annotation_name = path.stem.split("@")
            else:
                raise ValueError(f"Invalid file format for median spectra table: {path}")

            assert table_type == "median_spectra", (
                f"Invalid table name for median spectra table ({table_type} instead of median_spectra, the general format"
                " should be <dataset_name>@median_spectra@<annotation_name>.feather or"
                f" <dataset_name>@<table_name>@median_spectra@<annotation_name>.feather): {path}"
            )

            table_identifier = (dataset_name, table_name)

            if table_identifier not in tables:
                tables[table_identifier] = {}
            tables[table_identifier][annotation_name] = path

        return tables

    def _read_dataset(self) -> pd.DataFrame:
        if (self.dataset_name, self.table_name) not in self.tables:
            error_message = (
                f"Could not find the table {self.dataset_name}{'@' + self.table_name if self.table_name != '' else ''} in the"
                f" registered median tables (from all available datasets):\n{self.tables.keys()}"
            )
            if "#" not in self.dataset_name:
                # If the dataset consists only of subdatasets but the main dataset is requested, collect all tables and merge them, e.g.
                # 2022_10_24_Tivita_sepsis_ICU = 2022_10_24_Tivita_sepsis_ICU#calibrations + 2022_10_24_Tivita_sepsis_ICU#subjects
                parent_tables = []
                for dataset_name, table_name in self.tables.keys():
                    if dataset_name.startswith(self.dataset_name) and self.table_name == table_name:
                        parent_tables.append(self._read_table(dataset_name, table_name, self.annotation_name))
                if len(parent_tables) > 0:
                    df = pd.concat(parent_tables, ignore_index=True)
                else:
                    raise ValueError(error_message)
            else:
                raise ValueError(error_message)
        else:
            df = self._read_table(self.dataset_name, self.table_name, self.annotation_name)
            if len(df) == 0:
                settings.log.warning(
                    f"Could not find a table for the dataset {self.dataset_name}, the table name {self.table_name} and the"
                    f" annotation name {self.annotation_name}"
                )

        # In case the dataset contains links, add them to the resulting table
        data_dir = settings.data_dirs[self.dataset_name]
        if (links_file := data_dir / "path_links.json").exists():
            with links_file.open() as f:
                link_data = json.load(f)

            self.image_names = []
            for links in link_data.values():
                self.image_names += links

            df = pd.concat([df, self._read_images()])

        return df

    def _read_images(self) -> pd.DataFrame:
        if self.paths is not None:
            assert self.image_names is None, "image_names must be None if paths is specified"
            image_names_only, annotation_images, image_names_ordering = self._parse_paths(self.paths)
        elif self.image_names is not None:
            assert self.paths is None, "paths must be None if image_names is specified"
            # Theoretically, we could also parse the image names to paths and only use the paths function
            # However, it is faster to use the image names directly if available (and we need image names anyway for the table)
            image_names_only, annotation_images, image_names_ordering = self._parse_image_names(self.image_names)
        else:
            raise ValueError("image_names or paths must be supplied if dataset_names is None")

        image_names = image_names_only + list(itertools.chain.from_iterable(annotation_images.values()))
        image_names = pd.unique(np.asarray(image_names))  # Unique without sorting
        image_names_ordering = pd.unique(np.asarray(image_names_ordering))

        # First all the images without annotation name requirements
        dfs = []
        remaining_images = set(image_names_only)
        considered_datasets = set()
        for dataset_name, table_name in self.tables.keys():
            if table_name != self.table_name:
                continue

            df = self._read_table(
                dataset_name, table_name, self.annotation_name, requested_image_names=remaining_images
            )
            if len(df) > 0:
                dfs.append(df)
                remaining_images = remaining_images - set(df["image_name"].values)
                considered_datasets.add(dataset_name)

                if len(remaining_images) == 0:
                    # We already have all image_names, we can stop looping over the tables
                    break

        # Then all images with annotation names
        if len(annotation_images) > 0:
            remaining_images = {name: set(images) for name, images in annotation_images.items()}
            is_done = False
            for dataset_name, table_name in self.tables.keys():
                if table_name != self.table_name:
                    continue
                if is_done:
                    break

                for table_annotation_name in self.tables[(dataset_name, table_name)].keys():
                    if table_annotation_name not in annotation_images.keys():
                        # If the table does not contain any of the requested annotations, we can skip it
                        continue

                    df = self._read_table(
                        dataset_name,
                        table_name,
                        table_annotation_name,
                        requested_image_names=remaining_images[table_annotation_name],
                    )
                    if len(df) > 0:
                        dfs.append(df)
                        remaining_images[table_annotation_name] = remaining_images[table_annotation_name] - set(
                            df["image_name"].values
                        )
                        considered_datasets.add(dataset_name)

                        if all(len(r) == 0 for r in remaining_images.values()):
                            is_done = True
                            # We already have all image_names, we can stop looping over the tables
                            break

        # We cannot assert that there are no remaining images anymore because some images may get excluded due to the label mapping or some images maybe don't even have annotations (so they can't be included)
        if len(dfs) == 0:
            error_message = (
                f"Could not find any of the requested images (first image: {image_names[0]}) in the tables"
                f" ({considered_datasets = }). This could mean that some of the intermediate files are missing or that"
                " you do not have access to them (e.g. human data)."
            )
            if self.label_mapping is not None:
                error_message += (
                    f" Please make also sure that the label mapping ({self.label_mapping}) is correct and does not exclude"
                    " all images."
                )
            raise ValueError(error_message)

        with warnings.catch_warnings():
            # The same columns might have different dtypes in the dataframes depending on missing values
            warnings.filterwarnings(
                "ignore", message=".*object-dtype columns with all-bool values", category=FutureWarning
            )
            df = pd.concat(dfs)
        if len(dfs) > 1 and "label_index" in df.columns:
            # label_index is potentially incorrect when paths from multiple datasets are used, so it is safer to remove it
            df.drop(columns="label_index", inplace=True)

        # Same order as defined by the paths
        df["image_name"] = df["image_name"].astype("category")
        df["image_name"] = df["image_name"].cat.set_categories(image_names_ordering)
        df.sort_values("image_name", inplace=True, ignore_index=True)

        # Make sure we have all requested image_names (it is possible that some image_names are missing if they contain only labels which were filtered out by the label mapping)
        image_names_df = set(df["image_name"].unique())
        assert image_names_df.issubset(image_names), (
            "Could not find all image_names in the median spectra tables. Please make sure that the median table exists"
            " for every dataset where the image_names come from"
        )

        if self.label_mapping is not None:
            assert set(df["label_index_mapped"].values).issubset(set(self.label_mapping.label_indices())), (
                "Found at least one label_index which is not part of the mapping"
            )
        if len(image_names_df) < len(image_names):
            settings.log.warning(
                f"{len(image_names) - len(image_names_df)} image_names are not used because they were filtered out"
                f" (e.g. by the label mapping). The following tables were considered: {considered_datasets}"
            )

        return df

    def _read_table(
        self,
        dataset_name: str,
        table_name: str,
        annotation_name: str | list[str] | None,
        requested_image_names: list[str] = None,
    ) -> pd.DataFrame:
        table_identifier = (dataset_name, table_name)

        # Find the default annotation_name
        if annotation_name is None:
            data_dir = settings.data_dirs[dataset_name]
            if data_dir is not None:
                dsettings = DatasetSettings(data_dir)
                annotation_name = dsettings.get("annotation_name_default")

        if annotation_name is None or annotation_name == "all":
            assert table_identifier in self.tables, (
                f"Could not find the table {table_identifier} in the tables\n{self.tables.keys()}"
            )
            annotation_name = list(self.tables[table_identifier].keys())

        if type(annotation_name) == str:
            annotation_name = [annotation_name]

        df = []
        for name in annotation_name:
            if name not in self.tables[table_identifier]:
                continue

            df_a = pd.read_feather(self.tables[table_identifier][name])
            if name is not None:
                df_a["annotation_name"] = name
            else:
                assert len(annotation_name) == 1
            df.append(df_a)

        if len(df) == 0:
            return pd.DataFrame()

        df = pd.concat(df)

        if requested_image_names is not None:
            # Select relevant images so that we don't change the labels if we don't need to
            df = df[df["image_name"].isin(requested_image_names)]

        if len(df) > 0:
            if self.label_mapping is not None:
                # Mapping from path to config (the mapping depends on the dataset and must be done separately)
                df = df.query("label_name in @self.label_mapping.label_names(all_names=True)").copy()
                if len(df) > 0:
                    assert settings.data_dirs[dataset_name] is not None, (
                        f"Cannot find the path to the dataset {dataset_name} but this is required for remapping the"
                        " labels"
                    )

                    original_mapping = LabelMapping.from_data_dir(settings.data_dirs[dataset_name])
                    label_indices = df["label_index"].values.astype(np.int64, copy=True)
                    self.label_mapping.map_tensor(label_indices, original_mapping)  # Operates in-place
                    df["label_index_mapped"] = label_indices
                    df["label_name_mapped"] = [self.label_mapping.index_to_name(i) for i in df["label_index_mapped"]]

            for name, mapping in self.additional_mappings.items():
                df[f"{name}_index"] = [mapping.name_to_index(x) for x in df[name]]

        return df.reset_index(drop=True)

    @staticmethod
    def _parse_paths(paths: list[DataPath]) -> tuple[list[str], dict[str, list[str]], list[str]]:
        image_names_ordering = []
        image_names_only = []
        annotation_images = {}
        for p in paths:
            image_names_ordering.append(p.image_name())
            names = p.annotation_names()

            if len(names) > 0:
                for a in names:
                    if a not in annotation_images:
                        annotation_images[a] = []
                    annotation_images[a].append(p.image_name())
            else:
                image_names_only.append(p.image_name())

        return image_names_only, annotation_images, image_names_ordering

    @staticmethod
    def _parse_image_names(names: list[str]) -> tuple[list[str], dict[str, list[str]], list[str]]:
        image_names_ordering = []
        image_names_only = []
        annotation_images = {}
        for name in names:
            if "@" in name:
                image_name, annotation_names = name.split("@")
                annotation_names = annotation_names.split("&")
                for a in annotation_names:
                    if a not in annotation_images:
                        annotation_images[a] = []
                    annotation_images[a].append(image_name)
                image_names_ordering.append(image_name)
            else:
                image_names_only.append(name)
                image_names_ordering.append(name)

        return image_names_only, annotation_images, image_names_ordering
