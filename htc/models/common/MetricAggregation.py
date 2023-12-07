# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from htc.models.data.DataSpecification import DataSpecification
from htc.tivita.DataPath import DataPath
from htc.utils.Config import Config
from htc.utils.DomainMapper import DomainMapper
from htc.utils.helper_functions import sort_labels
from htc.utils.LabelMapping import LabelMapping


class MetricAggregation:
    def __init__(self, path_or_df: Union[Path, pd.DataFrame], config: Config = None, metrics: list[str] = None):
        """
        Class for calculating metrics for validation while respecting the hierarchical structure of the data. The metrics include dice metric for checkpointing and domain accuracy for domain adaptation problems.

        Args:
            path_or_df: The path or dataframe of the validation/test table.
            config: The Config class object for configs of current run.
            metrics: List containing the column names of metric columns.
        """
        self.config = config if config is not None else Config({})
        if metrics is None:
            metrics = [self.config.get("validation/checkpoint_metric", "dice_metric")]
        self.metrics = metrics

        if isinstance(path_or_df, pd.DataFrame):
            self.df = path_or_df
        elif isinstance(path_or_df, Path):
            self.df = pd.read_pickle(path_or_df)
        else:
            raise ValueError("Neither a dataframe nor path given")

        assert all(m in self.df for m in self.metrics), f"Not all metrics are in the dataframe ({self.df.columns})"

        if "subject_name" not in self.df and "image_name" in self.df:
            type_infos = []
            for name in self.df["image_name"]:
                if DataPath.image_name_exists(name):
                    type_infos.append(DataPath.from_image_name(name).image_name_typed())
                else:
                    # If the path is not available (e.g. because the user has no access to the dataset), then assume the first part is the image name
                    subject_name = name.split("#")[0]
                    assert subject_name != "ref", f"Cannot infer subject name from references: {name}"
                    type_infos.append({"subject_name": subject_name})

            # Reconstruct missing information
            df_meta = pd.DataFrame(type_infos)
            self.df = pd.concat([self.df.reset_index(drop=True), df_meta], axis=1)
            assert len(df_meta) == len(self.df), "The length of the dataframe should not change"

        assert "subject_name" in self.df.columns, "The dataframe misses some of the required columns"

    def checkpoint_metric(self, domains: Union[str, list[str], bool] = None, mode: str = None) -> float:
        """
        Calculates a metric value for checkpointing optionally utilizing one or more domains. Depending on the 'validation/checkpoint_metric_mode' in the config, image or class scores are obtained. Aggregation is always performed along the hierarchy (image, subject, domains).

        Args:
            domains: The domains to consider additionally.
                - If string or list, metric values for these domains are treated separately before aggregating to the final score (e.g. camera_index).
                - If None, it uses the 'input/target_domain' value from the config.
                - If False, domain columns are ignored.
            mode: Aggregation mode. Either `class_level` (one metric value per label) or `image_level` (multiple classes in an image are aggregated first to get a metric value for one image). If None, the value from the config (`validation/checkpoint_metric_mode`) is used. Defaults to `class_level`.

        Returns: Metric values which can be used for checkpointing.
        """
        assert all(
            c in self.df.columns for c in ["used_labels"] + self.metrics
        ), "The dataframe misses some of the required columns"

        domains = self._domain_defaults(domains)
        if mode is None:
            mode = self.config.get("validation/checkpoint_metric_mode", "class_level")

        if mode == "class_level":
            df_g = self.df.explode(self.metrics + ["used_labels"])
            df_g = df_g.groupby(domains + ["subject_name", "used_labels"], as_index=False)[self.metrics].agg(
                self._default_aggregator
            )
            df_g = df_g.groupby(domains + ["used_labels"], as_index=False)[self.metrics].agg(self._default_aggregator)
            df_g = df_g.groupby(["used_labels"], as_index=False)[self.metrics].agg(self._default_aggregator)
        elif mode == "image_level":
            df_g = self.df.explode(self.metrics + ["used_labels"])
            df_g = df_g.groupby(domains + ["subject_name", "image_name"], as_index=False)[self.metrics].agg(
                self._default_aggregator
            )
            df_g = df_g.groupby(domains + ["subject_name"], as_index=False)[self.metrics].agg(self._default_aggregator)
            df_g = df_g.groupby(["subject_name"], as_index=False)[self.metrics].agg(self._default_aggregator)
        else:
            raise ValueError(f"Invalid mode {mode}")

        return df_g[self.metrics].mean().mean()

    def grouped_metrics(
        self,
        domains: Union[str, list[str], bool] = None,
        keep_subjects: bool = False,
        no_aggregation: bool = False,
        mode: str = None,
        dataset_index: Union[int, None] = 0,
        best_epoch_only: bool = True,
    ) -> pd.DataFrame:
        """
        Calculates a table with metric values for each label and (optionally) the corresponding domains. The scores are first aggregated per subject and then across subjects.

        Args:
            domains: The domains to consider additionally, i.e. the columns which should be kept in the output.
                - If string or list, the corresponding domains will occur as columns in the result table (e.g. camera_index).
                - If None, it uses the 'input/target_domain' value from the config (if the value does not exist in the config, domain columns will be ignored).
                - If False, domain columns are ignored.
            keep_subjects: If True, metrics are only aggregated across images but not across subjects and a subject column will remain in the output table.
            no_aggregation: If True, no aggregation is performed and the output table will contain one row per image. This is useful if results should be visualized per organ but you still want to show the distribution across subjects.
            mode: Aggregation mode. Either `class_level` (one metric value per label) or `image_level` (multiple classes in an image are aggregated first to get a metric value for one image). If None, the value from the config (`validation/checkpoint_metric_mode`) is used. Defaults to `class_level`.
            dataset_index: The index of the dataset which is selected in the table (if available). If None, no selection is carried out.
            best_epoch_only: If True, only results from the best epoch are considered (if available). If False, no selection is carried out and you will get aggregated results per epoch_index (which will also be a column in the resulting table).

        Returns: Table with metric values.
        """
        assert all(c in self.df.columns for c in self.metrics), "The dataframe misses some of the required columns"

        df = self.df
        domains = self._domain_defaults(domains)

        if dataset_index is not None and "dataset_index" in df:
            df = df[df["dataset_index"] == dataset_index]
        if best_epoch_only and "epoch_index" in df and "best_epoch_index" in df:
            df = df[df["epoch_index"] == df["best_epoch_index"]]

        if not best_epoch_only and "epoch_index" not in domains:
            domains.append("epoch_index")

        if mode is None:
            mode = self.config.get("validation/checkpoint_metric_mode", "class_level")

        if "confusion_matrix" in self.metrics:
            assert mode == "image_level", "The confusion matrix can only be used with the image level"
            df_g = self.grouped_cm(
                domains,
                dataset_index,
                best_epoch_only,
                additional_metrics=[m for m in self.metrics if m != "confusion_matrix"],
            )
        else:
            if mode == "class_level":
                assert (
                    "used_labels" in self.df.columns
                ), "used_labels columns is required for class-level aggregation mode"
                df_g = df.explode(self.metrics + ["used_labels"])

                if not no_aggregation:
                    df_g = df_g.groupby(domains + ["subject_name", "used_labels"], as_index=False)[self.metrics].agg(
                        self._default_aggregator
                    )

                    if not keep_subjects:
                        df_g = df_g.groupby(domains + ["used_labels"], as_index=False)[self.metrics].agg(
                            self._default_aggregator
                        )
            elif mode == "image_level":
                if all("image" in m for m in self.metrics):
                    df_g = df
                else:
                    additional = ["used_labels"] if "used_labels" in self.df.columns else []
                    df_g = df.explode(self.metrics + additional)

                if not no_aggregation:
                    df_g = df_g.groupby(domains + ["subject_name", "image_name"], as_index=False)[self.metrics].agg(
                        self._default_aggregator
                    )
                    df_g = df_g.groupby(domains + ["subject_name"], as_index=False)[self.metrics].agg(
                        self._default_aggregator
                    )
            else:
                raise ValueError(f"Invalid mode {mode}")

        return self._resolve_id_columns(df_g, domains)

    def grouped_metrics_epochs(self, domains: Union[str, list[str], bool] = None, mode: str = None) -> pd.DataFrame:
        """
        Similar to grouped_metrics but aggregates the results per epoch.

        Args:
            domains: Additional domains to keep in the output. Per default, the columns `dataset_index`, `epoch_index` and `fold_name` are included in the output.
            mode: `class-level` or `image_level`. See grouped_metrics() for details.

        Returns: Table with aggregated metrics.
        """
        if domains is None:
            domains = []
        elif type(domains) == str:
            domains = [domains]

        for c in ["dataset_index", "epoch_index", "fold_name"]:
            if c not in domains and c in self.df:
                domains.append(c)

        return self.grouped_metrics(domains=domains, mode=mode, dataset_index=None, best_epoch_only=False)

    def grouped_cm(
        self,
        domains: Union[str, list[str], bool] = None,
        dataset_index: Union[int, None] = 0,
        best_epoch_only: bool = True,
        additional_metrics: list[str] = None,
    ) -> pd.DataFrame:
        """
        Similar to grouped_metrics but aggregates only confusion matrices. The confusion matrix for all images of one subject are summed up to one confusion matrix per subject.

        Note: You can for example use the function normalize_grouped_cm() to aggregate the confusion matrices across subject (with information about the standard deviation).

        Args:
            domains: Additional domains to keep in the output.
            dataset_index: The index of the dataset which is selected in the table (if available). If None, no selection is carried out.
            best_epoch_only: If True, only results from the best epoch are considered (if available). If False, no selection is carried out.
            additional_metrics: List of additional metrics which will be calculated and added to the table. These metrics are always calculated on the image level because the confusion matrix is also on the image level.

        Returns: Table with the aggregated confusion matrix.
        """
        df = self.df
        if dataset_index is not None and "dataset_index" in df:
            df = df[df["dataset_index"] == dataset_index]
        if best_epoch_only and "epoch_index" in df and "best_epoch_index" in df:
            df = df[df["epoch_index"] == df["best_epoch_index"]]

        domains = self._domain_defaults(domains)

        df_g = df.groupby(domains + ["subject_name"], as_index=False)["confusion_matrix"].agg(self._default_aggregator)

        if additional_metrics is not None and len(additional_metrics) > 0:
            agg = MetricAggregation(self.df, self.config, metrics=additional_metrics)
            df_additional = agg.grouped_metrics(
                domains, mode="image_level", dataset_index=dataset_index, best_epoch_only=best_epoch_only
            )
            assert (
                df_additional["subject_name"] == df_g["subject_name"]
            ).all(), "Cannot add additional metrics since the subjects do not match"
            df_g[additional_metrics] = df_additional[additional_metrics]

        return self._resolve_id_columns(df_g, domains)

    def domain_accuracy(self, domain: str) -> pd.DataFrame:
        """
        Calculates domain prediction accuracy first aggregated per subject and then across subjects.

        Args:
            domain: Name of the domain (e.g. camera_index).

        Returns: Table with prediction accuracy for each domain value (e.g. every camera).
        """
        assert all(c in self.df.columns for c in [f"{domain}_predicted"]), "The domain predictions are not available"

        df_domain = self.df.explode(["used_labels"])

        # There are extra rows since every label is stored as a row
        def reduce_unique(df: pd.DataFrame) -> pd.Series:
            assert (df.values == df.values[0]).all(), f"Cannot reduce the values {df} to a single value"

            return df.iloc[0]

        cols = [domain, "subject_name", "image_name", f"{domain}_predicted"]
        df_domain = df_domain.groupby(cols, as_index=False)[cols].agg(reduce_unique)

        # Accuracy for each pig
        df_domain[f"{domain}_correct"] = df_domain[domain] == df_domain[f"{domain}_predicted"]
        df_domain = df_domain.groupby([domain, "subject_name"], as_index=False)[f"{domain}_correct"].agg(
            lambda x: sum(x) / len(x)
        )
        df_domain.rename(columns={f"{domain}_correct": "accuracy"}, inplace=True)

        df_domain = self._resolve_id_columns(df_domain, [domain])

        return df_domain

    def _resolve_id_columns(self, df: pd.DataFrame, domains: list[str]) -> pd.DataFrame:
        if "used_labels" in df and self.config["label_mapping"]:
            label_mapping = LabelMapping.from_config(self.config)
            df["label_name"] = [label_mapping.index_to_name(i) for i in df["used_labels"]]
            df = df.rename(columns={"used_labels": "label_index"}).infer_objects()

        # Index to name
        if (
            len(domains) > 0
            and self.config["input/data_spec"]
            and any(d in self.config.get("input/target_domain", []) for d in domains)
        ):
            domain_mappings = DomainMapper.from_config(self.config)
            for domain in domains:
                if domain in domain_mappings:
                    mapper = domain_mappings[domain]
                    df[domain.replace("index", "name")] = [mapper.index_to_domain(i) for i in df[domain]]

        if "label_name" in df:
            df = sort_labels(df)

        return df

    def _domain_defaults(self, domains: Union[str, list[str], None, bool]) -> list[str]:
        if domains is False:
            domains = []
        elif domains is None:
            domains = self.config.get("input/target_domain", [])
        elif type(domains) == str:
            domains = [domains]

        for domain in domains:
            if domain not in self.df and domain.endswith("index") and self.config["input/data_spec"]:
                mapper = DomainMapper(DataSpecification.from_config(self.config), target_domain=domain)
                self.df[domain] = [mapper.domain_index(image_name) for image_name in self.df["image_name"]]

        return domains

    def _default_aggregator(self, series: pd.Series):
        if series.name == "confusion_matrix":
            return np.sum(np.stack(series.values), axis=0)
        else:
            return series.mean()
