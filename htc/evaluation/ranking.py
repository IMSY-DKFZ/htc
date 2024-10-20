# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class BootstrapRanking:
    def __init__(
        self,
        data: pd.DataFrame,
        task: str | None = "task",
        algorithm: str = "algorithm",
        case: str = "case",
        value: str = "value",
        n_bootstraps: int = 1000,
        smaller_better: bool = False,
    ):
        """
        Creates a bootstrap ranking similar to challengeR. You can access the ranking or derived statistics via properties of this class. Please refer to the BootstrapRankingExample notebook for an example of how to use this class to compare two training runs with a bubble plot.

        For each bootstrap, metric values are sampled with replacement for each task and algorithm and then a ranking is performed. The same samples are used across algorithms (e.g. the same set of organs are compared).

        Note: For reproducible results, set a numpy seed before using this class.

        Args:
            data: Dataframe with the metric data.
            task: Name of the task column in the dataframe. A ranking is performed separately per task. Set to None if you do not want to analyze separate tasks (a dummy task column will be added to the tables).
            algorithm: Name of the algorithm column in the dataframe. There must be at least two algorithms which you want to compare against each other.
            case: Name of the column which contains the sample names (e.g. label name or subject name).
            value: Name of the column which contains the metric data. There must be one value per sample.
            n_bootstraps: Number of bootstraps to perform.
            smaller_better: If True, smaller numbers indicate a better algorithm performance.
        """
        self.data = data
        self.task = task
        if self.task is None:
            assert "task" not in self.data.columns, "Cannot add a dummy task column because it already exists"
            self.task = "task"
            self.data[self.task] = "single_task"
        self.algorithm = algorithm
        self.case = case
        self.value = value
        self.n_bootstraps = n_bootstraps
        self.smaller_better = smaller_better

        self._bootstraps = None
        self._counts = None
        self._statistics = None

    @property
    def bootstraps(self) -> pd.DataFrame:
        """
        Returns: Table with the bootstrap results (columns bootstrap, task, algorithm and rank).
        """
        if self._bootstraps is not None:
            return self._bootstraps

        df_selection = self.data[[self.task, self.algorithm, self.case, self.value]].drop_duplicates()
        assert len(df_selection) == len(self.data), (
            f"The dataframe after selecting the columns has a different number of unique rows ({len(df_selection)}) as"
            f" before ({len(self.data)}). Please make sure that every row in the dataframe is unique"
        )
        assert not df_selection[self.value].isna().any(), "There must not be any nan values for the value column"
        assert df_selection[self.algorithm].nunique() >= 2, "There must be at least two algorithms"

        rows = {
            "bootstrap": [],
            "task": [],
            "algorithm": [],
            "rank": [],
        }

        max_value = df_selection[self.value].max()

        for task_name in df_selection[self.task].unique():
            df_task = df_selection[df_selection[self.task] == task_name]

            cases_per_algorithm = df_task.groupby(self.algorithm, as_index=False)[self.case].agg(
                unique_cases=lambda x: np.unique(x).tolist()
            )
            assert all(
                cases_per_algorithm.unique_cases[0] == cases_per_algorithm.unique_cases[i]
                for i in range(len(cases_per_algorithm.unique_cases))
            ), "The same cases must be used for all algorithms"

            # Select the sample indices for all bootstrap (the same samples will be used for all algorithms)
            n_cases = df_task[self.case].nunique()
            bootstrap_indices = np.random.randint(0, n_cases, (n_cases, self.n_bootstraps))

            algorithms = df_task[self.algorithm].unique()
            task_data = []
            for algorithm_name in algorithms:
                df_a = df_task[df_task[self.algorithm] == algorithm_name]

                # All sample values
                values = df_a[self.value].values
                if not self.smaller_better:
                    values = max_value - values

                # Bootstrap selection of the sample values and mean score per bootstrap
                means = np.mean(values[bootstrap_indices], axis=0)
                task_data.append(means)

            # Rankings for this task
            task_data = np.stack(task_data)
            rankings = rankdata(task_data, axis=0, method="min")

            rows["bootstrap"] += list(range(1, self.n_bootstraps + 1))
            rows["task"] += [task_name] * self.n_bootstraps
            rows["algorithm"] += [algorithms.tolist()] * self.n_bootstraps
            rows["rank"] += rankings.T.tolist()

        df_bootstraps = pd.DataFrame(rows)
        df_bootstraps = df_bootstraps.explode(["algorithm", "rank"])

        self._bootstraps = df_bootstraps
        return self._bootstraps

    @property
    def counts(self) -> pd.DataFrame:
        """
        Returns: Table with a rank count statistic per task and algorithm.
        """
        if self._counts is not None:
            return self._counts

        self._counts = (
            self.bootstraps.groupby(["task", "algorithm", "rank"], as_index=True)[["rank"]]
            .count()
            .rename(columns={"rank": "count"})
            .reset_index()
        )
        return self._counts

    @property
    def statistics(self) -> pd.DataFrame:
        """
        Returns: Table with rank statistics per task and algorithm (mean, median rank etc.). Useful for bubble plots.
        """
        if self._statistics is not None:
            return self._statistics

        rows = []
        for task in self.counts.task.unique():
            for alg in self.counts.algorithm.unique():
                df_sel = self.counts.query("task == @task & algorithm == @alg")
                ranks = np.concatenate([np.repeat(r["rank"], r["count"]) for _, r in df_sel.iterrows()])
                rows.append([
                    task,
                    alg,
                    np.mean(ranks),
                    np.median(ranks),
                    np.std(ranks),
                    np.quantile(ranks, 0.025, method="closest_observation"),
                    np.quantile(ranks, 0.975, method="closest_observation"),
                ])

        self._statistics = pd.DataFrame(
            rows, columns=["task", "algorithm", "mean_rank", "median_rank", "std_rank", "min_CI", "max_CI"]
        )
        self._statistics.sort_values(["task", "mean_rank"], inplace=True)

        return self._statistics


class BootstrapRankingSubjects(BootstrapRanking):
    def __init__(self, *args, subject_column: str, **kwargs):
        """
        Compared to BootstrapRanking, this class expects a table with subject-level scores per organ (e.g. obtained via the `keep_subjects=True` parameter of the MetricAggregation class). The bootstraps will be performed on the subject-level per organ. That is, if there are 5 subjects with scores for the label stomach, these five scores will be sampled 1000 times and then averaged across subjects. In total, there will be 1000 scores for each label and the ranking is performed across labels for each bootstrap.

        This allows to capture the sampling variability of the organ scores across subjects (and not across the already-averaged organ scores) which is more sound for class-level aggregated scores.
        """
        super().__init__(*args, **kwargs)
        self.subject_column = subject_column

    @property
    def bootstraps(self) -> pd.DataFrame:
        """
        Returns: Table with the bootstrap results (columns bootstrap, task, algorithm and rank).
        """
        if self._bootstraps is not None:
            return self._bootstraps

        df_selection = self.data[
            [self.task, self.algorithm, self.case, self.value, self.subject_column]
        ].drop_duplicates()
        assert len(df_selection) == len(self.data), (
            f"The dataframe after selecting the columns has a different number of unique rows ({len(df_selection)}) as"
            f" before ({len(self.data)}). Please make sure that every row in the dataframe is unique"
        )
        assert not df_selection[self.value].isna().any(), "There must not be any nan values for the value column"
        assert df_selection[self.algorithm].nunique() >= 2, "There must be at least two algorithms"

        rows = {
            "bootstrap": [],
            "task": [],
            "algorithm": [],
            "rank": [],
        }

        max_value = df_selection[self.value].max()

        for task_name in df_selection[self.task].unique():
            df_task = df_selection[df_selection[self.task] == task_name]

            cases_per_algorithm = df_task.groupby(self.algorithm, as_index=False)[self.case].agg(
                unique_cases=lambda x: np.unique(x).tolist()
            )
            assert all(
                cases_per_algorithm.unique_cases[0] == cases_per_algorithm.unique_cases[i]
                for i in range(len(cases_per_algorithm.unique_cases))
            ), "The same cases must be used for all algorithms"

            algorithms = df_task[self.algorithm].unique()
            case_data = np.empty((len(algorithms), self.n_bootstraps, df_task[self.case].nunique()))
            for c, case in enumerate(df_task[self.case].unique()):
                df_case = df_task[df_task[self.case] == case]

                # We sample the current available number of subjects multiple times
                n_subjects = df_case[self.subject_column].nunique()
                subjects_per_algorithm = df_case.groupby(self.algorithm, as_index=False)[self.subject_column].agg(
                    unique_subjects=lambda x: np.unique(x).tolist()
                )
                assert all(
                    subjects_per_algorithm.unique_subjects[0] == subjects_per_algorithm.unique_subjects[i]
                    for i in range(len(subjects_per_algorithm.unique_subjects))
                ), "The same subjects must be used for all algorithms"

                # Select the sample indices for all bootstrap (the same samples will be used for all algorithms)
                bootstrap_indices = np.random.randint(0, n_subjects, (n_subjects, self.n_bootstraps))

                for a, algorithm_name in enumerate(algorithms):
                    df_a = df_case[df_case[self.algorithm] == algorithm_name]

                    # All sample values
                    values = df_a[self.value].values
                    if not self.smaller_better:
                        values = max_value - values

                    # Bootstrap selection of the sample values and mean score per bootstrap (here: mean score across subjects)
                    means = np.mean(values[bootstrap_indices], axis=0)
                    case_data[a, :, c] = means

            # Aggregate back to the case level (e.g. ranking still based on the average organ scores)
            case_data = np.mean(case_data, axis=-1)

            # Rankings for this task
            rankings = rankdata(case_data, axis=0, method="min")

            rows["bootstrap"] += list(range(1, self.n_bootstraps + 1))
            rows["task"] += [task_name] * self.n_bootstraps
            rows["algorithm"] += [algorithms.tolist()] * self.n_bootstraps
            rows["rank"] += rankings.T.tolist()

        df_bootstraps = pd.DataFrame(rows)
        df_bootstraps = df_bootstraps.explode(["algorithm", "rank"])

        self._bootstraps = df_bootstraps
        return self._bootstraps
