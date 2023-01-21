# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Any

from htc.tivita.DataPath import DataPath
from htc.utils.parallel import p_map


class DatasetIteration(ABC):
    def __init__(self, paths: list[DataPath]):
        self.paths = paths

    @abstractmethod
    def compute(self, i: int) -> None:
        """Computations for the i-th path."""
        pass

    def run(self) -> Any:
        indices = list(range(len(self.paths)))
        results = p_map(self.compute, indices, task_name=self.__class__.__name__)

        self.finished(results)

    def finished(self, results: list[Any]) -> None:
        """
        Called when the computations for all paths are finished.

        Args:
            results: Results from the computation step (one entry per path).
        """
        pass
