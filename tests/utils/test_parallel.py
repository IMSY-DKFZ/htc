# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing
import os
from functools import partial

from htc.utils.parallel import p_map


# Tests adapted from p_tqdm: https://github.com/swansonk14/p_tqdm/blob/master/tests/tests.py
def parallel_add1(a: int) -> int:
    return a + 1


def parallel_add2(a: int, b: int) -> int:
    return a + b


def parallel_add3(a: int, b: int, c: int = 0) -> int:
    return a + 2 * b + 3 * c


def process_pid(a: int) -> int:
    return os.getpid()


class TestParallel:
    def test_one_list(self) -> None:
        array = [1, 2, 3]
        result = p_map(parallel_add1, array)

        correct_array = [2, 3, 4]
        assert correct_array == result

    def test_two_lists(self) -> None:
        array_1 = [1, 2, 3]
        array_2 = [10, 11, 12]
        result = p_map(parallel_add2, array_1, array_2)

        correct_array = [11, 13, 15]
        assert correct_array == result

    def test_two_lists_and_one_single(self) -> None:
        array_1 = [1, 2, 3]
        array_2 = [10, 11, 12]
        single = 5
        result = p_map(partial(parallel_add3, single), array_1, array_2)

        correct_array = [37, 42, 47]
        assert correct_array == result

    def test_one_list_and_two_singles(self) -> None:
        array = [1, 2, 3]
        single_1 = 5
        single_2 = -2
        result = p_map(partial(parallel_add3, single_1, c=single_2), array)

        correct_array = [1, 3, 5]
        assert correct_array == result

    def test_list_and_generator_and_single_equal_length(self) -> None:
        array = [1, 2, 3]
        generator = range(3)
        single = -3
        result = p_map(partial(parallel_add3, c=single), array, generator)

        correct_array = [-8, -5, -2]
        assert correct_array == result

    def test_num_cpus(self) -> None:
        array = list(range(multiprocessing.cpu_count()))

        pids = p_map(process_pid, array)
        # <= and not == since less process may be used if the task finishes fast
        assert len(set(pids)) <= multiprocessing.cpu_count()

        pids = p_map(process_pid, array, num_cpus=2)
        assert len(set(pids)) <= 2

        pids = p_map(process_pid, array, num_cpus=0.5)
        assert len(set(pids)) <= multiprocessing.cpu_count() // 2
