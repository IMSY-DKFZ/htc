# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import multiprocessing.pool as mpp
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import psutil
from rich.progress import Progress, ProgressColumn, Task, Text, TimeElapsedColumn


class ProcessingSpeedColumn(ProgressColumn):
    """Render the processing speed in steps per second (similar to TransferSpeedColumn)."""

    def render(self, task: Task) -> Text:
        """Show processing speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")

        return Text(f"{speed:>.2f} it/s", style="progress.data.speed")


def istarmap(pool: mpp.Pool, func: Callable, iterable: Iterable, chunksize: int = 1) -> Generator:
    """Starmap-version of imap to allow for a progress bar during iteration (https://stackoverflow.com/a/57364423)."""
    pool._check_running()
    if chunksize < 1:
        raise ValueError(f"Chunksize must be 1+, not {chunksize}")

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(pool)
    pool._taskqueue.put((pool._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length))

    return (item for chunk in result for item in chunk)


def p_map(
    func: Callable,
    *iterables: Iterable,
    num_cpus: int | float = None,
    task_name: str = "Working...",
    hide_progressbar: bool = False,
    use_threads: bool = False,
    use_executor: bool = False,
) -> Generator:
    """
    Iterate in parallel over a function with one or more iterables. Items are processed and returned in order. A progress bar (using the rich library) will be printed during execution.

    This function is similar to p_map from the p_tqdm package (https://github.com/swansonk14/p_tqdm) but offers Python 3.10+ support and uses rich for the progress bar.

    >>> a = [1, 2, 3]
    >>> b = [1, 2, 3]
    >>> p_map(pow, a, b)  # doctest: +ELLIPSIS
    Working...
    [1, 4, 27]

    Args:
        func: Function to call on each item by the subprocesses.
        iterables: One or more iterables to pass on to the function (multiple iterables map to multiple arguments).
        num_cpus: Number of processes to use (defaults to the number of real (not logical) CPU cores in the system). May also be a factor to denote int(num_cpus * n_cpus).
        task_name: Name of the task which will be printed left to the progress bar.
        hide_progressbar: If True, do not show a progress bar.
        use_threads: If True, use a thread pool instead of a processing pool. Python does not have real multi-threading (due to the GIL) but multiple threads may still result in better CPU utilization if external libraries (like numpy or torch) are used or if the task is I/O-heavy. Threads may be more stable than processes which is useful in Jupyter notebooks (e.g. cells can be executed multiple times without kernel restarts).
        use_executor: If True, use an executor class instead of the multiprocessing pools. There are many differences between those two (cf. [this article](https://superfastpython.com/multiprocessing-pool-vs-processpoolexecutor/) for an overview) but one key difference is that with executors it is possible that processes can have child-processes. For default multiprocessing pools, this would not be possible because they are daemonic (leading to errors like `daemonic processes are not allowed to have children`).

    Returns: List of processed items.
    """
    iterable_lengths = {len(i) for i in iterables}
    assert len(iterable_lengths) == 1, "All iterables must have the same length"

    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False)
    elif type(num_cpus) == float:
        num_cpus = int(round(num_cpus * psutil.cpu_count(logical=False)))

    if num_cpus > len(iterables[0]):
        # We don't need more cpus than tasks
        num_cpus = len(iterables[0])

    items = []
    with Progress(
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        ProcessingSpeedColumn(),
        disable=hide_progressbar,
        refresh_per_second=1,
    ) as progress:
        task_id = progress.add_task(f"[cyan]{task_name}[/]", total=next(iter(iterable_lengths)))

        if use_executor:
            PoolClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
            with PoolClass(num_cpus) as executor:
                for item in executor.map(func, *iterables):
                    progress.advance(task_id)
                    items.append(item)
        else:
            pool = mpp.ThreadPool(num_cpus) if use_threads else mpp.Pool(num_cpus)
            for item in istarmap(pool, func, zip(*iterables, strict=True)):
                progress.advance(task_id)
                items.append(item)

            # We need to properly close the pool for correct coverage (https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html)
            pool.close()
            pool.join()

        progress.refresh()

    return items
