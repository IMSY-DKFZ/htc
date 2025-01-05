# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

import sys
from pathlib import Path

import nbformat
import numpy as np


def check_notebook(file: Path) -> None:
    """
    Check for common errors in a Jupyter notebook file. Currently, this includes the following checks:
    - The notebook does not use p_map without specifying `use_threads=True`.
    - No errors (output cells with exceptions) in the notebook.
    - No unexecuted cells.
    - Cells are executed in order.

    Args:
        file: The path to the Jupyter notebook file.
    """
    nb = nbformat.read(file, as_version=nbformat.NO_CONVERT)

    # Check for errors
    for cell in nb.cells:
        if cell["cell_type"] == "code":
            source = cell["source"]
            if "p_map(" in source:
                assert "use_threads=True" in source, (
                    f"The notebook {file} uses p_map without specifying use_threads=True. This is not"
                    " recommended as it can lead to multiprocessing issues (especially during testing)"
                )

        if "outputs" in cell:
            for output in cell["outputs"]:
                assert "ename" not in output, f"The notebook {file} contains an {output['ename']} exception"

    # Check for unexecuted cells
    counts = [cell["execution_count"] for cell in nb.cells if "execution_count" in cell]

    # Only the last value is allowed to be None
    if counts[-1] is None:
        counts = counts[:-1]

    assert not any(c is None for c in counts), (
        f"Some cells are not executed in the notebook {file} (did you forget to delete them?)"
    )

    # Check for cell ordering
    counts = [c for c in counts if c is not None]
    assert not any(np.ediff1d(counts) != 1), (
        f"The cells in the notebook {file} are not executed in order! This is dangerous as some side"
        " effects may still be present in the notebook. Please restart the kernel and run all cells again to"
        " ensure a clean notebook state"
    )


if __name__ == "__main__":
    returncode = 0

    for file in sys.argv[1:]:
        try:
            check_notebook(Path(file))
        except Exception as e:
            print(e)
            returncode = 1

    sys.exit(returncode)
