# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from collections.abc import Callable

import torch


def bootstrapped_metric(
    n_samples: int,
    metric_calculator: Callable[[torch.Tensor], dict[str, torch.Tensor | tuple[torch.Tensor]]],
    n_bootstraps: int = 1000,
) -> dict[str, torch.Tensor | tuple[torch.Tensor]]:
    """
    General helper function to calculate bootstrapped metrics.

    Note: For reproducible results, set a PyTorch seed before using this function (`torch.manual_seed()`).

    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> target = torch.tensor([0, 1, 2, 3])
    >>> preds = torch.tensor([0, 2, 1, 3])
    >>> metrics = bootstrapped_metric(
    ...     n_samples=len(target),
    ...     metric_calculator=lambda indices: {"accuracy": (preds[indices] == target[indices]).sum() / len(target)},
    ... )
    >>> metrics["accuracy"].shape
    torch.Size([1000])

    Args:
        n_samples: Number of samples to bootstrap.
        metric_calculator: Function that takes a tensor of indices and returns a dictionary of metrics.
        n_bootstraps: Number of bootstraps to perform.

    Returns: Dictionary of bootstrapped metrics (with the same keys as returned by the metrics calculator function).
    """
    bootstraps = torch.randint(0, n_samples, (n_bootstraps, n_samples))
    outputs = {}

    for b in range(bootstraps.shape[0]):
        out = metric_calculator(bootstraps[b])
        for k, v in out.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)

    # Stack bootstrapped output together
    for k, v in outputs.items():
        if type(v[0]) == torch.Tensor:
            outputs[k] = torch.stack(v)
        elif type(v[0]) == tuple:
            outputs[k] = tuple([torch.stack(o) for o in zip(*v, strict=True)])

    return outputs
