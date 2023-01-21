# SPDX-FileCopyrightText: 2022 Division of Intelligent Medical Systems, DKFZ
# SPDX-License-Identifier: MIT

from typing import Any, Callable

import torch
import torch.nn as nn


class ForwardHookPromise:
    def __init__(self, module: nn.Module, hook: Callable = None):
        """
        This class can be used to safely register forward hooks on modules. It returns a promise for the forward pass which can be used to read the data afterwards. The promise is meant to be used in a with block and the data is only accessible inside this block. This ensures that the data is freed afterwards avoiding out of memory issues.

        The promise can be used on any torch module:

        >>> class MyModule(nn.Module):
        ...     def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        ...         return {'a': x, 'b': x * x}
        >>> m = MyModule()

        The Promise can only used in a context manager (to ensure proper cleanup):

        >>> own_hook = lambda module, module_in, module_out: module_out['a']
        >>> with ForwardHookPromise(m) as p1, ForwardHookPromise(m, own_hook) as p2:
        ...     print(m(torch.ones(1)))  # Trigger the forward pass
        ...     print(p1.data())
        ...     print(p2.data())
        {'a': tensor([1.]), 'b': tensor([1.])}
        {'a': tensor([1.]), 'b': tensor([1.])}
        tensor([1.])

        Hooks clean up after themselves:

        >>> len(m._forward_hooks)
        0

        Note: If you want to use a dynamic number of promises, you can use the ExitStack context manager (https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack).

        Args:
            module: The module where the forward pass should be applied.
            hook: An optional callable which can be used as hook function. If None, a default hook will be used which
                  just saves the output of the forward pass.
        """
        self._data = None
        self._in_context = False
        self._handle = module.register_forward_hook(self._hook)
        self.hook = hook

    def data(self) -> torch.Tensor:
        assert self._data is not None
        assert self._in_context, "Data of the tensor can only be accessed in a with block"

        return self._data

    def _hook(self, module: nn.Module, module_in: tuple, module_out: torch.Tensor) -> None:
        if self.hook is None:
            self._data = module_out
        else:
            self._data = self.hook(module, module_in, module_out)

    def __enter__(self) -> "ForwardHookPromise":
        self._in_context = True
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        self._handle.remove()
        del self._data
        self._in_context = False
