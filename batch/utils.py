from __future__ import annotations

import inspect
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")

try:
    import torch
except ImportError:
    torch = None


def first(it: Iterable[T]) -> T:
    """Return the first item from an iterable.

    Args:
        it (Iterable[T]): The iterable to get the first item from.

    Returns:
        T: The first item in the iterable.

    Raises:
        StopIteration: If the iterable is empty.

    """
    return next(iter(it))


def torch_or_np(item: Any):
    """Determine whether to use torch or numpy based on the input type.

    Args:
        item (Any): The input item to check.

    Returns:
        ModuleType: Either torch or numpy module.

    Raises:
        ImportError: If torch is not installed, but a tensor/array is passed.
        TypeError: If the input type is not supported.

    """
    if isinstance(item, dict | list | tuple):
        return torch_or_np(first(item.values()) if isinstance(item, dict) else item[0])
    if isinstance(item, np.ndarray):
        return np

    if not torch:
        msg = "Torch is not installed. Please install it with `pip install torch` to use this function."
        raise ImportError(msg)

    if isinstance(item, torch.Tensor):
        return torch

    msg = f"Unsupported input type: {type(item)}"
    raise ValueError(msg)


def is_method(func: Callable) -> bool:
    """Check if a callable is a method.

    Args:
        func (Callable): The function to check.

    Returns:
        bool: True if the callable is a method, False otherwise.

    """
    if not isinstance(func, (FunctionType, MethodType)):
        return False
    return next(iter(inspect.signature(func).parameters)) in ("self", "cls")
