import inspect
from types import FunctionType, MethodType
from typing import Any, Callable, Iterable, TypeVar, Union

import numpy as np
import torch

from batch.types import NDArrayOrTensor

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


def pad_to_max_length(
    inputs: Union[dict[str, NDArrayOrTensor], NDArrayOrTensor],
    max_length: int,
    pad_token_id: int,
) -> Union[dict[str, NDArrayOrTensor], NDArrayOrTensor]:
    """Pad the inputs to the maximum length with the pad token id.

    Args:
        inputs (Union[Dict[str, NDArrayOrTensor], NDArrayOrTensor]): The inputs to pad.
        max_length (int): The maximum length to pad to.
        pad_token_id (int): The pad token id to use.

    Returns:
        Union[Dict[str, NDArrayOrTensor], NDArrayOrTensor]: Padded inputs.

    Raises:
        TypeError: If inputs are neither a dictionary nor a tensor/array.

    """
    if isinstance(inputs, dict):
        return _pad_dict_inputs(inputs, max_length, pad_token_id)
    if isinstance(inputs, (torch.Tensor, np.ndarray)):
        return _pad_tensor_or_array(inputs, max_length, pad_token_id)

    msg = "Inputs must be either a dictionary or a tensor/array."
    raise TypeError(msg)


def _pad_dict_inputs(
    inputs: dict[str, NDArrayOrTensor], max_length: int, pad_token_id: int
) -> dict[str, NDArrayOrTensor]:
    """Helper function to pad dictionary inputs."""
    is_tensor = isinstance(first(inputs.values()), torch.Tensor)
    for key, value in inputs.items():
        if value.shape[-1] < max_length:
            if is_tensor:
                pre_allocated = torch.full(
                    (value.shape[0], max_length), pad_token_id, dtype=value.dtype, device=value.device
                )
                inputs[key] = torch.cat((value, pre_allocated[:, value.shape[-1] :]), dim=-1)
            else:
                pre_allocated = np.full((value.shape[0], max_length - value.shape[-1]), pad_token_id, dtype=value.dtype)
                inputs[key] = np.concatenate((value, pre_allocated), axis=-1)
    return inputs


def _pad_tensor_or_array(inputs: NDArrayOrTensor, max_length: int, pad_token_id: int) -> NDArrayOrTensor:
    """Helper function to pad tensor or array inputs."""
    if isinstance(inputs, torch.Tensor):
        pre_allocated = torch.full(
            (inputs.shape[0], max_length), pad_token_id, dtype=inputs.dtype, device=inputs.device
        )
        return torch.cat((inputs, pre_allocated[:, inputs.shape[-1] :]), dim=-1)

    pre_allocated = np.full((inputs.shape[0], max_length), pad_token_id, dtype=inputs.dtype)
    return np.concatenate((inputs, pre_allocated[:, inputs.shape[-1] :]), axis=-1)


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
