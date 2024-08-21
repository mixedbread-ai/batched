from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np

from batch.utils import first

try:
    import torch
except ImportError:
    torch = None


NDArrayOrTensor = TypeVar("NDArrayOrTensor", "np.ndarray", "torch.Tensor")

ModelFeatures = dict[str, NDArrayOrTensor | list[NDArrayOrTensor]]
ModelOutputs = NDArrayOrTensor | list[NDArrayOrTensor]

BatchInfer = Callable[[ModelFeatures], ModelOutputs]


def torch_or_np(item):
    """Determine whether to use torch or numpy based on the input type.

    Args:
        item (Any): The input item to check.

    Returns:
        ModuleType: Either torch or numpy module.

    Raises:
        ImportError: If torch is not installed, but a tensor is passed.
        ValueError: If the input type is not supported.
    """
    if isinstance(item, dict | list | tuple):
        return torch_or_np(first(item.values()) if isinstance(item, dict) else item[0])
    if isinstance(item, (np.ndarray, np.generic)):
        return np

    if not torch:
        msg = "Torch is not installed. Please install it with `pip install torch` to use this function."
        raise ImportError(msg)

    if torch.is_tensor(item) or isinstance(item, (torch.dtype, torch.Number)):
        return torch

    msg = f"Unsupported input type: {type(item)}"
    raise ValueError(msg)


def stack_features(inputs: list[ModelFeatures], pad_tokens: dict[str, int]) -> ModelFeatures:
    """
    Stack a list of model features into a single batch.

    Args:
        inputs (list[ModelFeatures]): List of input features to stack.
        pad_tokens (dict[str, int]): Dictionary of padding tokens for each feature.

    Returns:
        ModelFeatures: Stacked features as a single batch.
    """
    if not inputs:
        return {}

    lib = torch_or_np(inputs[0])
    keys = inputs[0].keys()
    max_length = max(item[first(keys)].shape[0] for item in inputs)

    padded_tensors = {key: lib.full((len(inputs), max_length), pad_tokens.get(key, 0), dtype=lib.int64) for key in keys}

    for i, item in enumerate(inputs):
        for key, tensor in padded_tensors.items():
            tensor_length = item[key].shape[0]
            tensor[i, :tensor_length] = item[key]

    return padded_tensors


def unstack_features(inputs: ModelFeatures) -> list[ModelFeatures]:
    """
    Unstack a batch of model features into a list of individual features.

    Args:
        inputs (ModelFeatures): Batch of input features to unstack.

    Returns:
        list[ModelFeatures]: List of individual feature dictionaries.
    """
    n_items = len(first(inputs.values()))
    keys = inputs.keys()
    return [{key: inputs[key][i] for key in keys} for i in range(n_items)]


def stack_outputs(outputs: list[ModelOutputs]) -> ModelOutputs:
    """
    Stack a list of model outputs into a single batch.

    Args:
        outputs (list[ModelOutputs]): List of model outputs to stack.

    Returns:
        ModelOutputs: Stacked outputs as a single batch.
    """
    lib = torch_or_np(outputs)
    if isinstance(outputs[0], dict):
        return {key: lib.stack([output[key] for output in outputs]) for key in outputs[0]}
    if isinstance(outputs[0], list):
        return [lib.stack(outputs) for outputs in zip(*outputs)]
    return lib.stack(outputs)


def unstack_outputs(outputs: ModelOutputs) -> list[ModelOutputs]:
    """
    Unstack a batch of model outputs into a list of individual outputs.

    Args:
        outputs (ModelOutputs): Batch of model outputs to unstack.

    Returns:
        list[ModelOutputs]: List of individual model outputs.
    """
    n_items = len(first(outputs)) if isinstance(outputs, list) else len(outputs)

    if isinstance(outputs, list):
        return [[output[i] for output in outputs] for i in range(n_items)]
    return [outputs[i] for i in range(n_items)]
