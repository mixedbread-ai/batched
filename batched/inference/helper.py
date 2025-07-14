from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batched.utils import first

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

if TYPE_CHECKING:
    from batched.types import Feature


def _is_np(item: Any) -> bool:
    return np and isinstance(item, (np.ndarray, np.generic))


def _is_torch(item: Any) -> bool:
    return torch and (torch.is_tensor(item) or isinstance(item, (torch.dtype, torch.Number)))


def torch_or_np(item: Any):
    """Determine whether to use torch or numpy based on the input type.

    Args:
        item (Any): The input item to check.

    Returns:
        ModuleType: Either torch or numpy module.

    Raises:
        ImportError: If torch is not installed, but a tensor is passed.
        ValueError: If the input type is not supported.
    """
    if not np and not torch:
        msg = "Either numpy or torch needs to be installed."
        raise ImportError(msg)

    if isinstance(item, (dict, list, tuple)) and item:
        return torch_or_np(first(item.values()) if isinstance(item, dict) else item[0])

    if _is_np(item):
        return np

    if _is_torch(item):
        return torch

    msg = f"Unsupported input type: {type(item)}"
    raise ValueError(msg)


def stack_features(
    inputs: list[dict[str, Feature]],
    pad_tokens: dict[str, int],
    padding_side: str = "right"
) -> dict[str, Feature]:
    """
    Stack a list of model features into a single batch.

    Args:
        inputs (list[ModelFeatures]): List of input features to stack.
        pad_tokens (dict[str, int]): Dictionary of padding tokens for each feature.
        padding_side (str): Side to add padding tokens. Either "left" or "right". Defaults to "right".

    Returns:
        ModelFeatures: Stacked features as a single batch.
    """
    lib = torch_or_np(inputs)
    keys = inputs[0].keys()
    max_length = max(item[first(keys)].shape[0] for item in inputs)

    padded_tensors = {key: lib.full((len(inputs), max_length), pad_tokens.get(key, 0), dtype=lib.int64) for key in keys}

    for i, item in enumerate(inputs):
        for key, tensor in padded_tensors.items():
            tensor_length = item[key].shape[0]
            if padding_side == "left":
                # Left padding: fill from the right side (end of sequence)
                start_idx = max_length - tensor_length
                tensor[i, start_idx:] = item[key]
            else:
                # Right padding: fill from the left side (start of sequence)
                tensor[i, :tensor_length] = item[key]

    return padded_tensors


def unstack_features(inputs: dict[str, Feature]) -> list[dict[str, Feature]]:
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


def stack_outputs(outputs: list[Feature]) -> Feature:
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


def unstack_outputs(outputs: Feature) -> list[Feature]:
    """
    Unstack a batch of model outputs into a list of individual outputs.

    Args:
        outputs (ModelOutputs): Batch of model outputs to unstack.

    Returns:
        list[ModelOutputs]: List of individual model outputs.
    """
    if isinstance(outputs, dict):
        n_items = len(first(outputs.values()))
        return [{key: value[i] for key, value in outputs.items()} for i in range(n_items)]
    if isinstance(outputs, list):
        return list(map(list, zip(*outputs)))
    return [outputs[i] for i in range(len(outputs))]
