from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

from batch.utils import first, torch_or_np

NDArrayOrTensor = NDArray | Tensor

ModelFeatures = dict[str, NDArrayOrTensor | list[NDArrayOrTensor]]
ModelOutputs = NDArrayOrTensor | list[NDArrayOrTensor]

BatchInfer = Callable[[ModelFeatures], ModelOutputs]


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
