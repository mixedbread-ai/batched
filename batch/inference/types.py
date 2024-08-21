from __future__ import annotations

from collections.abc import Callable

from numpy.typing import NDArray
from torch import Tensor

from batch.utils import first, torch_or_np

NDArrayOrTensor = NDArray | Tensor

ModelFeatures = dict[str, NDArrayOrTensor | list[NDArrayOrTensor]]
ModelOutputs = NDArrayOrTensor | list[NDArrayOrTensor]

BatchInfer = Callable[[ModelFeatures], ModelOutputs]


def stack_features(inputs: list[ModelFeatures], pad_tokens: dict[str, int]) -> ModelFeatures:
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
    n_items = len(first(inputs.values()))
    keys = inputs.keys()
    return [{key: inputs[key][i] for key in keys} for i in range(n_items)]


def stack_outputs(outputs: list[ModelOutputs]) -> ModelOutputs:
    lib = torch_or_np(outputs)
    if isinstance(outputs[0], dict):
        return {key: lib.stack([output[key] for output in outputs]) for key in outputs[0]}
    if isinstance(outputs[0], list):
        return [lib.stack(outputs) for outputs in zip(*outputs)]
    return lib.stack(outputs)


def unstack_outputs(outputs: ModelOutputs) -> list[ModelOutputs]:
    n_items = len(first(outputs)) if isinstance(outputs, list) else len(outputs)

    if isinstance(outputs, list):
        return [[output[i] for output in outputs] for i in range(n_items)]
    return [outputs[i] for i in range(n_items)]
