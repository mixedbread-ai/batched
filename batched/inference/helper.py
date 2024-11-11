from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
    return torch and torch.is_tensor(item)


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


def calculate_padding_dimensions(size1: Union[tuple, np.ndarray], size2: Union[tuple, np.ndarray]) -> np.ndarray:
    """
    Calculate the padding dimensions required to expand `size1` to match `size2`.

    This function computes the necessary padding for each dimension of `size1`
    to make it equal to `size2`. The padding is calculated such that the original
    content of `size1` is centered within the expanded dimensions.

    Args:
    size1 : Union[tuple, np.ndarray]
        The original size of the tensor or array, represented as a tuple or numpy array.
        This is the size that needs to be expanded.

    size2 : Union[tuple, np.ndarray]
        The target size of the tensor or array, represented as a tuple or numpy array.
        This is the size that `size1` should be expanded to match.

    Returns:
    np.ndarray
        A 1D numpy array of integers representing the padding required for each dimension.
        The array is structured as [pad_left, pad_right, pad_top, pad_bottom, ...],
        where each pair of values represents the padding on the left/right, top/bottom, etc.

    Raises:
        AssertionError: If `size1` and `size2` do not have the same number of dimensions.

    Example:
        >>> size1 = (3, 3)
        >>> size2 = (5, 5)
        >>> calculate_padding_dimensions(size1, size2)
        array([1, 1, 1, 1], dtype=int32)
    """
    assert len(size1) == len(size2), "Input tensors should have the same number of dimensions"
    size1 = np.array(size1)
    size2 = np.array(size2)
    
    dims_diffs = []
    for d in range(len(size1) - 1, -1, -1):
        dims_diffs.append(size2[d] - size1[d])

    padding_array = np.array([[d//2, d-(d//2)] for d in dims_diffs]).ravel()
    return padding_array.astype(np.int32)


def pad_numpy_tensor(tensor: Feature, pad_sizes: tuple, mode: str, pad_value: int) -> Feature:
    """
    Pads a given NumPy tensor with specified sizes, mode, and constant value.

    This function adapts the padding mechanism to match the structure expected by NumPy's `np.pad` function.
    Instead of passing a single tuple for padding sizes, it constructs an array of tuples, where each tuple
    corresponds to the padding for a specific dimension.

    Args:
        tensor (Feature): The input tensor to be padded.
        pad_sizes (tuple): A tuple specifying the padding sizes. The tuple should contain pairs of integers
                           where each pair represents the padding for a dimension (before, after).
        mode (str): The padding mode to be used. Refer to NumPy's `np.pad` documentation for available modes.
        pad_value (int): The constant value to use for padding when the mode is 'constant'.

    Returns:
        Feature: The padded tensor.

    Notes:
        - The `pad_sizes` tuple should contain an even number of integers, where each pair corresponds to
          the padding for a dimension (before, after). It has to be passed in torch format (left, right, top, bottom).
        - The function assumes that the input tensor is a NumPy array.
        - The `mode` parameter should be one of the valid modes supported by NumPy's `np.pad` function.
    """
    # Numpy has a different order and structure for padding, so instead of passing a tuple we need to pass an array of tuples.
    pad_width = [(pad_sizes[-(i*2+2)], pad_sizes[-(i*2+1)]) 
             for i in range(len(pad_sizes)//2)]
    return np.pad(tensor, pad_width, mode, constant_values=pad_value)


def stack_features(inputs: list[dict[str, Feature]], pad_tokens: dict[str, int], mode: Literal["around", "bottom_right"] = "bottom_right") -> dict[str, Feature]:
    """
    Stack a list of model features into a single batch.
    Accepts two padding modes: 'around' which pads the tokens around the input and bottom_right, which pads the tokens on the bottom and to the right.

    Args:
        inputs (list[ModelFeatures]): List of input features to stack.
        pad_tokens (dict[str, int]): Dictionary of padding tokens for each feature.
        mode (Literal['around', 'bottom_right']): Mode of the padding.

    Returns:
        ModelFeatures: Stacked features as a single batch.

    Example:
        >>> inputs = [{'f1': torch.randn(3, 154, 250)}, {'f1': torch.randn(2, 255, 255)}]
        >>> pad_tokens = {'f1': 0}
        >>> mode = 'around'
        >>> padded_tensor = stack_features(inputs, pad_tokens, mode)
        >>> print(padded_tensor.size())
        (3, 255, 255)

    Raises:
        AssertionError: If padding mode is not supported.

    Notes:
        - 'bottom_right' mode is 5 times more performant but uses more memory since it preallocates the tensor.
    """
    assert mode == "around" or mode == "bottom_right", "Padding modes available are ['around', 'bottom_right']."
    if mode == "around":
        return _stack_features_around(inputs, pad_tokens)
    else:
        return _stack_features_bottom_right(inputs, pad_tokens)


def _stack_features_around(inputs: list[dict[str, Feature]], pad_tokens: dict[str, int]) -> dict[str, Feature]:
    lib = torch_or_np(inputs)
    keys = inputs[0].keys()
    num_items = len(inputs)

    shapes = tuple(tuple(item[key].shape for item in inputs) for key in keys)
    max_shapes = tuple(tuple(max(dim) for dim in zip(*shape)) for shape in shapes)[0]

    padded_tensors = {}
    pad_f = torch.nn.functional.pad if lib == torch else pad_numpy_tensor

    for key in keys:
        padded_tensors[key] = lib.stack([
            pad_f(inputs[i][key], tuple(calculate_padding_dimensions(inputs[i][key].shape, max_shapes)), 'constant', pad_tokens.get(key, 0))
            for i in range(num_items)
        ])
    return padded_tensors


def _stack_features_bottom_right(inputs: list[dict[str, Feature]], pad_tokens: dict[str, int]) -> dict[str, Feature]:
    lib = torch_or_np(inputs)
    keys = inputs[0].keys()
    num_items = len(inputs)

    dtypes = tuple(inputs[0][key].dtype for key in keys)
    shapes = tuple(tuple(item[key].shape for item in inputs) for key in keys)
    max_shapes = tuple(tuple(max(dim) for dim in zip(*shape)) for shape in shapes)

    padded_tensors = {
        key: lib.full((num_items, *max_shape), pad_tokens.get(key, 0), dtype=dtype)
        for key, max_shape, dtype in zip(keys, max_shapes, dtypes)
    }

    for i in range(num_items):
        for key in keys:
            item_tensor = inputs[i][key]
            padded_tensor = padded_tensors[key][i]
            padded_tensor[tuple(slice(0, s) for s in item_tensor.shape)] = item_tensor
    
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
