from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, Union

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

NDArrayOrTensor = Union["NDArray", "Tensor"]

ModelFeatures = dict[str, Union[NDArrayOrTensor, list["Tensor"], list["NDArray"]]]
ModelOutputs = Union[NDArrayOrTensor, list[NDArrayOrTensor]]

InferenceFunction = Callable[[ModelFeatures], ModelOutputs]
AsyncInferenceFunction = Callable[[ModelFeatures], Awaitable[ModelOutputs]]
