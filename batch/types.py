from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import TypeVar, get_type_hints

T = TypeVar("T")
U = TypeVar("U")

BatchFunc = Callable[[list[T]], list[U]]


def _validate_batch_output(batch_inputs: list[T], batch_outputs: list[U]) -> None:
    """
    Validate that the batch output has the same length as the batch inputs.

    Args:
        batch_inputs (list): The batch inputs
        batch_outputs (list): The batch outputs

    Raises:
        ValueError: If the batch output length does not match the batch input length
    """
    if len(batch_inputs) != len(batch_outputs):
        msg = (f"Batch output length ({len(batch_outputs)}) "
               f"does not match batch input length ({len(batch_inputs)})")
        raise ValueError(msg)


def _ensure_batch_func(func: Callable) -> None:
    """
    Validate if a function is a BatchFunc.

    A valid BatchFunc should:
    1. Accept a single list argument
    2. Return a list

    Args:
        func (Callable): The function to validate

    Raises:
        TypeError: If the function is not a valid BatchFunc
    """
    # hints = get_type_hints(func)
    #
    # # Check if there's exactly one input argument and it's a list
    # args = list(hints.items())
    # if len(args) != 2 or args[0][0] == 'return' or not issubclass(args[0][1], list):
    #     raise TypeError("BatchFunc must accept a single list argument")
    #
    # # Check the return type
    # return_type = hints.get('return')
    # if return_type is None or not issubclass(return_type, list):
    #     raise TypeError("BatchFunc must return a list")
    pass


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0


@singledispatch
def stack(inputs, **kwargs):
    """
    Stack a list of inputs.

    Args:
        inputs (list): The list of inputs

    Returns:
        list: The stacked inputs
    """
    raise NotImplementedError(f"Cannot stack {type(inputs)}")


@singledispatch
def unstack(outputs, **kwargs):
    """
    Unstack a list of outputs.

    Args:
        outputs (list): The list of outputs

    Returns:
        list: The unstacked outputs
    """
    raise NotImplementedError(f"Cannot unstack {type(outputs)}")


