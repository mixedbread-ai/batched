from __future__ import annotations

import asyncio
import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Generator, Sequence, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")


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


def is_method(func: Callable) -> bool:
    """Check if a callable is a method.

    Args:
        func (Callable): The function to check.

    Returns:
        bool: True if the callable is a method, False otherwise.

    """
    if not callable(func):
        return False

    params = inspect.signature(func).parameters
    return len(params) > 1 and first(params) in ("self", "cls")


def ensure_async(func: Callable[..., T] | Coroutine[Any, Any, T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Ensure a function is async.

    This function takes any callable (sync or async) or coroutine and returns an async callable.
    If the input is already a coroutine function, it's returned as-is.
    Otherwise, it wraps the sync function to run in a separate thread.

    Args:
        func (Callable[..., T] | Coroutine[Any, Any, T]): The function or coroutine to ensure is async.

    Returns:
        Callable[..., Coroutine[Any, Any, T]]: An async callable with the same return type as the input function.

    """
    if asyncio.iscoroutinefunction(func):
        return func

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create an event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def batch_iter(seq: Sequence[T], batch_size: int) -> Generator[Sequence[T], None, None]:
    """Yield batches from a sequence.

    Args:
        seq (Sequence[T]): The sequence to batch.
        batch_size (int): The size of each batch.

    Yields:
        Sequence[T]: Batches of the input sequence.

    """
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def bucket_batch_iter(
    data: list[T], batch_size: int, *, descending: bool = True
) -> Generator[tuple[list[T], list[int]], None, None]:
    """Iterate over data in batches, sorted by length.

    This function sorts the input data by length, then yields batches of the specified size.
    It's useful for processing sequences of varying lengths efficiently.

    Args:
        data (List[T]): List of items to batch.
        batch_size (int): Number of items per batch.
        descending (bool): If True, sort by length in descending order. Defaults to True.

    Yields:
        Tuple[List[T], List[int]]: A tuple containing a batch of items and their original indices.

    """
    if len(data) <= batch_size:
        yield data, list(range(len(data)))
        return

    data_with_indices = sorted(enumerate(data), key=lambda x: len(x[1]), reverse=descending)
    for batch in batch_iter(data_with_indices, batch_size):
        indices, items = zip(*batch)
        yield list(items), list(indices)
