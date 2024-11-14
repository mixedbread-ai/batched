from __future__ import annotations

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import functools
import inspect
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Generator,
    Sequence,
    TypeVar,
    runtime_checkable,
    Tuple
)
import time

from batched.types import AsyncCache, BatchProcessorStats, Cache, BatchProcessorCacheStats

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")


@contextmanager
def ensure_import(install_name: str | None = None):
    """Ensure a module is imported, raising a meaningful error if not.

    Args:
        install_name (Optional[str]): Name of the package to install if import fails.

    Raises:
        ImportError: If the module cannot be imported.

    """
    try:
        yield
    except ImportError as e:
        module_name = str(e).split("'")[1]
        install_name = install_name or module_name
        msg = (
            f"Failed to import {module_name}. This is required for this feature. "
            f"Please install it using: 'pip install {install_name}'"
        )
        raise ImportError(msg) from e


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


def batch_iter_by_length(
    seq: Sequence[T], max_batch_length: int, batch_size: int
) -> Generator[Sequence[T], None, None]:
    """Yield batches from a sequence, respecting a maximum batch length."""
    batch = []
    current_length = 0

    for item in seq:
        item_length = len(item)

        if batch and (len(batch) >= batch_size or (current_length + item_length > max_batch_length)):
            yield batch
            batch = []
            current_length = 0

        batch.append(item)
        current_length += item_length

    if batch:
        yield batch


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


T = TypeVar("T")
U = TypeVar("U")


class AsyncDiskCache(AsyncCache[T, U]):
    def __init__(self, directory: str = "/tmp/batched", n_threads: int = 16, **kwargs):
        with ensure_import("diskcache"):
            import diskcache

        self._cache = diskcache.Cache(directory, **kwargs)
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    async def get(self, key: T) -> U | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, self._cache.get, key)

    async def set(self, key: T, value: U) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._pool, self._cache.set, key, value)


class AsyncMemoryCache(AsyncCache[T, U]):
    def __init__(self, maxsize: int = 10000, statistics: bool = True) -> None:
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._stats = BatchProcessorCacheStats()
        self._record_stats = statistics

    async def get(self, key: T) -> U | None:
        t0 = time.time()
        hit = self._cache.get(self._get_key(key))
        if hit is not None:
            self._cache.move_to_end(self._get_key(key))
        if self._record_stats:
            self._stats.update_get(hit is not None, time.time()-t0)
        return hit

    async def set(self, key: T, value: U) -> None:
        t0 = time.time()
        item_popped = False
        if len(self._cache) >= self._maxsize:
            item_popped = True
            self._cache.popitem(last=False)
        self._cache[self._get_key(key)] = value
        self._cache.move_to_end(self._get_key(key))
        if self._record_stats:
            self._stats.update_set(len(self._cache), self._maxsize, item_popped, time.time()-t0)

    def _restart_stats(self) -> None:
        self._stats = BatchProcessorCacheStats()

    def stats(self, processor_stats: BatchProcessorStats | None = None, enable: bool = True, reset: bool = False) -> BatchProcessorCacheStats:
        """
        This function returns the running statistics of the cache.
        If BatchProcessorStats is passed the latency reduction that the cache provided is calculated.

        Args:
            processor_stats (BatchProcessorStats | None): The BatchProcessorStats of the batched endpoint. If passed the BatchProcessorCacheStats will include the latency_reduction calculation.
            enable (bool): True if you want to continue to record statistics. False otherwise.
            reset (bool): True if you want to set to 0 the statistics recorded. False otherwise.

        Returns:
            BatchProcessorCacheStats: A dataclass containing the relevant statistics of the cache.
        """
        self._record_stats = enable
        stats = self._stats.get_stats(cache=self._cache, maxsize=self._maxsize, batch_processor_stats=processor_stats)
        if reset:
            self._restart_stats()
        return stats


    def _get_key(self, key: T) -> str:
        return str(key)
