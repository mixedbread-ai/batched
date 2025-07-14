from __future__ import annotations

import asyncio
import functools
import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Generator, Sequence, TypeVar

from batched.types import AsyncCache, CacheStats

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

T = TypeVar("T")
U = TypeVar("U")


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


class AsyncDiskCache(AsyncCache[T, U]):
    """An asynchronous disk-based cache implementation."""

    def __init__(
        self,
        cache_dir: str = "~/.cache/batched",
        *,
        max_threads: int = 16,
        collect_stats: bool = False,
        expiration_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        with ensure_import("diskcache"):
            import diskcache  # noqa: PLC0415

        path = Path(cache_dir).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)

        kwargs = {"statistics": False, **kwargs}
        self._cache = diskcache.Cache(str(path), **kwargs)
        self._get_pool = ThreadPoolExecutor(max_workers=max_threads)
        self._set_pool = ThreadPoolExecutor(max_workers=max_threads)
        self._stats = CacheStats()
        self._collect_stats = collect_stats
        self._expiration_seconds = expiration_seconds

    async def _to_thread(self, pool: ThreadPoolExecutor, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a function in a thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(pool, functools.partial(func, *args, **kwargs))

    def _get(self, key: T) -> U | None:
        key = self._get_key(key)
        return self._cache.get(key)

    async def get(self, key: T) -> U | None:
        """Get a value from the cache.

        Args:
            key: The key to look up

        Returns:
            The cached value if found, otherwise None
        """
        start_time = time.perf_counter()
        result = await self._to_thread(self._get_pool, self._get, key)

        if self._collect_stats:
            self._stats.update_get(hit=result is not None, time_taken=time.perf_counter() - start_time)

        return result

    def _set(self, key: T, value: U) -> None:
        """Set a value in the cache."""
        key = self._get_key(key)
        self._cache.set(key, value, expire=self._expiration_seconds)

    async def set(self, key: T, value: U) -> None:
        """Set a value in the cache.

        Args:
            key: The key to store under
            value: The value to cache
        """
        start_time = time.perf_counter()
        await self._to_thread(self._set_pool, self._set, key, value)

        if self._collect_stats:
            self._stats.update_set(time_taken=time.perf_counter() - start_time)

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        await asyncio.to_thread(self._cache.clear)

    def clear_stats(self) -> None:
        """Reset the cache statistics."""
        self._stats = CacheStats()

    def stats(self) -> CacheStats:
        return self._stats

    def _get_key(self, key: T):
        """Get the cache key from the input key."""
        return key
