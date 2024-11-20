from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from batched.types import AsyncCache, T, U
from batched.utils import batch_iter, batch_iter_by_length, first

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable


@dataclass(order=True)
class AsyncBatchItem(Generic[T, U]):
    """
    A dataclass representing an item in the batch processing queue.

    Attributes:
        content (T): The content of the item.
        future (asyncio.Future): An asyncio.Future object representing the eventual result of processing.
        priority (int): The priority of the item in the queue. Lower values indicate higher priority.

    Type Parameters:
        T: The type of the content.
        U: The type of the result.
    """

    content: T = field(compare=False)
    future: asyncio.Future[U] = field(compare=False)
    priority: int = field(default=1, compare=True)
    _len_fn: Callable[[T], int] = field(default=None, compare=False)

    def set_result(self, result: U) -> None:
        """
        Mark the item as complete with the given result.

        Args:
            result (U): The result of processing the item.
        """
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_result(result)

    def set_exception(self, exception: Exception) -> None:
        """
        Set an exception that occurred during processing.

        Args:
            exception (Exception): The exception that occurred.
        """
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_exception(exception)

    def done(self) -> bool:
        """Check if the item's future is done."""
        return self.future.done()

    @staticmethod
    def _get_len(content: T) -> int:
        """Get the length of the content."""
        if content is None:
            return 0

        if isinstance(content, Sized):
            return len(content)

        if isinstance(content, Mapping):
            value = first(content.values())
            return AsyncBatchItem._get_len(value)

        return 1

    def __len__(self) -> int:
        """Get the length of the content."""
        if self._len_fn is None:
            self._len_fn = AsyncBatchItem._get_len
        return self._len_fn(self.content)


class AsyncBatchGenerator(Generic[T, U]):
    """
    A generator class for creating optimal batches of items.

    This class manages an asyncio priority queue of items and generates batches
    based on the specified batch size and timeout.

    Attributes:
        cache (AsyncCache[T, U] | None): An optional cache to use for storing results.
        _queue (asyncio.PriorityQueue): An asyncio priority queue to store items.
        _batch_size (int): The maximum size of each batch.
        _timeout (float): The timeout in seconds between batch generation attempts.
        _max_batch_length (int | None): Used to count the length of each item and stay within the limit.

    Type Parameters:
        T: The type of the content in the BatchItem.
        U: The type of the result in the BatchItem.
    """

    def __init__(
        self,
        batch_size: int = 32,
        *,
        timeout_ms: float = 5.0,
        cache: AsyncCache[T, U] | None = None,
        max_batch_length: int | None = None,
        sort_by_priority: bool = False,
    ) -> None:
        """
        Initialize the BatchGenerator.

        Args:
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            cache (AsyncCache[T, U] | None): An optional cache to use for storing results.
            max_batch_length (int | None): Used to count the length of each item and stay within the limit.
            sort_by_priority (bool): Whether to sort the queue by priority. Defaults to False.
        """
        self.cache = cache

        self._queue: asyncio.PriorityQueue[AsyncBatchItem[T, U]] = (
            asyncio.PriorityQueue() if sort_by_priority else asyncio.Queue()
        )
        self._batch_size = batch_size
        self._timeout = timeout_ms / 1000  # Convert to seconds
        self._max_batch_length = max_batch_length
        self._background_tasks = set()

    def __len__(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self._queue.qsize()

    def _wrap_set_result(self, item: AsyncBatchItem[T, U]) -> Callable[[U], None]:
        """Wrap the set_result method to store results in the cache."""

        def wrapper(original_set_result):
            def wrapped_set_result(result: U) -> None:
                original_set_result(result)

                if self.cache is not None:
                    task = asyncio.create_task(self.cache.set(item.content, result))
                    task.add_done_callback(self._background_tasks.discard)
                    self._background_tasks.add(task)

            return wrapped_set_result

        return wrapper(item.set_result)

    async def _check_cache(self, item: AsyncBatchItem[T, U]) -> AsyncBatchItem[T, U]:
        """Check if the item is in the cache and set the result if it is."""
        if not self.cache:
            return item

        hit = await self.cache.get(item.content)
        if hit is not None:
            item.set_result(hit)
        else:
            item.set_result = self._wrap_set_result(item)

        return item

    async def extend(self, items: list[AsyncBatchItem[T, U]]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items (list[BatchItem[T, U]]): A list of items to add to the queue.
        """
        if self.cache is None:
            for item in items:
                await self._queue.put(item)
            return

        for item in asyncio.as_completed([self._check_cache(item) for item in items]):
            result = await item
            if not result.done():
                await self._queue.put(result)

    async def optimal_batches(self) -> AsyncGenerator[list[AsyncBatchItem[T, U]], None]:
        """
        Generate optimal batches of items from the queue.

        This method continuously generates batches of items, waiting for the specified
        timeout if the queue is empty or has fewer items than the batch size.

        Yields:
            list[BatchItem[T, U]]: A batch of items from the queue.
        """
        while True:
            if self._queue.qsize() < self._batch_size:
                await asyncio.sleep(self._timeout)

            queue_size = self._queue.qsize()
            if queue_size == 0:
                continue

            n_batches = max(1, queue_size // self._batch_size)
            size_batches = min(self._batch_size * n_batches, queue_size)
            batch_items = [self._queue._get() for _ in range(size_batches)]  # noqa: SLF001

            if self._max_batch_length:
                batch_items = batch_iter_by_length(
                    batch_items, max_batch_length=self._max_batch_length, batch_size=self._batch_size
                )
            else:
                batch_items = batch_iter(batch_items, self._batch_size)

            for batch in batch_items:
                filtered = [item for item in batch if not item.done()]
                if filtered:
                    yield filtered
