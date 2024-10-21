from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from batched import utils
from batched.types import T, U
from batched.utils import Cache, batch_iter, batch_iter_by_length

if TYPE_CHECKING:
    from collections.abc import Generator


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

    def __len__(self) -> int:
        """Get the length of the content."""
        return AsyncBatchItem._get_len(self.content)


class AsyncBatchGenerator(Generic[T, U]):
    """
    A generator class for creating optimal batches of items.

    This class manages an asyncio priority queue of items and generates batches
    based on the specified batch size and timeout.

    Attributes:
        _queue (asyncio.PriorityQueue): An asyncio priority queue to store items.
        _batch_size (int): The maximum size of each batch.
        _timeout (float): The timeout in seconds between batch generation attempts.

    Type Parameters:
        T: The type of the content in the BatchItem.
        U: The type of the result in the BatchItem.
    """

    def __init__(
        self,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        n_threads: int = 8,
        cache: utils.Cache[T, U] | None = None,
        max_batch_length: int | None = None,
    ) -> None:
        """
        Initialize the BatchGenerator.

        Args:
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
        """
        self._queue: asyncio.PriorityQueue[AsyncBatchItem[T, U]] = asyncio.PriorityQueue()
        self._batch_size = batch_size
        self._timeout = timeout_ms / 1000  # Convert to seconds
        self._max_batch_length = max_batch_length

        self._cache = cache
        self._pool = ThreadPoolExecutor(max_workers=n_threads) if cache else None

    def __len__(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self._queue.qsize()

    def _check_cache(self, item: AsyncBatchItem[T, U]) -> AsyncBatchItem[T, U]:
        hit = self._cache.get(item.content)
        if hit is not None:
            item.set_result(hit)
        else:
            self._wrap_set_result(item)

        return item

    def _wrap_set_result(self, item: AsyncBatchItem[T, U]) -> None:
        original_set_result = item.set_result

        def wrapped_set_result(result: U) -> None:
            if self._cache is not None:
                self._pool.submit(self._cache.set, item.content, result)
            original_set_result(result)

        item.set_result = wrapped_set_result

    async def extend(self, items: list[AsyncBatchItem[T, U]]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items (list[BatchItem[T, U]]): A list of items to add to the queue.
        """
        if self._cache is None:
            for item in items:
                await self._queue.put(item)
            return

        items = list(self._pool.map(self._check_cache, items))
        for item in items:
            if item.done():
                continue
            await self._queue.put(item)

    async def optimal_batches(self) -> Generator[list[AsyncBatchItem[T, U]], None, None]:
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
            batch_items = [await self._queue.get() for _ in range(size_batches)]

            if self._max_batch_length is not None:
                batch_items = batch_iter_by_length(
                    batch_items, max_batch_length=self._max_batch_length, batch_size=self._batch_size
                )
            else:
                batch_items = batch_iter(batch_items, self._batch_size)

            for batch in batch_items:
                yield batch
