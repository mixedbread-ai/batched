"""Async batch generator implementation with lock-free optimization."""
# ruff: noqa: PERF203

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Mapping, Sized
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from batched.types import AsyncCache, T, U
from batched.utils import batch_iter_by_length, first

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable


@dataclass(order=True)
class AsyncBatchItem(Generic[T, U]):
    """
    Represents an item in the batch processing queue.

    Attributes:
        content: The item's content.
        future: An asyncio.Future for the result.
        priority: Item priority (lower is higher).
        _len_fn: Function to calculate length, if needed.
    """

    content: T = field(compare=False)
    future: asyncio.Future[U] = field(compare=False)
    priority: int = field(default=1, compare=True)
    _len_fn: Callable[[T], int] | None = field(default=None, compare=False)

    def set_result(self, result: U) -> None:
        """Completes the item's future with a result."""
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_result(result)

    def set_exception(self, exception: Exception) -> None:
        """Sets an exception on the item's future."""
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_exception(exception)

    def done(self) -> bool:
        """Checks if the item's future is done."""
        return self.future.done()

    @staticmethod
    def _get_len(content: T) -> int:
        """Calculates the length of the content, supporting Sized and Mappings."""
        if content is None:
            return 0

        if isinstance(content, Sized):
            return len(content)

        if isinstance(content, Mapping):
            return AsyncBatchItem._get_len(first(content.values()))

        return 1

    def __len__(self) -> int:
        """Returns the length of the content."""
        if self._len_fn is None:
            self._len_fn = AsyncBatchItem._get_len

        return self._len_fn(self.content)


class AsyncBatchGenerator(Generic[T, U]):
    """
    Lock-free batch generator using multiple queues for better concurrency.

    This implementation uses multiple internal queues to reduce contention
    and improve performance under high concurrency. Items are distributed
    across queues using round-robin, and collection happens without locks.

    Attributes:
        cache: Optional cache for storing results.
        _batch_size: Maximum size of each batch.
        _timeout: Timeout in seconds between batch generation attempts.
        _max_batch_length: Maximum total length of items in a batch.
        _num_queues: Number of internal queues (default: 4).
        _queues: List of asyncio queues.
        _producer_counter: Counter for round-robin distribution.
        _consumer_counter: Counter for round-robin collection.
        _background_tasks: Set of background cache tasks.

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
        num_queues: int = 4,
    ) -> None:
        """
        Initialize the AsyncBatchGenerator.

        Args:
            batch_size: Maximum size of each batch. Defaults to 32.
            timeout_ms: Timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            cache: Optional cache to use for storing results.
            max_batch_length: Maximum total length of items in a batch.
            sort_by_priority: Whether to sort by priority (Note: priority queues reduce performance).
            num_queues: Number of internal queues for lock-free operation. Defaults to 4.
        """
        self.cache = cache
        self._batch_size = batch_size
        self._timeout = timeout_ms / 1000  # Convert to seconds
        self._max_batch_length = max_batch_length
        self._sort_by_priority = sort_by_priority
        self._background_tasks = set()

        # Lock-free implementation with multiple queues
        self._num_queues = num_queues
        if sort_by_priority:
            # Fall back to single priority queue if sorting is required
            self._num_queues = 1
            self._queues = [asyncio.PriorityQueue(maxsize=100000)]
        else:
            # Use multiple regular queues for lock-free operation
            self._queues = [asyncio.Queue(maxsize=25000) for _ in range(num_queues)]

        self._producer_counter = 0
        self._consumer_counter = 0

        # Pre-allocated buffers for better performance
        self._buffer_pool = deque([[] for _ in range(100)])

    def __len__(self) -> int:
        """
        Get the current total size across all queues.

        Returns:
            The total number of items in all queues.
        """
        return sum(q.qsize() for q in self._queues)

    def _cache_result(self, item: AsyncBatchItem[T, U], result: U) -> None:
        """Stores result in the cache if enabled."""
        if self.cache:
            task = asyncio.create_task(self.cache.set(item.content, result))
            task.add_done_callback(self._background_tasks.discard)
            self._background_tasks.add(task)

    async def _check_cache_and_set_result(self, item: AsyncBatchItem[T, U]) -> AsyncBatchItem[T, U]:
        """Checks cache and sets result if cached, otherwise prepares for set."""
        if not self.cache:
            return item

        cached_result = await self.cache.get(item.content)
        if cached_result is not None:
            item.set_result(cached_result)
        else:
            original_set_result = item.set_result

            def set_result_with_cache(result: U) -> None:
                original_set_result(result)
                self._cache_result(item, result)

            item.set_result = set_result_with_cache

        return item

    def _get_buffer(self) -> list:
        """Get a pre-allocated buffer from the pool."""
        if self._buffer_pool:
            return self._buffer_pool.popleft()
        return []

    def _return_buffer(self, buffer: list) -> None:
        """Return a buffer to the pool after clearing it."""
        buffer.clear()
        if len(self._buffer_pool) < 100:
            self._buffer_pool.append(buffer)

    async def extend(self, items: list[AsyncBatchItem[T, U]]) -> None:
        """
        Add multiple items to the queue using round-robin distribution.

        Args:
            items: A list of items to add to the queues.
        """
        if not items:
            return

        # Handle cache checking if enabled
        if self.cache:
            items_to_queue = []
            for item in items:
                cached_item = await self._check_cache_and_set_result(item)
                if not cached_item.done():
                    items_to_queue.append(cached_item)
            items = items_to_queue

        # Distribute items across queues using round-robin
        for item in items:
            queue_idx = self._producer_counter % self._num_queues
            self._producer_counter += 1
            await self._queues[queue_idx].put(item)

    async def optimal_batches(self) -> AsyncGenerator[list[AsyncBatchItem[T, U]], None]:  # noqa: C901
        """
        Generate optimal batches of items from the queues.

        This method continuously generates batches by collecting items from
        multiple queues in a lock-free manner, improving performance under
        high concurrency.

        Yields:
            Batches of items from the queues.
        """
        batch_size = self._batch_size
        timeout = self._timeout
        queues = self._queues
        num_queues = self._num_queues
        max_batch_length = self._max_batch_length

        while True:
            buffer = self._get_buffer()

            try:
                # Fast path: try to collect items without waiting
                attempts = 0
                while len(buffer) < batch_size and attempts < num_queues * 2:
                    queue_idx = self._consumer_counter % num_queues
                    self._consumer_counter += 1
                    queue = queues[queue_idx]
                    attempts += 1

                    # Try to get items without blocking
                    items_to_get = min(batch_size - len(buffer), queue.qsize())
                    for _ in range(items_to_get):
                        try:
                            item = queue.get_nowait()
                            if not item.done():
                                buffer.append(item)
                        except asyncio.QueueEmpty:
                            break

                    if len(buffer) >= batch_size:
                        break

                # If we have a partial batch, wait briefly for more items
                if buffer and len(buffer) < batch_size:
                    await asyncio.sleep(min(timeout, 0.001))

                    # One more sweep to collect any new items
                    for queue in queues:
                        items_to_get = min(batch_size - len(buffer), queue.qsize())
                        for _ in range(items_to_get):
                            try:
                                item = queue.get_nowait()
                                if not item.done():
                                    buffer.append(item)
                            except asyncio.QueueEmpty:
                                break

                        if len(buffer) >= batch_size:
                            break

                # If no items yet, wait for at least one
                if not buffer:
                    # Try each queue with timeout
                    item_found = False
                    for queue in queues:
                        if not queue.empty() or not item_found:
                            try:
                                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                                if not item.done():
                                    buffer.append(item)
                                    item_found = True
                            except asyncio.TimeoutError:
                                continue

                    # If still no items, sleep and continue
                    if not buffer:
                        await asyncio.sleep(timeout)
                        continue

                # Apply max_batch_length if specified
                if buffer:
                    if max_batch_length:
                        # Group by length constraints
                        batches = list(
                            batch_iter_by_length(buffer, max_batch_length=max_batch_length, batch_size=batch_size)
                        )
                        for batch in batches:
                            if batch:
                                yield list(batch)
                    else:
                        yield list(buffer)

            finally:
                self._return_buffer(buffer)
