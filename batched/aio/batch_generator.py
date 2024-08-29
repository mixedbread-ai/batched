from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from batched.types import T, U
from batched.utils import batch_iter

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

    def __len__(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self._queue.qsize()

    async def extend(self, items: list[AsyncBatchItem[T, U]]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items (list[BatchItem[T, U]]): A list of items to add to the queue.
        """
        for item in items:
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
            batch_items = [self._queue._get() for _ in range(size_batches)]  # noqa: SLF001
            for batch in batch_iter(batch_items, self._batch_size):
                yield batch
