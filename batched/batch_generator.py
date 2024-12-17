from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import TYPE_CHECKING, Generic

from batched.types import T, U
from batched.utils import batch_iter

if TYPE_CHECKING:
    from collections.abc import Generator


from concurrent.futures import Future, InvalidStateError


@dataclass(order=True)
class BatchItem(Generic[T, U]):
    """
    A dataclass representing an item in the batch processing queue.

    Attributes:
        content (T): The content of the item.
        priority (int): The priority of the item in the queue. Lower values indicate higher priority.
        future (Future): A Future object representing the eventual result of processing.

    Type Parameters:
        T: The type of the content.
        U: The type of the result.
    """

    content: T = field(compare=False)
    priority: int = field(default=1, compare=True)
    future: Future = field(default_factory=Future, compare=False)

    def to_awaitable(self) -> asyncio.Future[U]:
        """Await the result of the item's future."""
        return asyncio.wrap_future(self.future)

    def set_result(self, result: U) -> None:
        """
        Mark the item as complete with the given result.

        Args:
            result (U): The result of processing the item.
        """
        with contextlib.suppress(InvalidStateError):
            self.future.set_result(result)

    def set_exception(self, exception: Exception) -> None:
        """
        Set an exception that occurred during processing.

        Args:
            exception (Exception): The exception that occurred.
        """
        with contextlib.suppress(InvalidStateError):
            self.future.set_exception(exception)

    def result(self) -> U:
        """
        Get the result of processing the item.

        Returns:
            U: The result of processing.

        Raises:
            Exception: If an exception occurred during processing.
        """
        return self.future.result()

    def done(self) -> bool:
        """Check if the item's future is done."""
        return self.future.done()


class BatchGenerator(Generic[T, U]):
    """
    A generator class for creating optimal batches of items.

    This class manages a priority queue of items and generates batches
    based on the specified batch size and timeout.

    Attributes:
        _queue (PriorityQueue): A priority queue to store items.
        _batch_size (int): The maximum size of each batch.
        _timeout (float): The timeout in seconds between batch generation attempts.
        _stop_requested (bool): Flag to indicate if the generator should stop.

    Type Parameters:
        T: The type of the content in the Item.
        U: The type of the result in the Item.
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
        self._queue = PriorityQueue()
        self._batch_size = batch_size
        self._timeout = timeout_ms / 1000
        self._stop_requested = False

    def __len__(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self._queue.qsize()

    def extend(self, items: list[BatchItem[T, U]]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items (list[Item[T, U]]): A list of items to add to the queue.
        """
        for item in items:
            self._queue.put(item)

    def optimal_batches(self) -> Generator[list[BatchItem[T, U]], None, None]:
        """
        Generate optimal batches of items from the queue.

        This method continuously generates batches of items, waiting for the specified
        timeout if the queue is empty or has fewer items than the batch size.

        Yields:
            list[Item[T, U]]: A batch of items from the queue.
        """
        while not self._stop_requested:
            if self._queue.qsize() < self._batch_size:
                time.sleep(self._timeout)

            queue_size = self._queue.qsize()
            if queue_size == 0:
                continue

            n_batches = max(1, queue_size // self._batch_size)
            size_batches = min(self._batch_size * n_batches, queue_size)
            batch_items = [self._queue.get() for _ in range(size_batches)]
            for batch in batch_iter(batch_items, self._batch_size):
                if self._stop_requested:
                    break

                yield batch

    def stop(self):
        """
        Request the generator to stop.

        This method sets the _stop_requested flag to True, which will cause the
        optimal_batches generator to exit its loop on the next iteration.
        """
        self._stop_requested = True
