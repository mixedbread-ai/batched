from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import PriorityQueue
from threading import Event
from typing import TYPE_CHECKING, Generic

from batch.types import T, U

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(order=True)
class Item(Generic[T, U]):
    content: T = field(compare=False)
    prioritized: bool = field(default=False, compare=True)

    result: U = field(default=None, compare=False)
    event: Event = field(default_factory=Event, compare=False)
    exception: Exception = field(default=None, compare=False)

    def complete(self, result: U) -> None:
        self.result = result
        self.event.set()

    def get_result(self) -> U:
        self.event.wait()
        if self.exception:
            raise self.exception
        return self.result

    def set_exception(self, exception: Exception) -> None:
        self.exception = exception
        self.event.set()


class BatchGenerator(Generic[T, U]):
    """
    A generator class for creating optimal batches of items.

    This class manages a priority queue of items and generates batches
    based on the specified batch size and timeout.

    Attributes:
        _queue (PriorityQueue): A priority queue to store items.
        _batch_size (int): The maximum size of each batch.
        _timeout (float): The timeout in seconds between batch generation attempts.

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

    def __len__(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            int: The number of items in the queue.
        """
        return self._queue.qsize()

    def extend(self, items: list[Item[T, U]]) -> None:
        """
        Add multiple items to the queue.

        Args:
            items (list[Item[T, U]]): A list of items to add to the queue.
        """
        for item in items:
            self._queue.put(item)

    def optimal_batches(self) -> Generator[list[Item[T, U]], None, None]:
        """
        Generate optimal batches of items from the queue.

        This method continuously generates batches of items, waiting for the specified
        timeout if the queue is empty or has fewer items than the batch size.

        Yields:
            list[Item[T, U]]: A batch of items from the queue.
        """
        while True:
            queue_size = self._queue.qsize()

            if queue_size == 0:
                time.sleep(self._timeout)
                continue

            elif queue_size < self._batch_size:
                time.sleep(self._timeout)

            n_batches = max(1, queue_size // self._batch_size)
            size_batches = self._batch_size * n_batches

            new_items_l = [self._queue.get() for _ in range(size_batches) if not self._queue.empty()]

            for i in range(n_batches):
                mini_batch = new_items_l[self._batch_size * i : self._batch_size * (i + 1)]
                if mini_batch:
                    yield mini_batch
