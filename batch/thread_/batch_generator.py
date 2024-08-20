from __future__ import annotations

import time
from dataclasses import dataclass, field
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING, Generic

from batch.types import T, U

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(order=True)
class Item(Generic[T, U]):
    content: T = field(compare=False)
    prioritized: bool = field(default=False, compare=False)

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
    def __init__(
        self,
        batch_size: int = 32,
        timeout: float = 5.0,
    ) -> None:
        self._queue = Queue()
        self._batch_size = batch_size
        self._timeout = timeout / 1000

    def __len__(self) -> int:
        return self._queue.qsize()

    def extend(self, items: list[Item[T, U]]) -> None:
        for item in items:
            self._queue.put(item)

    def optimal_batches(self) -> Generator[list[Item[T, U]], None, None]:
        while True:
            queue_size = self._queue.qsize()
            if queue_size == 0:
                time.sleep(self._timeout)
                continue
            elif queue_size < self._batch_size:
                time.sleep(self._timeout)

            n_batches = max(1, queue_size // self._batch_size)
            size_batches = self._batch_size * n_batches

            new_items_l = []
            for _ in range(size_batches):
                if not self._queue.empty():
                    new_items_l.append(self._queue.get())

            for i in range(n_batches):
                mini_batch = new_items_l[self._batch_size * i: self._batch_size * (i + 1)]
                if mini_batch:
                    yield mini_batch

