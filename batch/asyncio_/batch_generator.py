from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from batch.types import T, U

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(order=True)
class Item(Generic[T, U]):
    content: T = field(compare=False)
    prioritized: bool = field(default=False, compare=False)

    future: asyncio.Future = field(default=None, compare=False)

    def complete(self, result: U) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_result(result)

    async def get_result(self) -> U:
        return await self.future

    def done(self) -> bool:
        return self.future.done()

    def set_exception(self, exception: Exception) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_exception(exception)


class BatchGenerator(Generic[T, U]):
    def __init__(
        self,
        batch_size: int = 32,
        timeout: float = 5.0,
    ) -> None:
        self._queue: list[Item[T, U]] = []
        self._batch_size = batch_size
        self._timeout = timeout / 1000

    def __len__(self) -> int:
        return len(self._queue)

    def extend(self, items: list[Item[T, U]]) -> None:
        self._queue.extend(items)

    async def optimal_batches(self) -> Generator[list[Item[T, U]], None, None]:
        while True:
            if not self._queue:
                await asyncio.sleep(self._timeout)
                continue

            if len(self._queue) < self._batch_size:
                await asyncio.sleep(self._timeout)

            n_batches = max(1, len(self._queue) // self._batch_size)
            size_batches = self._batch_size * n_batches
            new_items_l = self._queue[:size_batches]
            self._queue = self._queue[size_batches:]

            for i in range(n_batches):
                mini_batch = new_items_l[self._batch_size * i: self._batch_size * (i + 1)]
                mini_batch = [item for item in mini_batch if not item.done()]
                if mini_batch:
                    yield mini_batch
