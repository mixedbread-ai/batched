from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator

    from ofen.types import ModelFeatures, NDArrayOrTensor


@dataclass(order=True)
class InferenceItem:
    size: int
    future: asyncio.Future = field(compare=False)
    content: ModelFeatures = field(compare=False)
    prioritized: bool = field(default=False, compare=False)

    def complete(self, result: NDArrayOrTensor | list[NDArrayOrTensor]) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_result(result)

    async def get_result(self) -> Awaitable[NDArrayOrTensor | list[NDArrayOrTensor]]:
        return await self.future

    def done(self) -> bool:
        return self.future.done()

    def set_exception(self, exception: Exception) -> None:
        with contextlib.suppress(asyncio.InvalidStateError):
            self.future.set_exception(exception)


class BatchGenerator:
    def __init__(
        self,
        batch_size: int = 32,
        timeout: float = 5.0,
    ) -> None:
        self._queue: list[InferenceItem] = []
        self._batch_size = batch_size
        self._timeout = timeout / 1000
        self._lock = Lock()

    def __len__(self) -> int:
        return len(self._queue)

    def extend(self, items: list[InferenceItem]) -> None:
        with self._lock:
            self._queue.extend(items)

    async def optimal_batches(self) -> Generator[list[InferenceItem], None]:
        while True:
            if len(self._queue) == 0:
                await asyncio.sleep(self._timeout)
                continue
            elif len(self._queue) < self._batch_size:
                await asyncio.sleep(self._timeout)

            n_batches = max(1, len(self._queue) // self._batch_size)
            size_batches = self._batch_size * n_batches
            with self._lock:
                new_items_l = self._queue[:size_batches]
                self._queue = self._queue[size_batches:]

            for i in range(n_batches):
                mini_batch = new_items_l[self._batch_size * i : self._batch_size * (i + 1)]
                mini_batch = [item for item in mini_batch if not item.done()]
                if mini_batch:
                    yield mini_batch
