import asyncio
import threading
from functools import partial
from typing import Generic

from batch.asyncio_.batch_generator import BatchGenerator, Item
from batch.decorator import _dynamic_batch
from batch.types import BatchFunc, _ensure_batch_func, _validate_batch_output
from batch.types import BatchProcessorStats
from batch.types import T, U
from batch.utils import ensure_async


class BatchProcessor(Generic[T, U]):
    def __init__(
            self,
            _func: BatchFunc[T, U],
            batch_size: int = 32,
            timeout: float = 5.0,
            small_batch_threshold: int = 8,
    ):
        _ensure_batch_func(_func)
        self.batch_func = ensure_async(_func)

        self.batch_queue = BatchGenerator[T, U](batch_size, timeout)
        self.small_batch_threshold = small_batch_threshold

        self._running = False
        self._loop = None
        self._thread = None
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[T]) -> list[bool]:
        prioritized = len(items) <= self.small_batch_threshold
        return [prioritized] * len(items)

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._process_batches())
        self._start_event.set()
        self._loop.run_forever()

    def start(self):
        with self._start_lock:
            if self._running:
                return
            self._thread = threading.Thread(
                target=self._start_loop,
                daemon=True,
            )
            self._thread.start()
            self._start_event.wait()
            self._running = True

    def shutdown(self):
        if self._running:
            self._running = False
            if self._loop:
                self._loop.call_soon_threadsafe(
                    self._loop.stop
                )
            if self._thread:
                self._thread.join()

    def __call__(self, items: list[T]) -> list[U]:
        if not self._running:
            self.start()

        return asyncio.run_coroutine_threadsafe(
            self._schedule(items), self._loop
        ).result()

    async def _schedule(self, items: list[T]) -> list[U]:
        prioritized = self.prioritize(items)

        new_priority_queue = [
            Item[T, U](
                content=item,
                prioritized=prio,
                future=self._loop.create_future(),
            )
            for item, prio in zip(items, prioritized)
        ]

        self.batch_queue.extend(new_priority_queue)
        futures = [item.future for item in new_priority_queue]
        return [await asyncio.gather(*futures)]

    async def _process_batches(self) -> None:
        async for batch in self.batch_queue.optimal_batches():
            if not self._running:
                break

            start_time = self._loop.time()

            try:
                batch_inputs = [item.content for item in batch]
                batch_outputs = await self.batch_func(batch_inputs)

                _validate_batch_output(batch_inputs, batch_outputs)

                for item, output in zip(batch, batch_outputs):
                    item.complete(output)

            except Exception as e:
                for item in batch:
                    item.set_exception(e)

            finally:
                processing_time = self._loop.time() - start_time
                self._update_stats(len(batch), processing_time)

    def _update_stats(self, batch_size: int, processing_time: float):
        self._stats.total_processed += batch_size
        self._stats.total_batches += 1
        self._stats.avg_batch_size = self._stats.total_processed / self._stats.total_batches
        self._stats.avg_processing_time = (
            self._stats.avg_processing_time * (self._stats.total_batches - 1) + processing_time
        ) / self._stats.total_batches

    @property
    def stats(self):
        self._stats.queue_size = len(self.batch_queue)
        return self._stats


def _make_processor(func: BatchFunc, **kwargs) -> BatchProcessor:
    batch_size = kwargs.pop('batch_size', 32)
    timeout = kwargs.pop('timeout', 5.0)
    small_batch_threshold = kwargs.pop('small_batch_threshold', 8)

    return BatchProcessor(func, batch_size, timeout, small_batch_threshold)


dynamically = partial(_dynamic_batch, _make_processor)







