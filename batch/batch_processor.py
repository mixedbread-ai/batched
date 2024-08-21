import threading
import time
from collections.abc import Callable
from typing import Generic

from batch.batch_generator import BatchGenerator, Item
from batch.decorator import _dynamic_batch
from batch.types import BatchFunc, BatchProcessorStats, T, U, _validate_batch_output


class BatchProcessor(Generic[T, U]):
    def __init__(
        self,
        _func: BatchFunc[T, U],
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
    ):
        self.batch_func = _func

        self.batch_queue = BatchGenerator(batch_size, timeout_ms)
        self.small_batch_threshold = small_batch_threshold

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[T]) -> list[bool]:
        prioritized = len(items) <= self.small_batch_threshold
        return [prioritized] * len(items)

    def start(self):
        with self._lock:
            if self._running:
                return
            self._thread = threading.Thread(target=self._process_batches, daemon=True)
            self._thread.start()
            self._running = True

    def shutdown(self):
        with self._lock:
            if not self._running:
                return
            self._running = False

        if self._thread:
            self._thread.join()

    def __call__(self, items: list[T]) -> list[U]:
        if not self._running:
            self.start()

        return self._schedule(items)

    def _schedule(self, items: list[T]) -> list[U]:
        prioritized = self.prioritize(items)

        new_priority_queue = [
            Item[T, U](
                content=item,
                prioritized=prio,
            )
            for item, prio in zip(items, prioritized)
        ]

        self.batch_queue.extend(new_priority_queue)
        return [item.get_result() for item in new_priority_queue]

    def _process_batches(self) -> None:
        for batch in self.batch_queue.optimal_batches():
            if not self._running:
                break

            start_time = time.time()

            try:
                batch_inputs = [item.content for item in batch]
                batch_outputs = self.batch_func(batch_inputs)

                _validate_batch_output(batch_inputs, batch_outputs)

                for item, output in zip(batch, batch_outputs):
                    item.complete(output)

            except Exception as e:
                for item in batch:
                    item.set_exception(e)

            finally:
                processing_time = time.time() - start_time
                self._stats.update(len(batch), processing_time)

    @property
    def stats(self):
        return self._stats.clone(queue_size=len(self.batch_queue))


def dynamically(
        func: BatchFunc | None = None,
        /,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
) -> Callable:
    def make_processor(_func: BatchFunc) -> BatchProcessor:
        return BatchProcessor(
            _func,
            batch_size,
            timeout_ms,
            small_batch_threshold
        )

    return _dynamic_batch(make_processor, func)
