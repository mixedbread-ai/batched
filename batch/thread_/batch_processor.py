import threading
import time
from functools import partial
from typing import Generic

from batch.decorator import _dynamic_batch
from batch.thread_.batch_generator import BatchGenerator, Item
from batch.types import BatchFunc
from batch.types import BatchProcessorStats, _ensure_batch_func, _validate_batch_output
from batch.types import T, U


class BatchProcessor(Generic[T, U]):
    def __init__(
            self,
            _func: BatchFunc[T, U],
            batch_size: int = 32,
            timeout: float = 5.0,
            small_batch_threshold: int = 8,
    ):
        _ensure_batch_func(_func)
        self.batch_func = _func

        self.batch_queue = BatchGenerator(batch_size, timeout)
        self.small_batch_threshold = small_batch_threshold

        self._running = False
        self._thread = None
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[T]) -> list[bool]:
        prioritized = len(items) <= self.small_batch_threshold
        return [prioritized] * len(items)

    def start(self):
        with self._start_lock:
            if self._running:
                return
            self._thread = threading.Thread(target=self._process_batches, daemon=True)
            self._thread.start()
            self._running = True

    def shutdown(self):
        if self._running:
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

                _validate_batch_output(batch_inputs,batch_outputs)

                for item, output in zip(batch, batch_outputs):
                    item.complete(output)

            except Exception as e:
                for item in batch:
                    item.set_exception(e)

            finally:
                processing_time = time.time() - start_time
                self._update_stats(len(batch), processing_time)

    def _update_stats(self, batch_size: int, processing_time: float):
        self._stats.total_processed += batch_size
        self._stats.total_batches += 1
        self._stats.avg_batch_size = self._stats.total_processed / self._stats.total_batches
        self._stats.avg_processing_time = (
              self._stats.avg_processing_time * (
                  self._stats.total_batches - 1) + processing_time
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




