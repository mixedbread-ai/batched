from __future__ import annotations

import threading
import time
from functools import partial
from typing import overload

from batch.decorator import _dynamic_batch
from batch.inference.types import ModelFeatures, ModelOutputs, BatchInfer, stack_features, unstack_features, stack_outputs, unstack_outputs
from batch.thread_.batch_generator import Item, BatchGenerator
from batch.types import BatchProcessorStats


class BatchProcessor:
    def __init__(
        self,
        _func: BatchInfer,
        batch_size: int = 32,
        timeout: float = 5.0,
        small_batch_threshold: int = 8,
        pad_tokens: dict[str, int] | None = None,

    ):
        self.batch_func = _func

        self.batch_queue = BatchGenerator(batch_size, timeout)
        self.small_batch_threshold = small_batch_threshold
        self.pad_tokens = pad_tokens or {}

        self._running = False
        self._thread = None
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[ModelFeatures]) -> list[bool]:
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

    @overload
    def __call__(self, features: ModelFeatures, /) -> ModelOutputs:
        ...

    @overload
    def __call__(self, **features: ModelFeatures) -> ModelOutputs:
        ...

    def __call__(self, *args, **kwargs) -> ModelOutputs:
        if not self._running:
            self.start()

        features = args[0] if args else kwargs

        return self._schedule(features)

    def _schedule(self, features: ModelFeatures) -> ModelOutputs:
        items = unstack_features(features)
        prioritized = self.prioritize(items)

        new_priority_queue = [
            Item(content=item, prioritized=prio)
            for item, prio in zip(items, prioritized)
        ]

        self.batch_queue.extend(new_priority_queue)
        results = [item.get_result() for item in new_priority_queue]

        return stack_outputs(results)

    def _process_batches(self):
        for batch in self.batch_queue.optimal_batches():
            if not self._running:
                break

            start_time = time.time()

            try:
                batch_inputs = stack_features(
                    [item.content for item in batch],
                    pad_tokens=self.pad_tokens,
                )

                batch_outputs = self.batch_func(batch_inputs)
                unstacked_outputs = unstack_outputs(batch_outputs)

                for item, output in zip(batch, unstacked_outputs):
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
            self._stats.avg_processing_time * (self._stats.total_batches - 1) + processing_time
        ) / self._stats.total_batches

    def stats(self):
        self._stats.queue_size = len(self.batch_queue)
        return self._stats

def _make_processor(func: BatchInfer, **kwargs) -> BatchProcessor:
    batch_size = kwargs.pop('batch_size', 32)
    timeout = kwargs.pop('timeout', 5.0)
    small_batch_threshold = kwargs.pop('small_batch_threshold', 8)
    pad_tokens = kwargs.pop('pad_tokens', None)

    return BatchProcessor(func, batch_size, timeout, small_batch_threshold, pad_tokens)


dynamically = partial(_dynamic_batch, _make_processor)


