from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, overload

from batch.batch_generator import BatchGenerator, Item
from batch.decorator import _dynamic_batch
from batch.inference.types import (
    BatchInfer,
    ModelFeatures,
    ModelOutputs,
    stack_features,
    stack_outputs,
    unstack_features,
    unstack_outputs,
)
from batch.types import BatchProcessorStats

if TYPE_CHECKING:
    from collections.abc import Callable


class BatchProcessor:
    def __init__(
        self,
        _func: BatchInfer,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
        pad_tokens: dict[str, int] | None = None,
    ):
        self.batch_func = _func

        self.batch_queue = BatchGenerator[ModelFeatures, ModelOutputs](
            batch_size=batch_size, timeout_ms=timeout_ms
        )
        self.small_batch_threshold = small_batch_threshold
        self.pad_tokens = pad_tokens or {}

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[ModelFeatures]) -> list[bool]:
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

    @overload
    def __call__(self, features: ModelFeatures, /) -> ModelOutputs: ...

    @overload
    def __call__(self, **features: ModelFeatures) -> ModelOutputs: ...

    def __call__(self, *args, **kwargs) -> ModelOutputs:
        if not self._running:
            self.start()

        features = args[0] if args else kwargs

        return self._schedule(features)

    def _schedule(self, features: ModelFeatures) -> ModelOutputs:
        items = unstack_features(features)
        prioritized = self.prioritize(items)

        new_priority_queue = [
            Item[ModelFeatures, ModelOutputs](
                content=item,
                prioritized=prio
            ) for item, prio in zip(items, prioritized)]

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
                self._stats.update(len(batch), processing_time)

    @property
    def stats(self):
        return self._stats.clone(queue_size=len(self.batch_queue))


def dynamically(
        func: BatchInfer | None = None,
        /,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
        pad_tokens: dict[str, int] | None = None,
) -> Callable:
    def make_processor(_func: BatchInfer) -> BatchProcessor:
        return BatchProcessor(
            _func,
            batch_size,
            timeout_ms,
            small_batch_threshold,
            pad_tokens
        )

    return _dynamic_batch(make_processor, func)
