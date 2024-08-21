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
    from typing import Optional


class BatchProcessor:
    """
    A class for processing batches of model features asynchronously.

    This class manages a queue of model features, processes them in batches,
    and returns model outputs.

    Attributes:
        batch_func (BatchInfer): The function to process batches.
        batch_queue (BatchGenerator): The generator for creating optimal batches.
        small_batch_threshold (int): The threshold for considering a batch as small.
        pad_tokens (dict[str, int]): Dictionary of padding tokens for each feature.
        _running (bool): Flag indicating if the processor is running.
        _thread (threading.Thread): The thread for processing batches.
        _lock (threading.Lock): Lock for thread-safe operations.
        _stats (BatchProcessorStats): Statistics about the batch processing.
    """

    def __init__(
        self,
        _func: BatchInfer,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
        pad_tokens: Optional[dict[str, int]] = None,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            _func (BatchInfer): The function to process batches.
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.
            pad_tokens (dict[str, int] | None): Dictionary of padding tokens for each feature. Defaults to None.
        """
        self.batch_func = _func

        self.batch_queue = BatchGenerator[ModelFeatures, ModelOutputs](batch_size=batch_size, timeout_ms=timeout_ms)
        self.small_batch_threshold = small_batch_threshold
        self.pad_tokens = pad_tokens or {}

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stats = BatchProcessorStats()

    def prioritize(self, items: list[ModelFeatures]) -> list[int]:
        """
        Determine if items should be prioritized based on the batch size.

        Args:
            items (list[ModelFeatures]): The list of items to prioritize.

        Returns:
            list[int]: A list of int values indicating the priority of each item.
        """
        priority = 0 if len(items) <= self.small_batch_threshold else 1
        return [priority] * len(items)

    def start(self):
        """
        Start the batch processing thread if it's not already running.
        """
        with self._lock:
            if self._running:
                return
            self._thread = threading.Thread(target=self._process_batches, daemon=True)
            self._thread.start()
            self._running = True

    def shutdown(self):
        """
        Shutdown the batch processing thread and wait for it to finish.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self.batch_queue.stop()

        if self._thread:
            self._thread.join()

    @overload
    def __call__(self, features: ModelFeatures, /) -> ModelOutputs: ...

    @overload
    def __call__(self, **features: ModelFeatures) -> ModelOutputs: ...

    def __call__(self, *args, **kwargs) -> ModelOutputs:
        """
        Process the input features and return the model outputs.

        This method can be called with either positional or keyword arguments.

        Args:
            *args: Positional arguments (should be a single ModelFeatures object).
            **kwargs: Keyword arguments representing ModelFeatures.

        Returns:
            ModelOutputs: The processed model outputs.
        """
        if not self._running:
            self.start()

        features = args[0] if args else kwargs

        return self._schedule(features)

    def _schedule(self, features: ModelFeatures) -> ModelOutputs:
        """
        Schedule the processing of input features.

        Args:
            features (ModelFeatures): The input features to process.

        Returns:
            ModelOutputs: The processed model outputs.
        """
        items = unstack_features(features)
        prioritized = self.prioritize(items)

        new_priority_queue = [
            Item[ModelFeatures, ModelOutputs](content=item, priority=prio) for item, prio in zip(items, prioritized)
        ]

        self.batch_queue.extend(new_priority_queue)
        results = [item.get_result() for item in new_priority_queue]

        return stack_outputs(results)

    def _process_batches(self):
        """
        Process batches of items from the queue.

        This method runs in a separate thread and continuously processes
        batches until the processor is shut down.
        """
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
        """
        Get the current batch processing statistics.

        Returns:
            BatchProcessorStats: The current statistics.
        """
        return self._stats.clone(queue_size=len(self.batch_queue))


def dynamically(
    func: Optional[BatchInfer] = None,
    /,
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
    pad_tokens: Optional[dict[str, int]] = None,
) -> Callable:
    """
    A decorator to create a BatchProcessor for the given function.

    This decorator can be used with or without arguments.

    Args:
        func (BatchInfer | None): The function to be wrapped. If None, returns a decorator.
        batch_size (int): The maximum size of each batch. Defaults to 32.
        timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
        small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.
        pad_tokens (dict[str, int] | None): Dictionary of padding tokens for each feature. Defaults to None.

    Returns:
        Callable: A decorator that creates a BatchProcessor for the given function.
    """

    def make_processor(_func: BatchInfer) -> BatchProcessor:
        return BatchProcessor(_func, batch_size, timeout_ms, small_batch_threshold, pad_tokens)

    return _dynamic_batch(make_processor, func)
