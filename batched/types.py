from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, Union, runtime_checkable

try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

T = TypeVar("T")
U = TypeVar("U")

BatchFunc = Callable[[list[T]], list[U]]


NDArrayOrTensor = TypeVar("NDArrayOrTensor", "np.ndarray", "torch.Tensor")
Feature = Union[NDArrayOrTensor, list[NDArrayOrTensor]]
BatchInfer = Callable[[dict[str, Feature]], Feature]


def _validate_batch_output(batch_inputs: list[T], batch_outputs: list[U]) -> None:
    """
    Validate that the batch output has the same length as the batch inputs.

    Args:
        batch_inputs (list[T]): The batch inputs.
        batch_outputs (list[U]): The batch outputs.
    """
    assert len(batch_inputs) == len(  # noqa: S101
        batch_outputs
    ), f"Batch output length ({len(batch_outputs)}) does not match batch input length ({len(batch_inputs)})"


@dataclass
class BatchProcessorCacheStats:
    total_gets: int = 0
    total_sets: int = 0
    total_hits: int = 0
    hit_rate: float = 0.0
    utilization_rate: float = 0.0
    total_pops: int = 0.0
    eviction_rate: float = 0.0
    total_get_time: float = 0.0
    total_set_time: float = 0.0

    def update_get(self, hit: T | None, get_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            batch_size (int): The size of the processed batch.
            processing_time (float): The time taken to process the batch in seconds.
        """
        self.total_gets += 1
        self.total_hits += 1 if hit is not None else 0
        self.hit_rate = self.total_hits / self.total_gets
        self.total_get_time += get_time


    def update_set(self, cache_len: float, cache_size: float, item_popped: bool, set_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            batch_size (int): The size of the processed batch.
            processing_time (float): The time taken to process the batch in seconds.
        """
        self.total_sets += 1

        # This is almost always going to be 100% since we don't have time expiry in any of the cache implementations.
        self.utilization_rate = cache_len / cache_size

        self.total_pops += item_popped

        # This also does not make much sense without a time expiry for cache items, since total_pops will always be total_sets-maxsize if total_sets > maxsize
        self.eviction_rate = self.total_pops / self.total_sets
        self.total_set_time += set_time


    def clone(self, *, queue_size: int) -> BatchProcessorStats:
        """
        Create a clone of the current statistics object with an updated queue size.

        Args:
            queue_size (int): The new queue size to set in the cloned object.

        Returns:
            BatchProcessorStats: A new BatchProcessorStats object with updated queue size and copied statistics.
        """
        return BatchProcessorCacheStats(
            total_accessed=self.total_accessed,
            total_hits=self.total_hits,
            total_batches=self.total_batches,
            avg_batch_size=self.avg_batch_size,
            avg_processing_time=self.avg_processing_time,
        )


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0

    def update(self, batch_size: int, processing_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            batch_size (int): The size of the processed batch.
            processing_time (float): The time taken to process the batch in seconds.
        """
        self.total_processed += batch_size
        self.total_batches += 1
        self.avg_batch_size = self.total_processed / self.total_batches
        self.avg_processing_time = (
            self.avg_processing_time * (self.total_batches - 1) + processing_time
        ) / self.total_batches

    def clone(self, *, queue_size: int) -> BatchProcessorStats:
        """
        Create a clone of the current statistics object with an updated queue size.

        Args:
            queue_size (int): The new queue size to set in the cloned object.

        Returns:
            BatchProcessorStats: A new BatchProcessorStats object with updated queue size and copied statistics.
        """
        return BatchProcessorStats(
            queue_size=queue_size,
            total_processed=self.total_processed,
            total_batches=self.total_batches,
            avg_batch_size=self.avg_batch_size,
            avg_processing_time=self.avg_processing_time,
        )


@runtime_checkable
class AsyncCache(Protocol, Generic[T, U]):
    async def get(self, key: T) -> U | None:
        """
        Get a value from the cache.

        Args:
            key (T): The key to retrieve from the cache.

        Returns:
            Optional[U]: The value associated with the key, or None if the key is not found.
        """

    async def set(self, key: T, value: U) -> None:
        """
        Set a value in the cache.

        Args:
            key (T): The key to set in the cache.
            value (U): The value to associate with the key.
        """


@runtime_checkable
class Cache(Protocol, Generic[T, U]):
    def get(self, key: T) -> U | None:
        """
        Get a value from the cache.

        Args:
            key (T): The key to retrieve from the cache.

        Returns:
            Optional[U]: The value associated with the key, or None if the key is not found.
        """
        ...

    def set(self, key: T, value: U) -> None:
        """
        Set a value in the cache.

        Args:
            key (T): The key to set in the cache.
            value (U): The value to associate with the key.
        """
        ...
