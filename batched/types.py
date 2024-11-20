from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
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
BatchInfer = Union[Callable[[dict[str, Feature]], Feature], Callable[..., Feature]]


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


class PriorityStrategy(int, Enum):
    """
    An enumeration of strategies for prioritizing items in a batch queue.

    Attributes:
        NONE: No prioritization - items are processed in FIFO order.
        LENGTH: Prioritize based on item length - longer items get higher priority.
        PRIORITY: Prioritize based on explicit priority values assigned to items.
    """

    NONE = auto()
    LENGTH = auto()
    PRIORITY = auto()


@dataclass
class CacheStats:
    gets: int = 0
    hits: int = 0
    sets: int = 0

    # Cumulative timing metrics
    total_get_time: float = 0.0
    total_hit_time: float = 0.0
    total_set_time: float = 0.0

    @property
    def avg_get_time(self) -> float:
        return self.total_get_time / self.gets if self.gets > 0 else 0.0

    @property
    def avg_hit_time(self) -> float:
        return self.total_hit_time / self.hits if self.hits > 0 else 0.0

    @property
    def avg_set_time(self) -> float:
        return self.total_set_time / self.sets if self.sets > 0 else 0.0

    def update_get(self, *, hit: bool, time_taken: float) -> None:
        self.gets += 1
        self.total_get_time += time_taken

        if hit:
            self.hits += 1
            self.total_hit_time += time_taken

    def update_set(self, *, time_taken: float) -> None:
        self.sets += 1
        self.total_set_time += time_taken


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0
    cache_stats: CacheStats | None = None

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

    def clone(self, *, queue_size: int, cache_stats: CacheStats | None = None) -> BatchProcessorStats:
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
            cache_stats=cache_stats,
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

    async def clear(self) -> None:
        """Clear the cache"""

    def stats(self) -> CacheStats:
        """Return the cache statistics"""
