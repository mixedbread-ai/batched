from __future__ import annotations

from collections.abc import Callable, Sized
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
    total_pops: int = 0
    eviction_rate: float = 0.0
    total_get_hit_time: float = 0.0
    total_get_miss_time: float = 0.0
    total_set_time: float = 0.0
    latency_reduction: float = 0.0

    def update_get(self, is_hit: bool, get_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            hit (bool): The hit element, None if there is no hit.
            get_time (float): The time taken to process the get function.
        """
        self.total_gets += 1
        self.total_hits += is_hit

        if is_hit:
            self.total_get_hit_time += get_time
        else:
            self.total_get_miss_time += get_time


    def update_batch_get(self, n_gets: int, n_hits: int, get_time: float) -> None:
        """
        Update the statistics in batch, given a set of parameters.

        Args:
            n_gets (int): How many get operations were performed.
            n_hits (int): How many hits were achieved.
            get_time (float): The total time spent in the operations.
        """
        self.total_gets += n_gets
        self.total_hits += n_hits
        self.total_get_hit_time += (get_time / n_gets) * n_hits
        self.total_get_miss_time += (get_time / n_gets) * (n_gets - n_hits)


    def update_set(self, item_popped: bool, set_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            item_popped (bool): True if an item was popped from the cache in this set instance. False otherwise.
            set_time (float): The time taken to process the set function.
        """
        self.total_sets += 1

        self.total_pops += item_popped

        self.total_set_time += set_time
        

    def get_stats(self, cache: object | None = None, maxsize: int | None = None, batch_processor_stats: BatchProcessorStats | None = None) -> BatchProcessorCacheStats:
        """
        Gets the cache stats.

        Args:
            cache (object | None): The cache object.
            maxsize (int | None): The max size of the cache.
            batch_processor_stats (BatchProcessorStats | None): The stats of the batch processor. Used for the latency reduction calculation.

        Returns:
            BatchProcessorCacheStats:
                total_gets (int): How many get operations were performed.
                total_sets (int): How many set operations were performed.
                total_hits (int): How many of the get operations were a hit.
                hit_rate (float): The proportion of hits wrt the total_gets.
                utilization_rate (float): How much of the cache size are we using.
                total_pops (int): How many times did we pop an item.
                eviction_rate (float): How many times did we pop wrt the total_sets.
                total_get_hit_time (float): The total time spent in get operations in case of hit.
                total_get_nonhit_time (float): The total time spent in get operations in case of a miss.
                total_set_time (float): The total time spent in set operations.
                latency_reduction (float): The total time we saved using the cache implementation.
        """

        if maxsize is not None and cache is not None:
            if isinstance(cache, Sized):
                # Mostly irrelevant without a timeout for cache items. Can still be useful if cache_size is too big.
                self.utilization_rate = len(cache) / maxsize

        # This does not make much sense without a time expiry for cache items, since total_pops will always be total_sets-maxsize if total_sets > maxsize
        self.eviction_rate = (self.total_pops / self.total_sets) if self.total_sets > 0 else 0

        self.hit_rate = (self.total_hits / self.total_gets) if self.total_gets > 0 else 0

        if batch_processor_stats is not None and batch_processor_stats.avg_batch_size > 0:
            # avg non hit processing time per element
            avg_processing_time_per_element = batch_processor_stats.avg_processing_time / batch_processor_stats.avg_batch_size

            # how much time we would have spent calculating if we did not have a cache
            total_processing_time_per_element = avg_processing_time_per_element * self.total_hits 
            
        else:
            total_processing_time_per_element = 0

        # the latency reduction is the time we would have spent if we had no cache minus the total cache overhead
        latency_reduction = total_processing_time_per_element - (self.total_get_hit_time + self.total_get_miss_time + self.total_set_time)

        return BatchProcessorCacheStats(
            total_gets=self.total_gets,
            total_sets=self.total_sets,
            total_hits=self.total_hits,
            hit_rate=self.hit_rate,
            utilization_rate=self.utilization_rate,
            total_pops=self.total_pops,
            eviction_rate=self.eviction_rate,
            total_get_hit_time=self.total_get_hit_time,
            total_get_miss_time=self.total_get_miss_time,
            total_set_time=self.total_set_time,
            latency_reduction=latency_reduction
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
