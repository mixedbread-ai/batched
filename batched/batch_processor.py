import threading
import time
from collections.abc import Callable
from typing import Generic, Optional, Union, overload

from batched.batch_generator import BatchGenerator, BatchItem
from batched.decorator import _dynamic_batch
from batched.types import BatchFunc, BatchProcessorStats, T, U, _validate_batch_output


class BatchProcessor(Generic[T, U]):
    """
    A class for processing batches of items asynchronously.

    This class manages a queue of items, processes them in batches, and returns results.

    Attributes:
        batch_func (BatchFunc[T, U]): The function to process batches.
        batch_queue (BatchGenerator): The generator for creating optimal batches.
        small_batch_threshold (int): The threshold for considering a batch as small.
        _running (bool): Flag indicating if the processor is running.
        _thread (threading.Thread): The thread for processing batches.
        _lock (threading.Lock): Lock for thread-safe operations.
        _stats (BatchProcessorStats): Statistics about the batch processing.

    Type Parameters:
        T: The type of input items.
        U: The type of output items.
    """

    def __init__(
        self,
        _func: BatchFunc[T, U],
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            _func (BatchFunc[T, U]): The function to process batches.
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.
        """
        self.batch_func = _func

        self.batch_queue = BatchGenerator(batch_size, timeout_ms)
        self.small_batch_threshold = small_batch_threshold

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._stats = BatchProcessorStats()

    def _determine_priority(self, items: list[T]) -> list[int]:
        """
        Determine if items should be prioritized based on the batch size.

        Args:
            items (list[T]): The list of items to prioritize.

        Returns:
            list[int]: A list of integer values indicating the priority of each item.
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
        Shutdown the batch processing thread, stop the batch queue, and wait for the thread to finish.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            self.batch_queue.stop()

        if self._thread:
            self._thread.join()

    def _schedule(self, items: list[T]) -> list[BatchItem[T, U]]:
        """
        Schedule items for processing and return their results.

        Args:
            items (list[T]): The list of items to schedule.

        Returns:
            list[U]: The list of processed results.
        """
        if not self._running:
            self.start()

        prioritized = self._determine_priority(items)

        new_priority_queue = [
            BatchItem[T, U](
                content=item,
                priority=prio,
            )
            for item, prio in zip(items, prioritized)
        ]

        self.batch_queue.extend(new_priority_queue)
        return new_priority_queue

    def _process_batches(self) -> None:
        """
        Continuously process batches of items from the queue.
        """
        for batch in self.batch_queue.optimal_batches():
            if not self._running:
                break

            start_time = time.time()
            try:
                batch_inputs = [item.content for item in batch]
                batch_outputs = self.batch_func(batch_inputs)

                _validate_batch_output(batch_inputs, batch_outputs)

                for item, output in zip(batch, batch_outputs):
                    item.set_result(output)

            except Exception as e:  # noqa: BLE001
                for item in batch:
                    item.set_exception(e)

            finally:
                processing_time = time.time() - start_time
                self._stats.update(len(batch), processing_time)

    @property
    def stats(self):
        """
        Get the current statistics of the batch processor.

        Returns:
            BatchProcessorStats: The current statistics.
        """
        return self._stats.clone(queue_size=len(self.batch_queue))

    @overload
    def __call__(self, item: T) -> U: ...

    @overload
    def __call__(self, items: list[T]) -> list[U]: ...

    def __call__(self, items: Union[T, list[T]]) -> Union[U, list[U]]:
        """
        Process a single item or a list of items.

        This method starts the batch processor if it's not already running,
        then schedules the item(s) for processing.

        Args:
            items (T | list[T]): A single item or a list of items to process.

        Returns:
            U | list[U]: The processed result for a single item, or a list of results for multiple items.
        """
        if isinstance(items, list):
            items = self._schedule(items)
            return [item.result() for item in items]

        return self._schedule([items])[0].result()

    @overload
    async def acall(self, item: T) -> U: ...

    @overload
    async def acall(self, items: list[T]) -> list[U]: ...

    async def acall(self, item: Union[T, list[T]]) -> Union[U, list[U]]:
        """
        Infer the result of a single item asynchronously.

        Args:
            item (T): The item to infer.

        Returns:
            U: The inferred result.
        """
        import asyncio  # noqa: PLC0415

        if isinstance(item, list):
            futures = self._schedule(item)
            return list(await asyncio.gather(*[f.to_awaitable() for f in futures]))
        return await self._schedule([item])[0].to_awaitable()

    def __del__(self):
        self.shutdown()


def dynamically(
    func: Optional[BatchFunc] = None,
    /,
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
) -> Callable:
    """
    Dynamically batch inputs for processing in a thread.

    This decorator is designed for functions that can process batches of data without using asyncio.
    The decorated function should accept a list of input items and return a list of output items
    of the same length. The function should not be awaitable.

    Args:
        func (BatchFunc | None): The function to be wrapped. If None, returns a partial function.
        batch_size (int): The maximum size of each batch. Defaults to 32.
        timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
        small_batch_threshold (int): The threshold to give priority to small batches. Defaults to 8.

    Returns:
        Callable: A decorator that creates a BatchProcessor for the given function.

    Example:
        @dynamically(batch_size=64, timeout_ms=10.0)
        def process_items(items: list[str]) -> list[int]:
            return [len(item) for item in items]

        # Single item processing
        single_result = process_items("hello")
        # Returns: 5

        # Batch processing
        batch_result = process_items(["hello", "world", "python"])
        # Returns: [5, 5, 6]

        # Asynchronous processing
        import asyncio

        async def main():
            single_result_async = await process_items.acall("hello")
            batch_result_async = await process_items.acall(["hello", "world", "python"])

        asyncio.run(main())
    """

    def make_processor(_func: BatchFunc) -> BatchProcessor:
        return BatchProcessor(_func, batch_size, timeout_ms, small_batch_threshold)

    return _dynamic_batch(make_processor, func)
