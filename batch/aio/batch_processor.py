import asyncio
from collections.abc import Callable
from typing import Generic, Optional, Union, overload

import batch.utils as utils
from batch.aio.batch_generator import AsyncBatchGenerator, AsyncBatchItem
from batch.decorator import _dynamic_batch
from batch.types import BatchFunc, BatchProcessorStats, T, U, _validate_batch_output


class AsyncBatchProcessor(Generic[T, U]):
    """
    A class for processing batches of items asynchronously.

    This class manages a queue of items, processes them in batches, and returns results.

    Attributes:
        batch_func (Callable[[list[T]], Awaitable[list[U]]]): The function to process batches.
        batch_queue (BatchGenerator[T, U]): The generator for creating optimal batches.
        small_batch_threshold (int): The threshold for considering a batch as small.
        _loop (asyncio.AbstractEventLoop): The event loop for asynchronous operations.
        _task (asyncio.Task): The task for processing batches.
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
        self.small_batch_threshold = small_batch_threshold
        self.batch_func = utils.ensure_async(_func)
        self.batch_queue = AsyncBatchGenerator[T, U](batch_size, timeout_ms)

        self._loop = utils.get_or_create_event_loop()
        self._task = self._loop.create_task(self._process_batches())
        self._stats = BatchProcessorStats()

    def _determine_priority(self, items: list[T]) -> list[int]:
        """
        Determine if items should be prioritized based on the batch size.

        Args:
            items (list[T]): The list of items to prioritize.

        Returns:
            list[bool]: A list of boolean values indicating the priority of each item.
        """
        priority = 0 if len(items) <= self.small_batch_threshold else 1
        return [priority] * len(items)

    async def _schedule(self, items: list[T]) -> list[U]:
        """
        Schedule items for processing and return their results.

        Args:
            items (list[T]): The list of items to schedule.

        Returns:
            list[U]: The list of processed results.
        """
        prioritized = self._determine_priority(items)

        batch_items = [
            AsyncBatchItem[T, U](
                content=item,
                priority=prio,
                future=self._loop.create_future(),
            )
            for item, prio in zip(items, prioritized)
        ]
        await self.batch_queue.extend(batch_items)

        futures = [item.future for item in batch_items]
        return list(await asyncio.gather(*futures))

    async def _process_batches(self) -> None:
        """
        Continuously process batches of items from the queue.
        """
        async for batch in self.batch_queue.optimal_batches():
            start_time = self._loop.time()

            try:
                batch_inputs = [item.content for item in batch if not item.done()]
                batch_outputs = await self.batch_func(batch_inputs)
                _validate_batch_output(batch_inputs, batch_outputs)

                for item, output in zip(batch, batch_outputs):
                    item.set_result(output)

            except Exception as e:  # noqa: BLE001
                for item in batch:
                    item.set_exception(e)
            finally:
                processing_time = self._loop.time() - start_time
                self._stats.update(len(batch), processing_time)

    @property
    def stats(self) -> BatchProcessorStats:
        """
        Get the current statistics of the batch processor.

        Returns:
            BatchProcessorStats: The current statistics.
        """
        return self._stats.clone(queue_size=len(self.batch_queue))

    @overload
    async def __call__(self, item: T) -> U: ...

    @overload
    async def __call__(self, items: list[T]) -> list[U]: ...

    async def __call__(self, items: Union[T, list[T]]) -> Union[U, list[U]]:
        """
        Process a single item or a list of items.

        Args:
            items (T | list[T]): A single item or a list of items to process.

        Returns:
            U | list[U]: The processed result for a single item, or a list of results for multiple items.
        """
        if isinstance(items, list):
            return await self._schedule(items)
        return (await self._schedule([items]))[0]

    def __del__(self):
        if self._task is None or not self._loop.is_running():
            return

        self._task.cancel()


def dynamically(
    func: Optional[BatchFunc[T, U]] = None,
    /,
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
) -> Callable:
    """
    A decorator for creating a BatchProcessor with dynamic batching.

    Args:
        func (BatchFunc[T, U] | None): The function to be wrapped. If None, returns a partial function.
        batch_size (int): The maximum size of each batch. Defaults to 32.
        timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
        small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.

    Returns:
        Callable: A decorator that creates a BatchProcessor for the given function.
    """

    def make_processor(_func: BatchFunc[T, U]) -> AsyncBatchProcessor[T, U]:
        return AsyncBatchProcessor(_func, batch_size, timeout_ms, small_batch_threshold)

    return _dynamic_batch(make_processor, func)
