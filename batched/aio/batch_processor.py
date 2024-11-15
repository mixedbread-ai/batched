import asyncio
from collections.abc import Callable
from typing import Generic, Optional, Union, overload

import batched.utils as utils
from batched.aio.batch_generator import AsyncBatchGenerator, AsyncBatchItem
from batched.decorator import _dynamic_batch
from batched.types import AsyncCache, BatchFunc, BatchProcessorStats, T, U, _validate_batch_output


class AsyncBatchProcessor(Generic[T, U]):
    """
    A class for processing batches of items asynchronously.

    This class manages a queue of items, processes them in batches, and returns results.

    Attributes:
        batch_func (Callable[[list[T]], Awaitable[list[U]]]): The function to process batches.
        batch_queue (AsyncBatchGenerator[T, U]): The generator for creating optimal batches.
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
        *,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
        cache: AsyncCache[T, U] | None = None,
        prioritize_by_length: bool = False,
        max_batch_length: int | None = None,
        item_len_fn: Callable[[T], int] | None = None,
        use_batch_cache: bool = False
    ):
        """
        Initialize the AsyncBatchProcessor.

        Args:
            _func (BatchFunc[T, U]): The function to process batches.
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            small_batch_threshold (int): The threshold to give priority to small batches. Defaults to 8.
            cache (AsyncCache[T, U] | None): An optional cache for storing results. Defaults to None.
            prioritize_by_length (bool): Whether to prioritize items by length. Defaults to False.
            max_batch_length (int | None): The maximum length of a batch. Defaults to None.
            item_len_fn (Callable[[T], int] | None): A function to get the length of an item. Defaults to None.
            use_batch_cache (bool): If set to True the batch generator will construct a cache for each batch and then read on that instead of doing it on the main cache.
                This is useful if the main cache is stored on disk, since then the get/set is batched instead of doing it 1 by 1, 
                which can speed up results by 2X, otherwise the overhead is not worth it.
                If set to True the cache has to implement a get_all() method. Defaults to False.
        """
        self.batch_func = utils.ensure_async(_func)
        self.batch_queue = AsyncBatchGenerator[T, U](
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            cache=cache,
            max_batch_length=max_batch_length,
            use_batch_cache=use_batch_cache
        )
        self._stats = BatchProcessorStats()

        self._loop = None
        self._task = None
        self._len_fn = item_len_fn
        self._prioritize_by_length = prioritize_by_length
        self.small_batch_threshold = small_batch_threshold

    def _start(self) -> None:
        self._loop = utils.get_or_create_event_loop()
        self._task = self._loop.create_task(self._process_batches())

    def _determine_priority(self, items: list[T]) -> list[int]:
        """
        Determine the priority of items based on batch size and content length.

        Args:
            items (list[T]): The list of items to prioritize.

        Returns:
            list[int]: A list of integer values indicating the priority of each item.
        """
        if len(items) <= self.small_batch_threshold:
            return [0] * len(items)

        if not self._prioritize_by_length:
            return [1] * len(items)

        return [len(item) for item in items]

    async def _schedule(self, items: list[T]) -> list[U]:
        """
        Schedule items for processing and return their results.

        Args:
            items (list[T]): The list of items to schedule.

        Returns:
            list[U]: The list of processed results.
        """
        if self._loop is None:
            self._start()

        prioritized = self._determine_priority(items)

        batch_items = [
            AsyncBatchItem[T, U](
                content=item,
                priority=prio,
                future=self._loop.create_future(),
                _len_fn=self._len_fn,
            )
            for item, prio in zip(items, prioritized)
        ]
        await self.batch_queue.set_batch_cache(batch_items)
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
    async def __call__(self, item: T) -> U:
        ...

    @overload
    async def __call__(self, items: list[T]) -> list[U]:
        ...

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
        if self._loop is None or self._loop.is_closed():
            return
        if self._task is None:
            return
        self._task.cancel()


def dynamically(
    func: Optional[BatchFunc[T, U]] = None,
    /,
    *,
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
    max_batch_length: int | None = None,
    prioritize_by_length: bool = False,
    item_len_fn: Callable[[T], int] | None = None,
    cache: AsyncCache[T, U] | None = None,
    use_batch_cache: bool = False
) -> Callable:
    """
    Dynamically batch inputs for processing using asyncio.

    This decorator is designed for functions that can process batches of data in an asyncio-based environment.
    It is suitable for cases such as FastAPI handlers.

    The decorated function can be co-routines or not. It should accept a list of input items and return
    a list of output items of the same length. The returned function is a coroutine.

    Args:
        func (BatchFunc[T, U] | None): The function to be wrapped. If None, returns a partial function.
        batch_size (int): The maximum size of each batch. Defaults to 32.
        timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
        small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.
        max_batch_length (int | None): The maximum length of a batch. Defaults to None.
        prioritize_by_length (bool): Whether to prioritize items by length. Defaults to False.
        item_len_fn (Callable[[T], int] | None): A function to get the length of an item. Defaults to None.
        cache (AsyncCache[T, U] | None): An optional cache for storing results. Defaults to None.
        use_batch_cache (bool): If set to True the batch generator will construct a cache for each batch and then read on that instead of doing it on the main cache.
                This is useful if the main cache is stored on disk, since then the get/set is batched instead of doing it 1 by 1, 
                which can speed up results by 2X, otherwise the overhead is not worth it.
                If set to True the cache has to implement a get_all() method. Defaults to False.

    Returns:
        Callable: A decorator that creates an AsyncBatchProcessor for the given function.

    Example:
        @aio.dynamically(batch_size=64, timeout_ms=10.0)
        async def process_items(items: list[str]) -> list[int]:
            return [len(item) for item in items]

        # Single item processing
        result = await process_items("hello")
        # Returns: 5

        # Batch processing
        batch_result = await process_items(["hello", "world", "python"])
        # Returns: [5, 5, 6]

        # The decorator also works with non-awaitable functions
        @aio.dynamically(batch_size=32)
        def sync_process(items: list[str]) -> list[int]:
            return [len(item) * 2 for item in items]

        # The decorated function is still awaitable
        sync_result = await sync_process(["a", "bc", "def"])
        # Returns: [2, 4, 6]
    """

    def make_processor(_func: BatchFunc[T, U]) -> AsyncBatchProcessor[T, U]:
        return AsyncBatchProcessor(
            _func,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            small_batch_threshold=small_batch_threshold,
            max_batch_length=max_batch_length,
            cache=cache,
            prioritize_by_length=prioritize_by_length,
            item_len_fn=item_len_fn,
            use_batch_cache=use_batch_cache
        )

    return _dynamic_batch(make_processor, func)
