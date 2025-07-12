from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

from batched.aio.batch_generator import AsyncBatchItem
from batched.aio.batch_processor import AsyncBatchProcessor
from batched.decorator import _dynamic_batch
from batched.inference.helper import (
    stack_features,
    stack_outputs,
    unstack_features,
    unstack_outputs,
)
from batched.types import AsyncCache, BatchInfer, Feature, PriorityStrategy

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional


class AsyncModelBatchProcessor(AsyncBatchProcessor[dict[str, Feature], Feature]):
    def __init__(
        self,
        _func: BatchInfer,
        *,
        batch_size: int = 32,
        timeout_ms: float = 5.0,
        small_batch_threshold: int = 8,
        cache: AsyncCache[dict[str, Feature], Feature] | None = None,
        max_batch_length: int | None = None,
        pad_tokens: Optional[dict[str, int]] = None,
        padding_side: str = "right",
        priority_strategy: PriorityStrategy = PriorityStrategy.NONE,
        batch_item_cls: type[AsyncBatchItem[dict[str, Feature], Feature]] = AsyncBatchItem[dict[str, Feature], Feature],
        spread_kwargs: bool = False,
    ):
        """
        Initialize the BatchProcessor.

        Args:
            _func (BatchInfer): The function to process batches.
            batch_size (int): The maximum size of each batch. Defaults to 32.
            timeout_ms (float): The timeout in milliseconds between batch generation attempts. Defaults to 5.0.
            small_batch_threshold (int): The threshold for considering a batch as small. Defaults to 8.
            cache (AsyncCache | None): An optional cache for storing results. Defaults to None.
            max_batch_length (int | None): The maximum length of a batch. Defaults to None.
            pad_tokens (dict[str, int] | None): Dictionary of padding tokens for each feature. Defaults to None.
            padding_side (str): Side to add padding tokens. Either "left" or "right". Defaults to "right".
            priority_strategy (PriorityStrategy): The strategy to use for prioritizing items.
            batch_item_cls (type[AsyncBatchItem]): The class to use for batch items. Defaults to AsyncBatchItem.
            spread_kwargs (bool): Whether to spread the kwargs over passing dict as args. Defaults to False.
        """
        super().__init__(
            func=_func,  # type: ignore[arg-type]
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            small_batch_threshold=small_batch_threshold,
            cache=cache,
            max_batch_length=max_batch_length,
            priority_strategy=priority_strategy,
            batch_item_cls=batch_item_cls,
        )

        self.pad_tokens = pad_tokens or {}
        self.padding_side = padding_side
        self.spread_kwargs = spread_kwargs

    async def _process_batches(self):
        """
        Process batches of items from the queue.

        This method runs in a separate thread and continuously processes
        batches until the processor is shut down.
        """
        async for batch in self.batch_queue.optimal_batches():
            start_time = time.time()

            try:
                batch_inputs = stack_features(
                    [item.content for item in batch],
                    pad_tokens=self.pad_tokens,
                    padding_side=self.padding_side,
                )

                batch_outputs = (
                    await self.batch_func(**batch_inputs) if self.spread_kwargs else await self.batch_func(batch_inputs)
                )

                unstacked_outputs = unstack_outputs(batch_outputs)
                for item, output in zip(batch, unstacked_outputs):
                    item.set_result(output)

            except Exception as e:  # noqa: BLE001
                for item in batch:
                    item.set_exception(e)

            finally:
                processing_time = time.time() - start_time
                self._stats.update(len(batch), processing_time)

    @overload
    async def __call__(self, features: dict[str, Feature], /) -> Feature: ...

    @overload
    async def __call__(self, **features: Feature) -> Feature: ...

    async def __call__(self, *args, **kwargs) -> Feature:
        """
        Process the input features and return the model outputs.

        This method can be called with either positional or keyword arguments.

        Args:
            *args: Positional arguments (should be a single ModelFeatures object).
            **kwargs: Keyword arguments representing ModelFeatures.

        Returns:
            ModelOutputs: The processed model outputs.
        """
        features = args[0] if args else kwargs
        unstacked_features = unstack_features(features)
        outputs = await self._schedule(unstacked_features)
        return stack_outputs(outputs)


def dynamically(
    func: Optional[BatchInfer] = None,
    /,
    *,
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
    max_batch_length: int | None = None,
    pad_tokens: Optional[dict[str, int]] = None,
    padding_side: str = "right",
    priority_strategy: PriorityStrategy = PriorityStrategy.NONE,
    cache: AsyncCache[dict[str, Feature], Feature] | None = None,
    batch_item_cls: type[AsyncBatchItem[dict[str, Feature], Feature]] = AsyncBatchItem[dict[str, Feature], Feature],
    spread_kwargs: bool = False,
) -> Callable:
    """
    Dynamically batch numpy arrays or PyTorch Tensors for inference tasks using asyncio.

    Args:
        func (BatchInfer | None): The inference function to be wrapped. If None, returns a decorator.
        batch_size (int): The maximum number of samples in each batch. Defaults to 32.
        timeout_ms (float): The maximum wait time in milliseconds for batch formation. Defaults to 5.0.
        small_batch_threshold (int): The threshold to give priority to small batches. Defaults to 8.
        max_batch_length (int | None): The maximum length of a batch. Defaults to None.
        pad_tokens (dict[str, int] | None): Padding token values for each input feature. Defaults to None.
        padding_side (str): Side to add padding tokens. Either "left" or "right". Defaults to "right".
        priority_strategy (PriorityStrategy): The strategy to use for prioritizing items.
        cache (AsyncCache | None): An optional cache for storing results.
        batch_item_cls (type[AsyncBatchItem]): The class to use for batch items. Defaults to AsyncBatchItem.
        spread_kwargs (bool): Whether to spread the kwargs over passing dict as args. Defaults to False.

    Returns:
        Callable: A decorator that creates an AsyncModelBatchProcessor for efficient batched inference.

    Example:
        @aio.inference.dynamically(pad_tokens={'input_ids': 0})
        async def infer_batch(inputs: dict[str, np.ndarray]) -> np.ndarray:
            # Perform inference on the batch
            return model(inputs['input'])

        results = await infer_batch(multiple_samples)

    """

    def make_processor(_func: BatchInfer) -> AsyncModelBatchProcessor:
        return AsyncModelBatchProcessor(
            _func,
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            small_batch_threshold=small_batch_threshold,
            max_batch_length=max_batch_length,
            pad_tokens=pad_tokens,
            padding_side=padding_side,
            priority_strategy=priority_strategy,
            cache=cache,
            batch_item_cls=batch_item_cls,
            spread_kwargs=spread_kwargs,
        )

    return _dynamic_batch(make_processor, func)
