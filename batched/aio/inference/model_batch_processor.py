from __future__ import annotations

import time
from typing import TYPE_CHECKING, overload

from batched.aio.batch_processor import AsyncBatchProcessor
from batched.decorator import _dynamic_batch
from batched.inference.helper import (
    stack_features,
    stack_outputs,
    unstack_features,
    unstack_outputs,
)
from batched.types import BatchInfer, Feature

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional


class AsyncModelBatchProcessor(AsyncBatchProcessor[dict[str, Feature], Feature]):
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
        super().__init__(
            _func=_func,  # type: ignore[arg-type]
            batch_size=batch_size,
            timeout_ms=timeout_ms,
            small_batch_threshold=small_batch_threshold,
        )

        self.pad_tokens = pad_tokens or {}

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
                )

                batch_outputs = await self.batch_func(batch_inputs)

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
    batch_size: int = 32,
    timeout_ms: float = 5.0,
    small_batch_threshold: int = 8,
    pad_tokens: Optional[dict[str, int]] = None,
) -> Callable:
    """
    Dynamically batch numpy arrays or PyTorch Tensors for inference tasks using asyncio.

    This decorator is designed for inference functions that can process batches of data asynchronously.
    The decorated function should accept a dictionary of input arrays/tensors and return a dictionary
    of output arrays/tensors of the same length. The function should be a coroutine or be convertible to one.

    Args:
        func (BatchInfer | None): The inference function to be wrapped. If None, returns a decorator.
        batch_size (int): The maximum number of samples in each batch. Defaults to 32.
        timeout_ms (float): The maximum wait time in milliseconds for batch formation. Defaults to 5.0.
        small_batch_threshold (int): The threshold to give priority to small batches. Defaults to 8.
        pad_tokens (dict[str, int] | None): Padding token values for each input feature. Defaults to None.

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
        return AsyncModelBatchProcessor(_func, batch_size, timeout_ms, small_batch_threshold, pad_tokens)

    return _dynamic_batch(make_processor, func)
