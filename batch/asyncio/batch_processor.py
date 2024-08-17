from __future__ import annotations

import asyncio
import logging
import threading
from asyncio import iscoroutinefunction
from collections.abc import Awaitable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Union

from ofen.batch_processor.batch_generator import BatchGenerator, InferenceItem
from ofen.common.tensor_utils import torch_or_np
from ofen.common.utils import first, is_method
from ofen.configs.base.base_config import BaseConfig
from ofen.types import ModelFeatures, ModelOutputs

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger(__name__)


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0


@dataclass
class BatchProcessorConfig(BaseConfig):
    timeout: float = 0.005
    max_batch_size: int = 32
    max_batch_tokens: int = 0
    small_batch_threshold: int = 4
    pad_tokens: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> BaseConfig:
        return cls._from_registry(model_name) or cls(name_or_path=model_name)


InferenceFunction = Union[Callable[[ModelFeatures], ModelOutputs], Callable[[ModelFeatures], Awaitable[ModelOutputs]]]


class BatchProcessor:
    def __init__(
        self,
        inference_func: InferenceFunction,
        timeout: float = 0.005,
        max_batch_size: int = 32,
        max_batch_tokens: int = 0,
        small_batch_threshold: int = 4,
        pad_tokens: dict[str, int] | None = None,
    ):
        if pad_tokens is None:
            pad_tokens = {}
        self._config = BatchProcessorConfig(
            timeout=timeout,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            small_batch_threshold=small_batch_threshold,
            pad_tokens=pad_tokens,
        )
        self._inference_func = self._ensure_async(inference_func)
        self._batch_queue = BatchGenerator(self._config.max_batch_size, timeout=self._config.timeout)
        self._running = False
        self._loop = None
        self._thread = None
        self._start_lock = threading.Lock()
        self._stats = BatchProcessorStats()
        self._start_event = threading.Event()

    @classmethod
    def from_config(cls, config: BatchProcessorConfig, inference_func: InferenceFunction):
        return cls(
            inference_func=inference_func,
            timeout=config.timeout,
            max_batch_size=config.max_batch_size,
            max_batch_tokens=config.max_batch_tokens,
            small_batch_threshold=config.small_batch_threshold,
            pad_tokens=config.pad_tokens,
        )

    @staticmethod
    def _ensure_async(func: Callable):
        return func if iscoroutinefunction(func) else partial(asyncio.to_thread, func)

    def _start_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.create_task(self._process_batches())
        self._start_event.set()
        self._loop.run_forever()

    def start(self):
        with self._start_lock:
            if self._running:
                return
            self._thread = threading.Thread(target=self._start_loop, daemon=True)
            self._thread.start()
            self._start_event.wait()
            self._running = True

    def shutdown(self):
        if self._running:
            self._running = False
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join()

    def __call__(self, **features: ModelFeatures) -> ModelOutputs:
        return self.infer(**features).result()

    def infer(self, **features: ModelFeatures) -> Future[ModelOutputs]:
        if not self._running:
            self.start()
        return asyncio.run_coroutine_threadsafe(self._schedule(features), self._loop)

    async def _schedule(self, features: ModelFeatures) -> ModelOutputs:
        items = ModelInputHelper.unstack(features)

        prioritized = len(items) <= self._config.small_batch_threshold
        new_priority_queue = [
            InferenceItem(
                content=item, future=self._loop.create_future(), size=len(first(item.values())), prioritized=prioritized
            )
            for item in items
        ]
        self._batch_queue.extend(new_priority_queue)
        inferred = await asyncio.gather(*[item.get_result() for item in new_priority_queue])

        return ModelInputHelper.stack_results(inferred)

    async def _process_batches(self):
        async for batch in self._batch_queue.optimal_batches():
            if not self._running:
                break
            raw_batch = ModelInputHelper.pad_and_stack(batch, self._config.pad_tokens)
            start_time = asyncio.get_event_loop().time()
            batch_outputs = await self._inference_func(**raw_batch)
            processing_time = asyncio.get_event_loop().time() - start_time

            self._update_stats(len(batch), processing_time)
            self._distribute_results(batch, batch_outputs)

    def _update_stats(self, batch_size: int, processing_time: float):
        self._stats.total_processed += batch_size
        self._stats.total_batches += 1
        self._stats.avg_batch_size = self._stats.total_processed / self._stats.total_batches
        self._stats.avg_processing_time = (
            self._stats.avg_processing_time * (self._stats.total_batches - 1) + processing_time
        ) / self._stats.total_batches

    def _distribute_results(self, batch: list[InferenceItem], batch_outputs: ModelOutputs):
        keys = batch_outputs.keys() if isinstance(batch_outputs, dict) else None
        for i, item in enumerate(batch):
            try:
                if keys is not None:
                    item.complete({key: batch_outputs[key][i] for key in keys})
                elif isinstance(batch_outputs, list):
                    item.complete([output[i] for output in batch_outputs])
                else:
                    item.complete(batch_outputs[i])
            except Exception as e:
                logger.error(f"Error distributing results: {e}", exc_info=True)
                item.set_exception(e)

    def stats(self):
        self._stats.queue_size = len(self._batch_queue)
        return self._stats


class ModelInputHelper:
    @staticmethod
    def unstack(inputs: ModelFeatures) -> list[ModelFeatures]:
        length = len(first(inputs.values()))
        keys = inputs.keys()
        return [{key: inputs[key][i] for key in keys} for i in range(length)]

    @staticmethod
    def stack_results(results: list[Any]) -> ModelOutputs:
        lib = torch_or_np(results)
        if isinstance(results[0], dict):
            return {key: lib.stack([output[key] for output in results]) for key in results[0]}
        elif isinstance(results[0], list):
            return [lib.stack(outputs) for outputs in zip(*results)]
        return lib.stack(results)

    @staticmethod
    def pad_and_stack(items: list[InferenceItem], pad_tokens: dict[str, int]):
        lib = torch_or_np(items[0].content)
        keys = items[0].content.keys()
        max_length = max(item.content[first(keys)].shape[0] for item in items)
        padded_tensors = {
            key: lib.full((len(items), max_length), pad_tokens.get(key, 0), dtype=lib.int32) for key in keys
        }

        for i, item in enumerate(items):
            for key, tensor in padded_tensors.items():
                tensor_length = item.content[key].shape[0]
                tensor[i, :tensor_length] = item.content[key]

        return padded_tensors


def make_batch(config: BatchProcessorConfig):
    """Decorator to create a batch processor for a given inference function.
    If the inference function is a method, it will create a batch processor for each instance.
    """

    class _WrapperBatchProcessor:
        def __init__(self, forward: InferenceFunction):
            self.forward = forward

        def __get__(self, instance, owner):
            func_name = f"__batched_{self.forward.__name__}"
            if func_name not in instance.__dict__:
                forward = partial(self.forward, instance)
                processor = BatchProcessor.from_config(config, forward)
                instance.__dict__[func_name] = processor

            return instance.__dict__[func_name]

    def _batch_decorator(forward: InferenceFunction):
        if is_method(forward):
            return _WrapperBatchProcessor(forward)
        return BatchProcessor.from_config(config, forward)

    return _batch_decorator
