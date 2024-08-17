from __future__ import annotations
import inspect

import threading
import time
from functools import partial
from typing import Any, Optional, Union
from batch.batch_config import BatchProcessorConfig, DEFAULT_CONFIG, BatchProcessorStats

from batch.batch_generator import BatchGenerator, InferenceItem
from batch.logger import LOGGER
from batch.types import InferenceFunction, ModelFeatures, ModelOutputs
from batch.utils import first, is_method, torch_or_np


class BatchProcessor:
    def __init__(
        self,
        inference_func: InferenceFunction,
        timeout: float = 0.005,
        max_batch_size: int = 32,
        max_batch_tokens: int = 0,
        small_batch_threshold: int = 4,
        pad_tokens: dict[str, int] | None = None,
        device: Optional[str] = None,
    ):
        if pad_tokens is None:
            pad_tokens = {}
        self.config = BatchProcessorConfig(
            timeout=timeout,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            small_batch_threshold=small_batch_threshold,
            pad_tokens=pad_tokens,
        )
        self._device = device
        self._inference_func = inference_func
        self._inference_func_expects_dict = len(inspect.signature(inference_func).parameters) == 1

        self._batch_queue = BatchGenerator(self.config.max_batch_size, timeout=self.config.timeout)
        self._stats = BatchProcessorStats()

        self._running = False
        self._loop = None
        self._thread = None
        self._start_lock = threading.Lock()
        self._start_event = threading.Event()

    @classmethod
    def from_config(cls, config: BatchProcessorConfig, inference_func: InferenceFunction, **kwargs):
        return cls(
            inference_func=inference_func,
            timeout=config.timeout,
            max_batch_size=config.max_batch_size,
            max_batch_tokens=config.max_batch_tokens,
            small_batch_threshold=config.small_batch_threshold,
            pad_tokens=config.pad_tokens,
            **kwargs,
        )

    def start(self):
        with self._start_lock:
            if self._running:
                return
            self._thread = threading.Thread(target=self._process_batches, daemon=True)
            self._thread.start()
            self._running = True

    def shutdown(self):
        if self._running:
            self._running = False
            if self._thread:
                self._thread.join()

    def __call__(self, features: Optional[ModelFeatures] = None, **kwargs: ModelFeatures) -> ModelOutputs:
        if not self._running:
            self.start()

        if features is None:
            features = kwargs
        return self._schedule(features)

    def _schedule(self, features: ModelFeatures) -> ModelOutputs:
        items = ModelInputHelper.unstack(features)

        prioritized = len(items) <= self.config.small_batch_threshold
        new_priority_queue = [
            InferenceItem(content=item, size=len(first(item.values())), prioritized=prioritized) for item in items
        ]
        self._batch_queue.extend(new_priority_queue)
        inferred = [item.get_result() for item in new_priority_queue]

        return ModelInputHelper.stack_results(inferred)

    def _process_batches(self):
        for batch in self._batch_queue.optimal_batches():
            raw_batch = ModelInputHelper.pad_and_stack(batch, self.config.pad_tokens)

            start_time = time.time()
            batch_outputs = (
                self._inference_func(**raw_batch)
                if not self._inference_func_expects_dict
                else self._inference_func(raw_batch)
            )
            processing_time = time.time() - start_time

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
        try:
            keys = batch_outputs.keys() if isinstance(batch_outputs, dict) else None
            for i, item in enumerate(batch):
                if keys is not None:
                    item.complete({key: batch_outputs[key][i] for key in keys})
                elif isinstance(batch_outputs, list):
                    item.complete([output[i] for output in batch_outputs])
                else:
                    item.complete(batch_outputs[i])
        except Exception as e:  # noqa: BLE001
            LOGGER.error(f"Error distributing results: {e}", exc_info=True)
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
        if isinstance(results[0], list):
            return [lib.stack(outputs) for outputs in zip(*results)]
        return lib.stack(results)

    @staticmethod
    def pad_and_stack(items: list[InferenceItem], pad_tokens: dict[str, int]):
        lib = torch_or_np(items[0].content)
        keys = items[0].content.keys()
        max_length = max(item.content[first(keys)].shape[0] for item in items)

        padded_tensors = {
            key: lib.full((len(items), max_length), pad_tokens.get(key, 0), dtype=lib.int64) for key in keys
        }

        for i, item in enumerate(items):
            for key, tensor in padded_tensors.items():
                tensor_length = item.content[key].shape[0]
                tensor[i, :tensor_length] = item.content[key]

        return padded_tensors


def dynamic_batching(fn_or_config: Union[BatchProcessorConfig, InferenceFunction] = DEFAULT_CONFIG, **kwargs):
    """Decorator to create a batch processor for a given inference function.
    If the inference function is a method, it will create a batch processor for each instance.

    Args:
        fn_or_config (Union[BatchProcessorConfig, InferenceFunction], optional):
            Either a BatchProcessorConfig object to configure the batch processor,
            or the inference function to be decorated. Defaults to DEFAULT_CONFIG.
        **kwargs:
            Additional keyword arguments to pass to the BatchProcessor constructor.

    Returns:
        Union[BatchProcessor, Callable[[InferenceFunction], Union[BatchProcessor, _WrapperBatchProcessor]]]:
            If fn_or_config is an InferenceFunction, returns a BatchProcessor.
            If fn_or_config is a BatchProcessorConfig, returns a decorator that will create
            either a BatchProcessor or a _WrapperBatchProcessor depending on whether
            the decorated function is a method or not.
    """

    class _WrapperBatchProcessor:
        def __init__(self, config: BatchProcessorConfig, forward: InferenceFunction, **kwargs):
            self.forward = forward
            self.config = config
            self.kwargs = kwargs

        def __get__(self, instance, owner):
            func_name = f"__batched_{self.forward.__name__}"
            if func_name not in instance.__dict__:
                forward = partial(self.forward, instance)
                processor = BatchProcessor.from_config(self.config, forward, **kwargs)
                instance.__dict__[func_name] = processor

            return instance.__dict__[func_name]

    def _create_batch_processor(config: BatchProcessorConfig):
        def _batch_decorator(forward: InferenceFunction):
            if is_method(forward):
                return _WrapperBatchProcessor(config, forward, **kwargs)
            return BatchProcessor.from_config(config, forward, **kwargs)

        return _batch_decorator

    if callable(fn_or_config):
        return _create_batch_processor(DEFAULT_CONFIG)(fn_or_config)

    return _create_batch_processor(fn_or_config)
