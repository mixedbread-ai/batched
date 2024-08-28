import asyncio
import pytest
import numpy as np
from batched.aio.inference.model_batch_processor import AsyncModelBatchProcessor, dynamically
from batched.types import Feature

@pytest.mark.asyncio
async def test_async_model_batch_processor_initialization():
    async def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = AsyncModelBatchProcessor(dummy_batch_func, batch_size=32, timeout_ms=5.0, small_batch_threshold=8)
    
    assert processor.batch_func == dummy_batch_func
    assert processor.batch_queue._batch_size == 32
    assert processor.batch_queue._timeout == 0.005  # 5ms converted to seconds
    assert processor.small_batch_threshold == 8
    assert processor.pad_tokens == {}

@pytest.mark.asyncio
async def test_async_model_batch_processor_call():
    async def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    processor = AsyncModelBatchProcessor(dummy_batch_func)
    result = await processor(input=np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_array_equal(result, np.array([[2, 4, 6], [8, 10, 12]]))

@pytest.mark.asyncio
async def test_async_model_batch_processor_call_multiple():
    async def batch_multiply(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    processor = AsyncModelBatchProcessor(batch_multiply, batch_size=3, timeout_ms=10.0)

    result = await processor(input=np.array([[1], [2], [3], [4], [5]]))
    np.testing.assert_array_equal(result, np.array([[2], [4], [6], [8], [10]]))

@pytest.mark.asyncio
async def test_async_model_batch_processor_exception_handling():
    async def faulty_batch_func(features: dict[str, Feature]) -> Feature:
        raise ValueError("Test error")

    processor = AsyncModelBatchProcessor(faulty_batch_func)

    with pytest.raises(ValueError, match="Test error"):
        await processor(input=np.array([[1]]))

@pytest.mark.asyncio
async def test_async_model_batch_processor_concurrent():
    async def slow_batch_func(features: dict[str, Feature]) -> Feature:
        await asyncio.sleep(0.1)
        return features["input"] * 2

    processor = AsyncModelBatchProcessor(slow_batch_func, batch_size=5, timeout_ms=50.0)

    results = await asyncio.gather(*[processor(input=np.array([[i]])) for i in range(10)])

    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, np.array([[i * 2]]))
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 10

@pytest.mark.asyncio
async def test_async_dynamically_decorator():
    @dynamically(batch_size=3, timeout_ms=10.0)
    async def batch_multiply(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    results = await asyncio.gather(*[batch_multiply(input=np.array([[i]])) for i in range(10)])
    
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, np.array([[i * 2]]))

@pytest.mark.asyncio
async def test_async_model_batch_processor_with_padding():
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = AsyncModelBatchProcessor(batch_func, batch_size=3, pad_tokens={"input": 0})

    result = await processor(input=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    np.testing.assert_array_equal(result, np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 4
