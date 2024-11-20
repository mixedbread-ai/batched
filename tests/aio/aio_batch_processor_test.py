import asyncio

import pytest

from batched.aio.batch_processor import AsyncBatchProcessor, dynamically
from batched.types import PriorityStrategy


@pytest.mark.asyncio
async def test_async_batch_processor_initialization():
    async def dummy_batch_func(items):
        return items

    processor = AsyncBatchProcessor(dummy_batch_func, batch_size=32, timeout_ms=5.0, small_batch_threshold=8)
    
    assert processor.batch_func == dummy_batch_func
    assert processor.batch_queue._batch_size == 32
    assert processor.batch_queue._timeout == 0.005  # 5ms converted to seconds
    assert processor.small_batch_threshold == 8


@pytest.mark.asyncio
async def test_async_batch_processor_call_single_item():
    async def dummy_batch_func(items):
        return [item * 2 for item in items]

    processor = AsyncBatchProcessor(dummy_batch_func)
    
    result = await processor(5)
    assert result == 10


@pytest.mark.asyncio
async def test_async_batch_processor_call_multiple_items():
    async def dummy_batch_func(items):
        return [item * 2 for item in items]

    processor = AsyncBatchProcessor(dummy_batch_func)
    
    results = await processor([1, 2, 3, 4, 5])
    assert results == [2, 4, 6, 8, 10]


@pytest.mark.asyncio
async def test_async_batch_processor_prioritize():
    async def dummy_batch_func(items):
        return items

    processor = AsyncBatchProcessor(dummy_batch_func, small_batch_threshold=3, priority_strategy=PriorityStrategy.PRIORITY)
    
    assert processor._determine_priority([1, 2]) == [0, 0]
    assert processor._determine_priority([1, 2, 3, 4]) == [1, 1, 1, 1]

@pytest.mark.asyncio
async def test_async_batch_processor_stats():
    async def dummy_batch_func(items):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return items

    processor = AsyncBatchProcessor(dummy_batch_func, batch_size=2)
    await processor([1, 2, 3, 4])
    
    stats = processor.stats
    assert stats.total_batches == 2
    assert stats.total_processed == 4
    assert stats.avg_batch_size == 2.0
    assert 0.1 <= stats.avg_processing_time < 0.2


@pytest.mark.asyncio
async def test_async_dynamically_decorator():
    @dynamically(batch_size=3, timeout_ms=10.0)
    async def batch_multiply(items):
        return [item * 2 for item in items]

    results = await asyncio.gather(*[batch_multiply(i) for i in range(10)])
    
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


@pytest.mark.asyncio
async def test_async_batch_processor_exception_handling():
    async def faulty_batch_func(items):
        raise ValueError("Test error")

    processor = AsyncBatchProcessor(faulty_batch_func)
    
    with pytest.raises(ValueError, match="Test error"):
        await processor(5)


@pytest.mark.asyncio
async def test_async_batch_processor_concurrent_calls():
    async def slow_batch_func(items):
        await asyncio.sleep(0.1)
        return [item * 2 for item in items]

    processor = AsyncBatchProcessor(slow_batch_func, batch_size=5, timeout_ms=50.0)

    results = await asyncio.gather(*[processor(i) for i in range(10)])

    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 10
