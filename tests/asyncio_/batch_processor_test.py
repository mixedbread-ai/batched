import asyncio
import pytest
from batch.asyncio_.batch_processor import BatchProcessor


@pytest.mark.asyncio
async def test_batch_processor_basic():
    async def batch_func(items):
        await asyncio.sleep(0.1)  # Simulate some processing time
        return [item * 2 for item in items]

    processor = BatchProcessor(batch_func, batch_size=3, timeout=1.0)

    # Test single item
    result = await processor([1])
    assert result == [2]

    # Test multiple items
    result = await processor([1, 2, 3, 4, 5])
    assert result == [2, 4, 6, 8, 10]

@pytest.mark.asyncio
async def test_batch_processor_timeout():
    async def slow_batch_func(items):
        await asyncio.sleep(2)  # Simulate slow processing
        return [item * 2 for item in items]

    processor = BatchProcessor(slow_batch_func, batch_size=5, timeout=0.5)

    # This should trigger the timeout and process items in smaller batches
    result = await processor([1, 2, 3, 4, 5])
    assert result == [2, 4, 6, 8, 10]

@pytest.mark.asyncio
async def test_batch_processor_concurrent():
    async def batch_func(items):
        await asyncio.sleep(0.1)
        return [item * 2 for item in items]

    processor = BatchProcessor(batch_func, batch_size=3, timeout=1.0)

    # Run multiple calls concurrently
    tasks = [
        processor([1, 2]),
        processor([3, 4, 5]),
        processor([6]),
    ]
    results = await asyncio.gather(*tasks)

    assert results == [[2, 4], [6, 8, 10], [12]]

@pytest.mark.asyncio
async def test_batch_processor_error_handling():
    async def faulty_batch_func(items):
        if any(item == 0 for item in items):
            raise ValueError("Cannot process item with value 0")
        return [item * 2 for item in items]

    processor = BatchProcessor(faulty_batch_func, batch_size=3, timeout=1.0)

    # This should raise an exception
    with pytest.raises(ValueError):
        await processor([1, 0, 2])

    # This should still work
    result = await processor([1, 2, 3])
    assert result == [2, 4, 6]
