import asyncio
import pytest
from batched.aio.batch_generator import AsyncBatchGenerator, AsyncBatchItem


@pytest.mark.asyncio
async def test_async_batch_generator_initialization():
    generator = AsyncBatchGenerator(batch_size=32, timeout_ms=5.0)
    assert generator._batch_size == 32
    assert generator._timeout == 0.005  # 5ms converted to seconds
    assert len(generator) == 0


@pytest.mark.asyncio
async def test_async_batch_generator_extend():
    generator = AsyncBatchGenerator(batch_size=3)
    items = [
        AsyncBatchItem(content=i, future=asyncio.Future(), priority=1)
        for i in range(5)
    ]
    await generator.extend(items)
    assert len(generator) == 5


@pytest.mark.asyncio
async def test_async_batch_generator_optimal_batches():
    generator = AsyncBatchGenerator(batch_size=3, timeout_ms=10.0)
    items = [
        AsyncBatchItem(content=i, future=asyncio.Future(), priority=1)
        for i in range(5)
    ]
    await generator.extend(items)

    batches = []
    async for batch in generator.optimal_batches():
        batches.append(batch)
        if len(batches) == 2:
            break

    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2


@pytest.mark.asyncio
async def test_async_batch_item_set_result():
    future = asyncio.Future()
    item = AsyncBatchItem(content="test", future=future, priority=1)
    item.set_result("result")
    assert future.done()
    assert await future == "result"


@pytest.mark.asyncio
async def test_async_batch_item_set_exception():
    future = asyncio.Future()
    item = AsyncBatchItem(content="test", future=future, priority=1)
    exception = ValueError("Test exception")
    item.set_exception(exception)
    assert future.done()
    with pytest.raises(ValueError, match="Test exception"):
        await future


@pytest.mark.asyncio
async def test_async_batch_item_priority():
    items = [
        AsyncBatchItem(content=1, future=asyncio.Future(), priority=2),
        AsyncBatchItem(content=2, future=asyncio.Future(), priority=1),
        AsyncBatchItem(content=3, future=asyncio.Future(), priority=3),
    ]
    generator = AsyncBatchGenerator(batch_size=3, sort_by_priority=True)
    await generator.extend(items)

    batch = await anext(generator.optimal_batches())
    assert [item.content for item in batch] == [2, 1, 3]


@pytest.mark.asyncio
async def test_async_batch_generator_timeout():
    generator = AsyncBatchGenerator(batch_size=3, timeout_ms=50.0)
    items = [AsyncBatchItem(content=i, future=asyncio.Future(), priority=1) for i in range(2)]
    await generator.extend(items)

    start_time = asyncio.get_event_loop().time()
    batch = await anext(generator.optimal_batches())
    end_time = asyncio.get_event_loop().time()

    assert len(batch) == 2
    assert 0.05 <= (end_time - start_time) < 0.1  # Allow for some timing inaccuracy
