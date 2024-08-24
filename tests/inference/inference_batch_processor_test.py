import asyncio

import pytest
import time
import numpy as np
import torch
from batch.inference.model_batch_processor import ModelBatchProcessor, dynamically
from batch.types import BatchProcessorStats, Feature


def test_batch_processor_initialization():
    def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = ModelBatchProcessor(dummy_batch_func, batch_size=32, timeout_ms=5.0, small_batch_threshold=8)
    
    assert processor.batch_func == dummy_batch_func
    assert processor.batch_queue._batch_size == 32
    assert processor.batch_queue._timeout == 0.005  # 5ms converted to seconds
    assert processor.small_batch_threshold == 8
    assert not processor._running
    assert processor._thread is None


def test_batch_processor_start_and_shutdown():
    def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = ModelBatchProcessor(dummy_batch_func)
    
    processor.start()
    assert processor._running
    assert processor._thread is not None
    
    processor.shutdown()
    assert not processor._running
    assert not processor._thread.is_alive()


def test_batch_processor_call():
    def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    processor = ModelBatchProcessor(dummy_batch_func)
    result = processor(input=np.array([[1, 2, 3], [4, 5, 6]]))
    np.testing.assert_array_equal(result, np.array([[2, 4, 6], [8, 10, 12]]))


def test_batch_processor_prioritize():
    def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = ModelBatchProcessor(dummy_batch_func, small_batch_threshold=3)
    
    assert processor._determine_priority([{"input": np.array([[1, 2], [3, 4]])}, {"input": np.array([[5, 6], [7, 8]])}]) == [0, 0]
    assert processor._determine_priority([{"input": np.array([[1]])}, {"input": np.array([[2]])}, {"input": np.array([[3]])}, {"input": np.array([[4]])}]) == [1, 1, 1, 1]

def test_batch_processor_stats():
    def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        time.sleep(0.1)  # Simulate some processing time
        return features["input"]

    processor = ModelBatchProcessor(dummy_batch_func, batch_size=2)
    processor(input=np.array([[1, 2], [3, 4]]))
    processor(input=np.array([[5, 6], [7, 8]]))
    
    stats = processor.stats
    assert isinstance(stats, BatchProcessorStats)
    assert stats.total_processed == 4
    assert 0.1 <= stats.avg_processing_time <= 0.2
    assert stats.queue_size == 0


def test_dynamically_decorator():
    @dynamically(batch_size=3, timeout_ms=10.0)
    def batch_multiply(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    results = [batch_multiply(input=torch.tensor([[i]])) for i in range(5)]
    
    for i, result in enumerate(results):
        assert torch.equal(result, torch.tensor([[i * 2]]))

def test_batch_processor_exception_handling():
    def faulty_batch_func(features: dict[str, Feature]) -> Feature:
        raise ValueError("Test error")

    processor = ModelBatchProcessor(faulty_batch_func)
    
    with pytest.raises(ValueError, match="Test error"):
        processor(input=np.array([[1, 2, 3], [4, 5, 6]]))

def test_batch_processor_concurrent_calls():
    def slow_batch_func(features: dict[str, Feature]) -> Feature:
        time.sleep(0.1)
        return features["input"] * 2

    processor = ModelBatchProcessor(slow_batch_func, batch_size=5, timeout_ms=50.0)

    import threading

    results = []
    def worker():
        results.append(processor(input=np.array([[1]])))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for result in results:
        np.testing.assert_array_equal(result, np.array([[2]]))
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 10

def test_batch_processor_with_pad_tokens():
    def pad_aware_batch_func(features: dict[str, Feature]) -> Feature:
        return np.sum(features["input"], axis=1)

    processor = ModelBatchProcessor(pad_aware_batch_func, batch_size=2, pad_tokens={"input": 0})
    
    result1 = processor(input=np.array([[1, 2, 3]]))
    result2 = processor(input=np.array([[4, 5]]))
    
    np.testing.assert_array_equal(result1, np.array([6]))
    np.testing.assert_array_equal(result2, np.array([9]))  # [4, 5, 0] due to padding

@pytest.mark.asyncio
async def test_batch_processor_acall():
    def batch_multiply(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    processor = ModelBatchProcessor(batch_multiply, batch_size=3, timeout_ms=10.0)

    results = await asyncio.gather(*[processor.acall(input=np.array([[i]])) for i in range(5)])
    
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, np.array([[i * 2]]))

@pytest.mark.asyncio
async def test_batch_processor_acall_multiple():
    def batch_multiply(features: dict[str, Feature]) -> Feature:
        return features["input"] * 2

    processor = ModelBatchProcessor(batch_multiply, batch_size=3, timeout_ms=10.0)

    result = await processor.acall(input=np.array([[1], [2], [3], [4], [5]]))
    np.testing.assert_array_equal(result, np.array([[2], [4], [6], [8], [10]]))

@pytest.mark.asyncio
async def test_batch_processor_acall_exception_handling():
    def faulty_batch_func(features: dict[str, Feature]) -> Feature:
        raise ValueError("Test error")

    processor = ModelBatchProcessor(faulty_batch_func)

    with pytest.raises(ValueError, match="Test error"):
        await processor.acall(input=np.array([[1]]))

@pytest.mark.asyncio
async def test_batch_processor_acall_concurrent():
    def slow_batch_func(features: dict[str, Feature]) -> Feature:
        time.sleep(0.1)
        return features["input"] * 2

    processor = ModelBatchProcessor(slow_batch_func, batch_size=5, timeout_ms=50.0)

    results = await asyncio.gather(*[processor.acall(input=np.array([[i]])) for i in range(10)])

    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, np.array([[i * 2]]))
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 10
