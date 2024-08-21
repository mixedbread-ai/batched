import pytest
import time
from batch.batch_processor import BatchProcessor, dynamically
from batch.types import BatchProcessorStats

def test_batch_processor_initialization():
    def dummy_batch_func(items):
        return items

    processor = BatchProcessor(dummy_batch_func, batch_size=32, timeout_ms=5.0, small_batch_threshold=8)
    
    assert processor.batch_func == dummy_batch_func
    assert processor.batch_queue._batch_size == 32
    assert processor.batch_queue._timeout == 0.005  # 5ms converted to seconds
    assert processor.small_batch_threshold == 8
    assert not processor._running
    assert processor._thread is None

def test_batch_processor_start_and_shutdown():
    def dummy_batch_func(items):
        return items

    processor = BatchProcessor(dummy_batch_func)
    
    processor.start()
    assert processor._running
    assert processor._thread is not None
    
    processor.shutdown()
    assert not processor._running
    assert not processor._thread.is_alive()

def test_batch_processor_call_single_item():
    def dummy_batch_func(items):
        return [item * 2 for item in items]

    processor = BatchProcessor(dummy_batch_func)
    
    result = processor(5)
    assert result == 10

def test_batch_processor_call_multiple_items():
    def dummy_batch_func(items):
        return [item * 2 for item in items]

    processor = BatchProcessor(dummy_batch_func)
    
    results = processor([1, 2, 3, 4, 5])
    assert results == [2, 4, 6, 8, 10]

def test_batch_processor_prioritize():
    def dummy_batch_func(items):
        return items

    processor = BatchProcessor(dummy_batch_func, small_batch_threshold=3)
    
    assert processor.prioritize([1, 2]) == [0, 0]
    assert processor.prioritize([1, 2, 3, 4]) == [1, 1, 1, 1]

def test_batch_processor_stats():
    def dummy_batch_func(items):
        time.sleep(0.1)  # Simulate some processing time
        return items

    processor = BatchProcessor(dummy_batch_func, batch_size=2)
    
    processor([1, 2, 3, 4])
    
    stats = processor.stats
    assert isinstance(stats, BatchProcessorStats)
    assert stats.total_batches == 2
    assert stats.total_processed == 4
    assert 0.1 <= stats.avg_processing_time <= 0.2
    assert stats.queue_size == 0

def test_dynamically_decorator():
    @dynamically(batch_size=3, timeout_ms=10.0)
    def batch_multiply(items):
        return [item * 2 for item in items]

    results = [batch_multiply(i) for i in range(10)]
    
    assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

def test_batch_processor_exception_handling():
    def faulty_batch_func(items):
        raise ValueError("Test error")

    processor = BatchProcessor(faulty_batch_func)
    
    with pytest.raises(ValueError, match="Test error"):
        processor(5)

def test_batch_processor_concurrent_calls():
    def slow_batch_func(items):
        time.sleep(0.1)
        return [item * 2 for item in items]

    processor = BatchProcessor(slow_batch_func, batch_size=5, timeout_ms=50.0)

    import threading

    results = []
    def worker():
        results.append(processor(1))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [2] * 10
    assert processor.stats.total_batches == 2
    assert processor.stats.total_processed == 10

