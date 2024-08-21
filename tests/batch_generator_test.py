import pytest
import time
from batch.batch_generator import BatchGenerator, Item

def test_batch_generator_initialization():
    generator = BatchGenerator(batch_size=32, timeout_ms=5.0)
    
    assert generator._batch_size == 32
    assert generator._timeout == 0.005  # 5ms converted to seconds
    assert len(generator) == 0

def test_batch_generator_extend():
    generator = BatchGenerator(batch_size=2, timeout_ms=5.0)
    items = [Item(content=i) for i in range(5)]
    
    generator.extend(items)
    
    assert len(generator) == 5


def test_batch_generator_optimal_batches():
    generator = BatchGenerator(batch_size=2, timeout_ms=5.0)
    items = [Item(content=i) for i in range(5)]
    generator.extend(items)
    
    batches = []
    for batch in generator.optimal_batches():
        batches.append(batch)
        if len(batches) == 3:
            generator.stop()
    
    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1
    
    for batch in batches:
        for item in batch:
            assert isinstance(item, Item)

    with pytest.raises(StopIteration):
        next(generator.optimal_batches())

def test_batch_generator_timeout():
    generator = BatchGenerator(batch_size=2, timeout_ms=50.0)
    items = [Item(content=i) for i in range(1)]
    generator.extend(items)
    
    start_time = time.time()
    batch = next(generator.optimal_batches())
    end_time = time.time()
    
    assert len(batch) == 1
    assert 0.05 <= end_time - start_time <= 0.06  # Allow small margin for execution time

    generator.stop()
    with pytest.raises(StopIteration):
        next(generator.optimal_batches())

def test_batch_generator_prioritized_items():
    generator = BatchGenerator(batch_size=2, timeout_ms=5.0)
    items = [
        Item(content=1, priority=1),
        Item(content=2, priority=2),
        Item(content=3, priority=3),
        Item(content=4, priority=4),
    ]
    generator.extend(items)
    
    batches = []
    for batch in generator.optimal_batches():
        batches.append(batch)
        if len(batches) == 2:
            generator.stop()
    
    assert len(batches) == 2
    assert batches[0][0].content == 1
    assert batches[0][1].content == 2

    with pytest.raises(StopIteration):
        next(generator.optimal_batches())

def test_item_complete_and_get_result():
    item = Item(content="test")
    
    item.complete("result")
    
    assert item.get_result() == "result"

def test_item_exception():
    item = Item(content="test")
    
    item.set_exception(ValueError("Test exception"))
    
    with pytest.raises(ValueError, match="Test exception"):
        item.get_result()

def test_batch_generator_stop():
    generator = BatchGenerator(batch_size=2, timeout_ms=5.0)
    items = [Item(content=i) for i in range(5)]
    generator.extend(items)

    batches = []
    for batch in generator.optimal_batches():
        batches.append(batch)
        if len(batches) == 2:
            generator.stop()

    assert len(batches) == 2
    with pytest.raises(StopIteration):
        next(generator.optimal_batches())
