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


@pytest.mark.asyncio
async def test_async_model_batch_processor_with_padding_side_left():
    """Test AsyncModelBatchProcessor with left padding."""
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return features["input_ids"]

    processor = AsyncModelBatchProcessor(
        batch_func, 
        batch_size=2, 
        pad_tokens={"input_ids": 0},
        padding_side="left"
    )
    
    # Use asyncio.gather to ensure concurrent execution
    results = await asyncio.gather(
        processor(input_ids=np.array([[1, 2, 3]])),
        processor(input_ids=np.array([[4, 5]]))
    )
    
    # Both inputs should be batched together and padded to length 3
    # The second input should be left-padded: [4, 5] -> [0, 4, 5]
    expected_results = [
        np.array([[1, 2, 3]]),
        np.array([[0, 4, 5]])  # Left-padded
    ]
    
    # Sort results by first element to ensure consistent ordering
    results = sorted(results, key=lambda x: x[0, 0])
    expected_results = sorted(expected_results, key=lambda x: x[0, 0])
    
    for result, expected in zip(results, expected_results):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
async def test_async_model_batch_processor_with_padding_side_right():
    """Test AsyncModelBatchProcessor with explicit right padding."""
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return features["input_ids"]

    processor = AsyncModelBatchProcessor(
        batch_func, 
        batch_size=3, 
        pad_tokens={"input_ids": 0},
        padding_side="right"
    )
    
    # Process different length sequences concurrently
    results = await asyncio.gather(
        processor(input_ids=np.array([[1, 2, 3]])),
        processor(input_ids=np.array([[4, 5]]))
    )
    
    # The results should be padded correctly
    expected_results = [
        np.array([[1, 2, 3]]),
        np.array([[4, 5, 0]])
    ]
    
    # Check that we got the expected results (in any order)
    assert len(results) == 2
    found_first = any(np.array_equal(r, expected_results[0]) for r in results)
    found_second = any(np.array_equal(r, expected_results[1]) for r in results)
    assert found_first and found_second, f"Expected results not found in {results}"


@pytest.mark.asyncio
async def test_async_model_batch_processor_multiple_keys_left_padding():
    """Test AsyncModelBatchProcessor with multiple keys and left padding."""
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"]
        }

    processor = AsyncModelBatchProcessor(
        batch_func, 
        batch_size=2, 
        pad_tokens={"input_ids": 0, "attention_mask": 0},
        padding_side="left"
    )
    
    # Use asyncio.gather to ensure concurrent execution
    results = await asyncio.gather(
        processor(
            input_ids=np.array([[1, 2]]),
            attention_mask=np.array([[1, 1]])
        ),
        processor(
            input_ids=np.array([[3, 4, 5]]),
            attention_mask=np.array([[1, 1, 1]])
        )
    )
    
    # Both inputs should be batched together and padded to length 3
    # The first input should be left-padded: [1, 2] -> [0, 1, 2]
    expected_results = [
        {
            "input_ids": np.array([[0, 1, 2]]),
            "attention_mask": np.array([[0, 1, 1]])
        },
        {
            "input_ids": np.array([[3, 4, 5]]),
            "attention_mask": np.array([[1, 1, 1]])
        }
    ]
    
    # Sort results by first element to ensure consistent ordering
    results = sorted(results, key=lambda x: x["input_ids"][0, 0])
    expected_results = sorted(expected_results, key=lambda x: x["input_ids"][0, 0])
    
    for result, expected in zip(results, expected_results):
        np.testing.assert_array_equal(result["input_ids"], expected["input_ids"])
        np.testing.assert_array_equal(result["attention_mask"], expected["attention_mask"])


@pytest.mark.asyncio
async def test_async_dynamically_decorator_with_padding_side_left():
    """Test async dynamically decorator with left padding."""
    @dynamically(batch_size=2, pad_tokens={"input_ids": 0}, padding_side="left")
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return features["input_ids"]

    # Use asyncio.gather to ensure concurrent execution
    results = await asyncio.gather(
        batch_func(input_ids=np.array([[1, 2, 3]])),
        batch_func(input_ids=np.array([[4, 5]]))
    )
    
    # Both inputs should be batched together and padded to length 3
    # The second input should be left-padded: [4, 5] -> [0, 4, 5]
    expected_results = [
        np.array([[1, 2, 3]]),
        np.array([[0, 4, 5]])  # Left-padded
    ]
    
    # Sort results by first element to ensure consistent ordering
    results = sorted(results, key=lambda x: x[0, 0])
    expected_results = sorted(expected_results, key=lambda x: x[0, 0])
    
    for result, expected in zip(results, expected_results):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
async def test_async_dynamically_decorator_with_padding_side_right():
    """Test async dynamically decorator with right padding."""
    @dynamically(batch_size=2, pad_tokens={"input_ids": 0}, padding_side="right")
    async def batch_func(features: dict[str, Feature]) -> Feature:
        return features["input_ids"]

    # Use asyncio.gather to ensure concurrent execution
    results = await asyncio.gather(
        batch_func(input_ids=np.array([[1, 2, 3]])),
        batch_func(input_ids=np.array([[4, 5]]))
    )
    
    # Both inputs should be batched together and padded to length 3
    # The second input should be right-padded: [4, 5] -> [4, 5, 0]
    expected_results = [
        np.array([[1, 2, 3]]),
        np.array([[4, 5, 0]])  # Right-padded
    ]
    
    # Sort results by first element to ensure consistent ordering
    results = sorted(results, key=lambda x: x[0, 0])
    expected_results = sorted(expected_results, key=lambda x: x[0, 0])
    
    for result, expected in zip(results, expected_results):
        np.testing.assert_array_equal(result, expected)


@pytest.mark.asyncio
async def test_async_model_batch_processor_padding_side_initialization():
    """Test that padding_side is properly stored during initialization."""
    async def dummy_batch_func(features: dict[str, Feature]) -> Feature:
        return features["input"]

    processor = AsyncModelBatchProcessor(
        dummy_batch_func, 
        batch_size=32, 
        timeout_ms=5.0, 
        small_batch_threshold=8,
        padding_side="left"
    )
    
    assert processor.padding_side == "left"
    
    # Test default
    processor_default = AsyncModelBatchProcessor(dummy_batch_func)
    assert processor_default.padding_side == "right"


@pytest.mark.asyncio
async def test_async_model_batch_processor_concurrent_calls_with_padding():
    """Test AsyncModelBatchProcessor with concurrent calls and left padding."""
    async def batch_func(features: dict[str, Feature]) -> Feature:
        await asyncio.sleep(0.01)  # Simulate async processing
        return features["input_ids"]

    processor = AsyncModelBatchProcessor(
        batch_func, 
        batch_size=5, 
        timeout_ms=50.0,
        pad_tokens={"input_ids": 0},
        padding_side="left"
    )
    
    # Create inputs of different lengths
    inputs = [
        np.array([[1, 2, 3]]),
        np.array([[4, 5]]),
        np.array([[6, 7, 8, 9]]),
        np.array([[10, 11]]),
        np.array([[12, 13, 14]])
    ]
    
    # Execute concurrent calls
    results = await asyncio.gather(*[
        processor(input_ids=inp) for inp in inputs
    ])
    
    # Verify results - all should be left-padded to max length (4)
    expected = [
        np.array([[0, 1, 2, 3]]),
        np.array([[0, 0, 4, 5]]),
        np.array([[6, 7, 8, 9]]),
        np.array([[0, 0, 10, 11]]),
        np.array([[0, 12, 13, 14]])
    ]
    
    for result, exp in zip(results, expected):
        np.testing.assert_array_equal(result, exp)
