import pytest
import numpy as np
import torch
from batched.inference.helper import stack_features

@pytest.mark.parametrize("matrix_builder", [np.array, torch.tensor])
def test_stack_and_unstack_features(matrix_builder):
    # Prepare test data
    inputs = [
        {"input_ids": matrix_builder([1, 2, 3]), 
         "attention_mask": matrix_builder([1, 1, 1]), 
         "features": matrix_builder([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])},
        {"input_ids": matrix_builder([4, 5]), 
         "attention_mask": matrix_builder([1, 1]), 
         "features": matrix_builder([[[0.9, 1.0], [1.1, 1.2]]])},
        {"input_ids": matrix_builder([6, 7, 8, 9]), 
         "attention_mask": matrix_builder([1, 1, 1, 1]), 
         "features": matrix_builder([[[1.3, 1.4], [1.5, 1.6]], [[1.7, 1.8], [1.9, 2.0]], [[2.1, 2.2], [2.3, 2.4]]])},
    ]
    pad_tokens = {"input_ids": 0, "attention_mask": 0, "features": 0.0}

    # Test stack_features
    stacked = stack_features(inputs, pad_tokens)
    
    assert set(stacked.keys()) == {"input_ids", "attention_mask", "features"}
    assert stacked["input_ids"].shape == (3, 4)
    assert stacked["attention_mask"].shape == (3, 4)
    assert stacked["features"].shape == (3, 3, 2, 2)
    
    expected_input_ids = matrix_builder([
        [1, 2, 3, 0],
        [4, 5, 0, 0],
        [6, 7, 8, 9]
    ])
    expected_attention_mask = matrix_builder([
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1]
    ])
    expected_features = matrix_builder([
        [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]], [[0.0, 0.0], [0.0, 0.0]]],
        [[[0.9, 1.0], [1.1, 1.2]], [[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]],
        [[[1.3, 1.4], [1.5, 1.6]], [[1.7, 1.8], [1.9, 2.0]], [[2.1, 2.2], [2.3, 2.4]]]
    ])
    
    if matrix_builder == torch.tensor:
        assert torch.equal(stacked["input_ids"], expected_input_ids)
        assert torch.equal(stacked["attention_mask"], expected_attention_mask)
        assert torch.allclose(stacked["features"], expected_features)
    else:
        np.testing.assert_array_equal(stacked["input_ids"], expected_input_ids)
        np.testing.assert_array_equal(stacked["attention_mask"], expected_attention_mask)
        np.testing.assert_array_almost_equal(stacked["features"], expected_features)

@pytest.mark.parametrize("matrix_builder", [np.array, torch.tensor])
def test_stack_features_different_dtypes(matrix_builder):
    inputs = [
        {"float_feature": matrix_builder([1.0, 2.0]), 
         "int_feature": matrix_builder([1, 2]), 
         "features": matrix_builder([[[0.1, 0.2], [0.3, 0.4]]])},
        {"float_feature": matrix_builder([3.0, 4.0]), 
         "int_feature": matrix_builder([3, 4]), 
         "features": matrix_builder([[[0.5, 0.6], [0.7, 0.8]]])},
    ]
    pad_tokens = {"float_feature": 0.0, "int_feature": 0, "features": 0.0}

    stacked = stack_features(inputs, pad_tokens)
    
    if matrix_builder == torch.tensor:
        assert stacked["float_feature"].dtype in [torch.float32, torch.float64]
        assert stacked["int_feature"].dtype == torch.int64
        assert stacked["features"].dtype in [torch.float32, torch.float64]
    else:
        assert stacked["float_feature"].dtype in [np.float32, np.float64]
        assert stacked["int_feature"].dtype == np.int64
        assert stacked["features"].dtype in [np.float32, np.float64]



