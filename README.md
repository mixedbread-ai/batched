
# Batched (Dynamic Batching)

The Batched API provides a flexible and efficient way to process multiple requests in a batch, with a primary focus on dynamic batching of inference workloads. It is designed to optimize throughput while maintaining a low-latency experience, especially useful in scenarios where you need to handle a high volume of requests simultaneously. It is designed for both async and sync execution.

![Batch Performance](/examples/inference_speed.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Why Dynamic Batching?

Dynamic batching is a technique that automatically groups multiple incoming inference requests into a single batch for processing. This is particularly beneficial for inference workloads, where processing multiple inputs together can significantly improve throughput and efficiency.

In machine learning models, dynamic batching matters because it optimizes hardware utilization, especially for GPUs and specialized AI hardware designed for parallel processing. By batching requests, we can fully leverage this parallel processing, leading to higher throughput. It also reduces overhead by amortizing fixed costs (such as data transfer and model initialization) across multiple requests, improving overall efficiency. Furthermore, dynamic batching enhances real-time performance by adapting to varying request rates, maintaining low latency during quiet periods while maximizing throughput during busy times.

This makes dynamic batching a crucial technique for deploying ML models in production environments where request patterns can be unpredictable and resource optimization is key.

## Installation

To install the Batched, you can use pip:

```bash
pip install batched
```

## Usage

### Basic Example

Below is a basic example of how to use the Batched API to process text data in batches.

```diff
   from sentence_transformers import SentenceTransformer
   import numpy as np
+  import batched

   class SentenceEmbedder:
      def __init__(self, model_name='mixedbread-ai/mxbai-embed-large-v1'):
         self.model = SentenceTransformer(model_name)

+     @batched.dynamically
      def embed_sentences(self, sentences: list[str]) -> list[np.ndarray]:
         # Convert sentences to embeddings
         return self.model.encode(sentences)

   # Create an instance of SentenceEmbedder
   embedder = SentenceEmbedder()

   # Embed single sentences
   single_sent = "This is a test sentence."
   embedding = embedder.embed_sentences(single_sent)
+  awaited_embedding = await embedder.embed_sentences.acall(single_sent)

   # Embed a batch of 1000 sentences
   batch_sentences = [f"This is test sentence number {i}." for i in range(1000)]
   batch_embeddings = embedder.embed_sentences(batch_sentences)
+  awaited_batch_embeddings = await embedder.embed_sentences.acall(batch_sentences)

   # Check the statistics
+  stats = embedder.embed_sentences.stats
```

### Advanced Usage

For more advanced usage, such as customizing batch size and timeout dynamically, the Batched API provides decorators that allow fine-grained control over the batching process.

- **Batch Size**: You can specify the max. number of requests to group together in a single batch.
- **Timeout**: The maximum time to wait for more requests before processing the batch.
- **Small Batch Threshold**: The threshold to give more priority to smaller batches.
- **Pad Token**: The token to use for padding when batching tensors, only for `@inference.dynamically` and `@aio.inference.dynamically`.

For example:

```python
@batched.dynamically(batch_size=64, timeout_ms=5.0, small_batch_threshold=2)
def custom_batch_function(data):
    # Custom processing logic here
    pass
```

### API Reference

The API offers both thread and asyncio implementations for batching general tasks and inference tasks:

#### Thread Implementation

- `@batched.dynamically`: Allows dynamic batching for general tasks (Both sync and async supported).
- The decorated method should:
  - Take in a list of items (`list[T]`)
  - Return a list of results (`list[U]`) of the same length.

```python
import batched


@batched.dynamically(batch_size=64)
def my_function(items: list[int]) -> list[str]:
  # Custom processing logic here
  return [f"{item * 2}" for item in items]

# Sync call with single item
my_function(2)

# Sync call with a batch of items
my_function([2, 3, 4])

# Call with asyncio
await my_function.acall(2)
await my_function.acall([2, 3, 4])

# Support stat checking
print(my_function.stats)
```

- `@batched.inference.dynamically`: Allows dynamic batching for inference tasks, handling numpy arrays and tensors with padding.
- The decorated method should:
  - Take in a dictionary of tensors or numpy arrays (`dict[str, Feature]`). `Feature` is a batch of item values for a single feature, and the keys are the feature names.
  - Return a tensor or numpy array (`Feature`). Each row is a single inference result.
  - `Feature` can be any of the following types: `np.ndarray`, `torch.Tensor`, `list[np.ndarray]` and `list[torch.Tensor]`.
  - `features[feature_name].shape[0] == outputs.shape[0]`

```python
from batched import inference
import torch

@inference.dynamically(pad_token={"input_ids": 0})
def my_inference_function(features: dict[str, torch.Tensor]) -> torch.Tensor:
  # input_ids = features["input_ids"]
  # attention_mask = features["attention_mask"]
  # token_type_ids = features["token_type_ids"]

  logits = model(**features)
  return logits

# Sync call
my_inference_function(data)

# Call with asyncio
await my_inference_function.acall(data)

print(my_inference_function.stats)
```

#### Asyncio Implementation

- `@batched.aio.dynamically`: Allows dynamic batching for general tasks using `asyncio`.
- The decorated method should:
  - Take in a list of items (`list[T]`)
  - Return a list of results (`list[U]`) of the same length.

```python
from batched import aio

@aio.dynamically(batch_size=64, timeout_ms=20.0, small_batch_threshold=10)
def my_function(items: list[int]) -> list[int]:  # can also be an async function: async def ...
  # Custom processing logic here
  return [item * 2 for item in items]


# Allow single item
await my_function(2)

# Allow batch of items
await my_function([2, 3, 4])

# Support stat checking
print(my_function.stats)
```

- `@batched.aio.inference.dynamically`: Allows dynamic batching for inference tasks, handling numpy arrays and tensors with padding, using `asyncio`.
- The decorated method should:
  - Take in a dictionary of tensors or numpy arrays (`dict[str, Feature]`). `Feature` is a batch of item values for a single feature, and the keys are the feature names.
  - Return a tensor or numpy array (`Feature`). Each row is a single inference result.
  - `Feature` can be any of the following types: `np.ndarray`, `torch.Tensor`, `list[np.ndarray]` and `list[torch.Tensor]`.
  - `features[feature_name].shape[0] == outputs.shape[0]`

```python
from batched import aio
import torch

@aio.inference.dynamically(pad_token={"input_ids": 0})
async def my_inference_function(features: dict[str, torch.Tensor]) -> list[torch.Tensor]:
  # input_ids = features["input_ids"]
  # attention_mask = features["attention_mask"]
  # token_type_ids = features["token_type_ids"]

  logits1 = await model1(**features)
  logits2 = await model2(**features)
  return [logits1, logits2]


await my_inference_function(data)

print(my_inference_function.stats)
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or report an issue on GitHub.

## Attribution

This project was inspired by the following projects:

- [Infinity](https://github.com/michaelfeil/infinity) by Michael Feil

## License

This project is licensed under the Apache License, Version 2.0.
