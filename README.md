
# Batch

The Batch API provides a flexible and efficient way to process multiple requests in a batch. It is designed to optimize throughput while maintaining a low-latency experience, especially useful in scenarios where you need to handle a high volume of requests simultaneously.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Basic Example](#basic-example)
  - [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the Batch API, you can use pip:

```bash
pip install batch
```

## Usage

### Basic Example

Below is a basic example of how to use the Batch API to process text data in batches.

```diff
   from sentence_transformers import SentenceTransformer
   import numpy as np
   import time
+  import batch

   class SentenceEmbedder:
      def __init__(self, model_name='all-MiniLM-L6-v2'):
         self.model = SentenceTransformer(model_name)

+     @batch.dynamically(batch_size=32, timeout_ms=50.0)
      def embed_sentences(self, sentences: list[str]) -> list[np.ndarray]:
         # Convert sentences to embeddings
         embeddings = self.model.encode(sentences)
         return [embedding for embedding in embeddings]

   # Create an instance of SentenceEmbedder
   embedder = SentenceEmbedder()

   # Embed single sentences
   single_sent = "This is a test sentence."
   embedding = embedder.embed_sentences(single_sent)

   print("Single sentence embedding shapes:")
   print(f"Embedding shape: {embedding.shape}")

   # Embed a batch of 1000 sentences
   batch_sentences = [f"This is test sentence number {i}." for i in range(1000)]

   start_time = time.time()
   batch_embeddings = embedder.embed_sentences(batch_sentences)
   end_time = time.time()

   print("\nBatch embedding shapes:")
   for i, embedding in enumerate(batch_embeddings[:5]):  # Print only first 5 for brevity
      print(f"Embedding {i+1} shape: {embedding.shape}")

   print(f"\nTime taken to embed 1000 sentences: {end_time - start_time:.4f} seconds")

   # Check the statistics
   print("\nBatch processing statistics:")
   print(embedder.embed_sentences.stats)
```

### Advanced Usage

For more advanced usage, such as customizing batch size and timeout dynamically, the Batch API provides decorators that allow fine-grained control over the batching process.

- **Batch Size**: You can specify the number of requests to group together in a single batch.
- **Timeout**: The maximum time to wait for more requests before processing the batch.
- **Small Batch Threshold**: The threshold to give more priority to smaller batches.
- **Pad Token**: The token to use for padding when batching tensors, only for `@inference.dynamically`.

For example:

```python
@batch.dynamically(batch_size=50, timeout_ms=5.0)
def custom_batch_function(data):
    # Custom processing logic here
    pass
```

### API Reference

The API offers several key decorators:

- `@batch.dynamically`: Allows dynamic batching based on specified parameters such as `batch_size` and `timeout_ms`
- The decorated method should:
  - Take in a list of items (`list[T]`)
  - Return a list of results (`list[U]`) of the same length.


```python
import batch

@batch.dynamically(batch_size=64, timeout_ms=20.0, small_batch_threshold=10)
def my_function(items: list[T]) -> list[U]:
   # Custom processing logic here
   return [item * 2 for item in items]

# Allow single item
my_function(2)

# Allow batch of items
my_function([2, 3, 4])

# Support stat checking
print(my_function.stats)
```

- `@batch.inference.dynamically`: Allows dynamic batching for inference tasks, handling numpy arrays and tensors with padding.
- The decorated method should:
  - Take in a dictionary of tensors or numpy arrays (`ModelFeatures`). Each tensor or numpy array is a value batch of a single feature, and the keys are the feature names.
  - Return a tensor or numpy array (`ModelOutputs`). Each row is a single inference result.
  - `ModelFeatures[feature_name].shape[0] == ModelOutputs.shape[0]`

```python
from batch import inference

@inference.dynamically(pad_token={"input_ids": 0})
def my_inference_function(features: ModelFeatures) -> ModelOutputs:
   # input_ids = features["input_ids"]
   # attention_mask = features["attention_mask"]
   # token_type_ids = features["token_type_ids"]

   logits = model(**features)
   return logits

my_inference_function(data)

print(my_inference_function.stats)
```



## Contributing

Contributions are welcome! Please feel free to submit a pull request or report an issue on GitHub.

## License

This project is licensed under the MIT License.
