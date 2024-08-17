from .batch_processor import BatchProcessor, BatchProcessorConfig, dynamic_batching

dynamically = dynamic_batching

__all__ = ["dynamically", "BatchProcessor", "dynamic_batching", "BatchProcessorConfig"]
