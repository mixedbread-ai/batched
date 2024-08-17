from dataclasses import dataclass, field


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0


@dataclass
class BatchProcessorConfig:
    timeout: float = 0.005
    max_batch_size: int = 32
    max_batch_tokens: int = -1
    small_batch_threshold: int = 4
    pad_tokens: dict[str, int] = field(default_factory=dict)


DEFAULT_CONFIG = BatchProcessorConfig()
