from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")

BatchFunc = Callable[[list[T]], list[U]]


def _validate_batch_output(batch_inputs: list[T], batch_outputs: list[U]) -> None:
    """
    Validate that the batch output has the same length as the batch inputs.

    Args:
        batch_inputs (list): The batch inputs
        batch_outputs (list): The batch outputs

    Raises:
        ValueError: If the batch output length does not match the batch input length
    """
    if len(batch_inputs) != len(batch_outputs):
        msg = f"Batch output length ({len(batch_outputs)}) " f"does not match batch input length ({len(batch_inputs)})"
        raise ValueError(msg)


@dataclass
class BatchProcessorStats:
    queue_size: int = 0
    total_processed: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0

    def update(self, batch_size: int, processing_time: float) -> None:
        """
        Update the statistics based on the batch size and processing time.

        Args:
            batch_size (int): The size of the batch
            processing_time (float): The time taken to process the batch
        """
        self.total_processed += batch_size
        self.total_batches += 1
        self.avg_batch_size = self.total_processed / self.total_batches
        self.avg_processing_time = (
            self.avg_processing_time * (self.total_batches - 1) + processing_time
        ) / self.total_batches

    def clone(self, *, queue_size: int) -> BatchProcessorStats:
        """
        Clone the statistics object.

        Returns:
            BatchProcessorStats: The cloned statistics object
        """
        return BatchProcessorStats(
            queue_size=queue_size,
            total_processed=self.total_processed,
            total_batches=self.total_batches,
            avg_batch_size=self.avg_batch_size,
            avg_processing_time=self.avg_processing_time,
        )

