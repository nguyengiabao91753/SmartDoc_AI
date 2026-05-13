from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """
    Canonical chunk configuration used across the whole project.

    Only supports:
        - chunk_size
        - chunk_overlap
    """

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self):
        if not (300 <= self.chunk_size <= 1200):
            raise ValueError("chunk_size must be within 300..1200")

        if not (1 <= self.chunk_overlap <= 10):
            raise ValueError("chunk_overlap must be within 1..10")