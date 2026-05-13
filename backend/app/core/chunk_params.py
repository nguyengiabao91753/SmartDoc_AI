from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkParams:
    chunk_size: int
    chunk_overlap_sentences: int


def validate_chunk_params(
    chunk_size: int,
    chunk_overlap_sentences: int,
) -> ChunkParams:
    if chunk_size is None:
        raise ValueError("chunk_size is required")

    chunk_size = int(chunk_size)

    # UI spec
    if chunk_size < 300 or chunk_size > 1200:
        raise ValueError("chunk_size must be within 300..1200")

    if chunk_overlap_sentences is None:
        raise ValueError("chunk_overlap_sentences is required")

    overlap = int(chunk_overlap_sentences)

    # UI spec
    if overlap < 1 or overlap > 10:
        raise ValueError("chunk_overlap_sentences must be within 1..10")

    return ChunkParams(
        chunk_size=chunk_size,
        chunk_overlap_sentences=overlap,
    )
