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

    # ===== QUAN TRỌNG: ràng buộc theo sentence chunking thực tế =====
    # Trung bình 1 câu ~ 80–120 ký tự trong tài liệu VN/EN
    # Một chunk 600 ký tự thường chỉ có ~5–7 câu.
    # Nếu overlap > 40% số câu ước tính → vỡ pipeline.

    estimated_sentences = max(1, chunk_size // 100)

    max_safe_overlap = max(1, int(estimated_sentences * 0.4))

    if overlap > max_safe_overlap:
        raise ValueError(
            f"overlap={overlap} too large for chunk_size={chunk_size}. "
            f"Recommended overlap ≤ {max_safe_overlap} sentences."
        )

    return ChunkParams(
        chunk_size=chunk_size,
        chunk_overlap_sentences=overlap,
    )