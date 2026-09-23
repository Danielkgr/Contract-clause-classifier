"""
Splitting contracts into chunks and labelling chunks from CUAD answer spans.

CUAD contracts run from a few thousand to over 300,000 characters, and the
clauses sit throughout them, so neither arm can look at a fixed prefix.
"""

from typing import Dict, List, Sequence, Tuple

Span = Tuple[int, int]


def char_chunks(text: str, size: int, overlap: int) -> List[Span]:
    """Split text into (start, end) character ranges of at most size characters.

    Consecutive chunks overlap by about overlap characters, so a clause that
    crosses a boundary appears whole in at least one chunk when it is shorter
    than the overlap.  Boundaries move back to the nearest whitespace where
    one exists in the last tenth of the chunk.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap <= size // 2:
        raise ValueError("overlap must be at least 0 and at most half of size")
    if not text:
        return []

    chunks = []
    start = 0
    while True:
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", end - size // 10, end)
            cut = max(cut, text.rfind("\n", end - size // 10, end))
            if cut > start:
                end = cut
        chunks.append((start, end))
        if end >= len(text):
            return chunks
        start = max(end - overlap, start + 1)


def overlaps(window: Span, spans: Sequence[Span]) -> bool:
    """True if the window shares at least one character with any span."""
    start, end = window
    return any(s < end and start < e for s, e in spans)


def window_labels(
    windows: Sequence[Span],
    spans_by_type: Dict[str, Sequence[Span]],
    clause_types: Sequence[str],
) -> List[List[int]]:
    """Label each window 1 or 0 for each clause type by span overlap."""
    return [
        [1 if overlaps(w, spans_by_type.get(ct, ())) else 0 for ct in clause_types]
        for w in windows
    ]
