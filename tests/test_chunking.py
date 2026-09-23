"""Tests for contract chunking and window labelling."""

import pytest

from utils.chunking import char_chunks, overlaps, window_labels


def test_chunks_cover_the_whole_text_within_size():
    text = " ".join(f"word{i}" for i in range(5000))
    chunks = char_chunks(text, size=1000, overlap=100)
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(text)
    assert all(end - start <= 1000 for start, end in chunks)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start < prev_end  # consecutive chunks overlap, so nothing is skipped


def test_a_clause_shorter_than_the_overlap_lands_whole_in_some_chunk():
    filler = "x " * 3000
    clause = "This Agreement is governed by the laws of Victoria."
    text = filler + clause + " " + filler
    start = text.index(clause)
    chunks = char_chunks(text, size=1000, overlap=200)
    assert any(s <= start and start + len(clause) <= e for s, e in chunks)


def test_short_text_is_one_chunk():
    assert char_chunks("short contract", size=1000, overlap=100) == [(0, 14)]
    assert char_chunks("", size=1000, overlap=100) == []


def test_overlap_larger_than_half_the_size_is_rejected():
    with pytest.raises(ValueError):
        char_chunks("text", size=100, overlap=60)


def test_overlap_needs_a_shared_character():
    assert overlaps((10, 20), [(19, 30)])
    assert not overlaps((10, 20), [(20, 30)])
    assert not overlaps((10, 20), [])


def test_window_labels_follow_span_overlap():
    windows = [(0, 100), (80, 180), (160, 260)]
    spans = {"Governing Law": [(150, 170)], "Insurance": []}
    assert window_labels(windows, spans, ["Governing Law", "Insurance"]) == [
        [0, 0], [1, 0], [1, 0],
    ]
