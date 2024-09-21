import pytest

from text_generation.text_processing import restart_chunk


def test_chunk_restarts_with_candidate_word_if_short_enough():
    """The function should return the candidate word when it is short enough"""
    assert restart_chunk("sátni", 10) == "sátni"


def test_chunk_restarts_with_empty_string_if_candidate_word_too_long():
    """The function should return an empty string when the candidate word is too long"""
    assert restart_chunk("sátni", 4) == ""
