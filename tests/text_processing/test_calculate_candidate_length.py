import pytest

from synthetic_ocr_data.text_processing import calculate_candidate_length


@pytest.mark.parametrize("candidate_word", ["sátni", "ord", "word"])
def test_calculate_candidate_length_when_current_chunk_is_empty(candidate_word):
    """The function should return the length of the candidate word when the current chunk is empty."""

    assert calculate_candidate_length("", candidate_word) == len(candidate_word)


@pytest.mark.parametrize("current_chunk", ["Bures", "Hello", "Hei"])
@pytest.mark.parametrize("candidate_word", ["sátni", "ord", "word"])
def test_calculate_candidate_length_when_current_chunk_is_not_empty(current_chunk, candidate_word):
    """When the current chunk is not empty, a space is needed, so the candidate length is one longer"""

    assert calculate_candidate_length(current_chunk, candidate_word) == len(
        current_chunk
    ) + 1 + len(candidate_word)


def test_calculate_candidate_length_correct_for_known_values():
    """The function should return the correct value for known inputs"""

    assert calculate_candidate_length("Hello", "world") == 11
    assert calculate_candidate_length("", "world") == 5
