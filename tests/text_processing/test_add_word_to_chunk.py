import pytest

from text_generation.text_processing import add_word_to_chunk


@pytest.mark.parametrize("candidate_word", ["sátni", "ord", "word"])
def test_word_is_added_to_empty_chunk(candidate_word):
    """The function should return the candidate word when the current chunk is empty."""

    assert add_word_to_chunk("", candidate_word) == candidate_word


def test_word_is_added_to_non_empty_chunk():
    """When the current chunk is not empty, the candidate word should be added with a space in between"""

    assert add_word_to_chunk("Buorre", "beaivvi!") == "Buorre beaivvi!"
