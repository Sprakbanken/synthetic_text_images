import pytest

from synthetic_ocr_data.text_processing import TextLine, add_transformed_text_lines


def test_uppercase_lines_are_added():
    """The function should add uppercase lines to the text when str.upper is given"""
    text = "Bourre beaivvi!"
    chunks = [TextLine(text, "SOME ID")]
    transformed_text = add_transformed_text_lines(chunks, str.upper)
    transformed_text_lines = tuple(chunk.text for chunk in transformed_text)

    assert transformed_text_lines == ("Bourre beaivvi!", "BOURRE BEAIVVI!")


@pytest.mark.parametrize("text_id", ["SOME ID", "ANOTHER ID"])
def test_tranformed_lines_have_same_id_as_untransformed(text_id):
    """The transformed lines should have the same id as their untransformed counterparts"""

    text = "Bourre beaivvi!"
    chunks = [TextLine(text, text_id)]
    transformed_text = tuple(add_transformed_text_lines(chunks, str.upper))

    assert transformed_text[0].text_line_id == text_id
    assert transformed_text[1].text_line_id == text_id


def test_if_transform_makes_no_difference_its_not_added():
    """If the transform function does not change the text, the transformed line is not added"""
    text = "BOURRE BEAIVVI!"
    chunks = [TextLine(text, "SOME ID")]
    transformed_text = tuple(add_transformed_text_lines(chunks, str.upper))
    assert len(transformed_text) == 1
