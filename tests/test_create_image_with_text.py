from text_generation import create_image_with_text


def test_image_file_is_created(tmp_path, ubuntu_sans_font):
    """When calling create_image_with_text a PNG file should be created."""

    text = "Bures boahtin!"
    color_pair = ((0, 0, 0), (255, 255, 255))
    top_margin = 10
    bottom_margin = 20
    left_margin = 30
    right_margin = 40
    output_path = tmp_path / "output.png"

    create_image_with_text(
        text=text,
        font=ubuntu_sans_font,
        color_pair=color_pair,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
        left_margin=left_margin,
        right_margin=right_margin,
        output_path=output_path,
    )

    assert output_path.exists()
