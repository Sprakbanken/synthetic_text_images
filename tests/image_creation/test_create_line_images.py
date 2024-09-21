import random
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageChops, ImageFont

from text_generation.fonts import FontInfo
from text_generation.image_creation import create_line_image, create_line_images
from text_generation.text_processing import TextLine
from text_generation.utils import parse_int_tuple


@pytest.mark.parametrize(
    "text_lines", [["Bures boahtin!"], ["Bures boahtin!", "Velkommen!", "Welcome!"]]
)
def test_one_file_per_line_is_created(
    tmp_path: Path, text_lines: list[str], ubuntu_sans_font_info: FontInfo
):
    """When calling create_datafiles one file per line should be created."""

    fonts = [ubuntu_sans_font_info]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    create_line_images(
        text_lines=[TextLine(line) for line in text_lines],
        output_dir=tmp_path,
        fonts=fonts,
        size_range=range(40, 100),
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    assert len(list(tmp_path.glob("line_images/*.png"))) == len(text_lines)


@pytest.mark.parametrize(
    "text_lines", [["Bures boahtin!"], ["Bures boahtin!", "Velkommen!", "Welcome!"]]
)
def test_csv_file_with_right_header_and_rownumber_is_created(
    tmp_path: Path, text_lines: list[str], ubuntu_sans_font_info: FontInfo
):
    """When calling create_datafiles a CSV file with the right header and rownumber should be created."""

    fonts = [ubuntu_sans_font_info]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    create_line_images(
        text_lines=[TextLine(line) for line in text_lines],
        output_dir=tmp_path,
        fonts=fonts,
        size_range=range(40, 100),
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    csv_file = tmp_path / "metadata.csv"
    assert csv_file.exists()
    df = pd.read_csv(csv_file)
    assert df.shape[0] == len(text_lines)
    columns = df.columns.tolist()
    expected_columns = {
        "unique_id",
        "text",
        "font_path",
        "font_size",
        "text_color",
        "background_color",
        "top_margin",
        "bottom_margin",
        "left_margin",
        "right_margin",
        "bbox_left",
        "bbox_top",
        "bbox_right",
        "bbox_bottom",
        "image_width",
        "image_height",
        "undistorted_file_name",
        "text_line_id",
        "text_transform",
        "raw_text",
    }
    assert set(columns) == expected_columns


def test_image_can_be_reproduced_from_csv_file(tmp_path: Path, ubuntu_sans_font_info: FontInfo):
    """It should be possible to reproduce the image from the CSV file."""

    text_lines = [TextLine("Bures boahtin!")]
    fonts = [ubuntu_sans_font_info]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    create_line_images(
        text_lines=text_lines,
        output_dir=output_dir,
        fonts=fonts,
        size_range=range(40, 100),
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    csv_file = output_dir / "metadata.csv"
    assert csv_file.exists()

    series = pd.read_csv(csv_file).squeeze()

    parameter_dict = {
        "text": series["text"],
        "font": ImageFont.truetype(series["font_path"], series["font_size"]),
        "color_pair": (
            parse_int_tuple(series["text_color"]),
            parse_int_tuple(series["background_color"]),
        ),
        "top_margin": series["top_margin"],
        "bottom_margin": series["bottom_margin"],
        "left_margin": series["left_margin"],
        "right_margin": series["right_margin"],
    }
    unique_id = series["unique_id"]

    recreated_image, _ = create_line_image(**parameter_dict)
    original_image = Image.open(output_dir / f"line_images/{unique_id}.png")

    # Check that the images are the same
    assert original_image.size == recreated_image.size
    assert ImageChops.difference(original_image, recreated_image).getbbox() is None
