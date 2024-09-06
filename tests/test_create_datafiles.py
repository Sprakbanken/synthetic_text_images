import random
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageChops, ImageFont

from text_generation import create_datafiles, create_image_with_text


@pytest.mark.parametrize(
    "text_lines", [["Bures boahtin!"],["Bures boahtin!", "Velkommen!", "Welcome!"]]
)
def test_one_file_per_line_is_created(tmp_path: Path, text_lines: list[str]):
    """When calling create_datafiles one file per line should be created."""

    fonts = [ImageFont.load_default()]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    create_datafiles(
        text_lines=text_lines,
        output_dir=tmp_path,
        fonts=fonts,
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    assert len(list(tmp_path.glob("images/*.png"))) == len(text_lines)


@pytest.mark.parametrize(
    "text_lines", [["Bures boahtin!"],["Bures boahtin!", "Velkommen!", "Welcome!"]]
)
def test_csv_file_with_right_header_and_rownumber_is_created(tmp_path: Path, text_lines: list[str]):
    """When calling create_datafiles a CSV file with the right header and rownumber should be created."""

    fonts = [ImageFont.load_default()]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    create_datafiles(
        text_lines=text_lines,
        output_dir=tmp_path,
        fonts=fonts,
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
    expected_columns = [
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
        "output_path",
    ]
    assert len(columns) == len(expected_columns)
    for column in expected_columns:
        assert column in columns

def test_image_can_be_reproduced_from_csv_file(tmp_path: Path, ubuntu_sans_font: ImageFont.FreeTypeFont):
    """It should be possible to reproduce the image from the CSV file."""

    text_lines = ["Bures boahtin!"]
    fonts = [ubuntu_sans_font]
    color_pairs = [((0, 0, 0), (255, 255, 255))]
    top_margins = [10]
    bottom_margins = [20]
    left_margins = [30]
    right_margins = [40]
    rng = random.Random(42)
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    create_datafiles(
        text_lines=text_lines,
        output_dir=output_dir,
        fonts=fonts,
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
    
    def parse_rgb_string(rgb_string: str):
        return tuple(int(number) for number in rgb_string.replace("(", "").replace(")", "").split(", "))

    parameter_dict = {
        "text": series["text"],
        "font": ImageFont.truetype(series["font_path"], series["font_size"]),
        "color_pair": (
            parse_rgb_string(series["text_color"]),
            parse_rgb_string(series["background_color"]),
        ),
        "top_margin": series["top_margin"],
        "bottom_margin": series["bottom_margin"],
        "left_margin": series["left_margin"],
        "right_margin": series["right_margin"],
    }
    unique_id = series["unique_id"]

    recreated_image, _ = create_image_with_text(**parameter_dict)
    original_image = Image.open(output_dir / f"images/{unique_id}.png")

    # Check that the images are the same
    assert original_image.size == recreated_image.size
    assert ImageChops.difference(original_image, recreated_image).getbbox() is None
   
