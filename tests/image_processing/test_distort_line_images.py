import filecmp
import random
from pathlib import Path

import pandas as pd
import pytest

from synthetic_ocr_data.augraphy_utils import create_scanned_book_pipeline
from synthetic_ocr_data.image_creation import create_line_images
from synthetic_ocr_data.image_processing import distort_line_images
from synthetic_ocr_data.text_processing import TextLine


@pytest.fixture
def example_line_images_directory(tmp_path: Path, ubuntu_sans_font_info) -> Path:
    lines = ["Boure beaivvi", "God dag", "Mo manná?", "Hvordan går det?"]
    lines = [TextLine(line) for line in lines]
    output_path = tmp_path / "raw"
    create_line_images(
        text_lines=lines,
        output_dir=output_path,
        fonts=[ubuntu_sans_font_info],
        size_range=range(40, 100),
        color_pairs=[((0, 0, 0), (255, 255, 255))],
        top_margins=[10],
        bottom_margins=[20],
        left_margins=[30],
        right_margins=[40],
        rng=random.Random(42),
    )
    return output_path


def test_datafiles_are_copied(example_line_images_directory: Path):
    """When distorting the images, the original images should be copied to the output directory as well."""
    output_path = example_line_images_directory.parent / "distorted_output"
    distort_line_images(
        example_line_images_directory, output_path, random.Random(42), create_scanned_book_pipeline
    )

    num_copied = len(list(output_path.glob("line_images/*.png")))
    num_original = len(list(example_line_images_directory.glob("line_images/*.png")))
    assert len(list(output_path.glob("line_images/*.png"))) == len(
        list(example_line_images_directory.glob("line_images/*.png"))
    )

    assert all(
        (
            filecmp.cmp(original, output_path / "line_images" / original.name)
            for original in example_line_images_directory.glob("line_images/*.png")
        )
    )


def test_same_output_directory_as_input_directory_is_allowed(example_line_images_directory: Path):
    """It should be possible to use the same directory for input and output."""

    output_path = example_line_images_directory
    distort_line_images(
        example_line_images_directory, output_path, random.Random(42), create_scanned_book_pipeline
    )


def test_distorted_datafiles_are_created(example_line_images_directory: Path):
    """When distorting the images, the distorted images should be created in the output directory."""
    output_path = example_line_images_directory.parent / "distorted_output"
    distort_line_images(
        example_line_images_directory, output_path, random.Random(42), create_scanned_book_pipeline
    )

    num_distorted = len(list(output_path.glob("distorted_images/*.png")))
    num_original = len(list(example_line_images_directory.glob("images/*.png")))
    assert num_distorted == num_original


def test_metadata_is_created(example_line_images_directory: Path):
    """The metadata file should be created and contain the correct columns and correct amount of rows."""
    output_path = example_line_images_directory.parent / "distorted_output"
    distort_line_images(
        example_line_images_directory, output_path, random.Random(42), create_scanned_book_pipeline
    )

    assert (output_path / "metadata.csv").exists()

    distorted_metadata = pd.read_csv(output_path / "metadata.csv")
    original_metadata = pd.read_csv(example_line_images_directory / "metadata.csv")

    assert len(distorted_metadata) == len(original_metadata)

    columns = distorted_metadata.columns.tolist()
    expected_columns = [
        "unique_id",
        "text",
        "raw_text",
        "font_path",
        "font_size",
        "text_line_id",
        "text_transform",
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
        "distorted_bbox_left",
        "distorted_bbox_top",
        "distorted_bbox_right",
        "distorted_bbox_bottom",
        "file_name",
        "augraphy_log_path",
    ]
    assert set(columns) == set(expected_columns)
    assert len(columns) == len(expected_columns)
    for column in expected_columns:
        assert column in columns


def test_metadata_has_functioning_paths(example_line_images_directory: Path):
    """The metadata file should contain functioning paths to the original and distorted images and the log files."""

    output_path = example_line_images_directory.parent / "distorted_output"
    distort_line_images(
        example_line_images_directory, output_path, random.Random(0), create_scanned_book_pipeline
    )

    distorted_metadata = pd.read_csv(output_path / "metadata.csv")
    for idx, row in distorted_metadata.iterrows():
        assert (output_path / row["undistorted_file_name"]).exists()
        assert (output_path / row["file_name"]).exists()
        assert (output_path / row["augraphy_log_path"]).exists()


def test_metadata_keeps_original_data(example_line_images_directory: Path):
    """The metadata file should keep the original data from the input metadata file."""
    output_path = example_line_images_directory.parent / "distorted_output"
    distort_line_images(
        example_line_images_directory, output_path, random.Random(42), create_scanned_book_pipeline
    )

    assert (output_path / "metadata.csv").exists()

    distorted_metadata = pd.read_csv(output_path / "metadata.csv")
    original_metadata = pd.read_csv(example_line_images_directory / "metadata.csv")
    assert set(distorted_metadata.columns).issuperset(set(original_metadata.columns))
    pd.testing.assert_frame_equal(
        distorted_metadata[original_metadata.columns], original_metadata, check_like=True
    )
