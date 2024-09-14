import random
from pathlib import Path

from PIL import ImageFont
import itertools

from text_generation.create import create_line_images
from text_generation.color import get_dark_mode_color_pair, get_light_mode_color_pair
from text_generation.image import distort_line_images
from text_generation.pipeline import create_scanned_book_pipeline
from text_generation.text_processing import (
    TargetLengthDecider,
    add_transformed_text_lines,
    chunkify_words,
)

if __name__ == "__main__":
    text_file_path = Path(__file__).parent / "example_text.txt"
    font_path = "fonts/Ubuntu_Sans/UbuntuSans-VariableFont_wdth,wght.ttf"
    font_paths = [Path(font_path)]
    text_lines = text_file_path.read_text(encoding="utf-8").splitlines()
    output_dir = Path(__file__).parent / "distorted_output"
    output_dir.mkdir(exist_ok=True)

    font_sizes = [32, 64, 128, 256]
    rng = random.Random(42)
    light_mode_color_pairs = [get_light_mode_color_pair(rng) for _ in range(10)]
    dark_mode_color_pairs = [get_dark_mode_color_pair(rng) for _ in range(10)]
    color_pairs = light_mode_color_pairs + dark_mode_color_pairs

    top_margins = [20]
    bottom_margins = [30]
    left_margins = [30]
    right_margins = [20]

    fonts = [ImageFont.truetype(font_path, rng.choice(font_sizes)) for font_path in font_paths]

    words = itertools.chain.from_iterable((sentence.split() for sentence in text_lines))
    target_length = 35
    min_length = 5
    max_length = 100
    chunks = chunkify_words(words, min_length, max_length, TargetLengthDecider(35, 10, 5, 100, rng))
    transformed_chunks = add_transformed_text_lines(chunks)

    create_line_images(
        text_lines=transformed_chunks,
        output_dir=output_dir,
        fonts=fonts,
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    # TODO: maybe the generator for text should also provide a "textline_id" to keep track of multiple versions based on the same text?

    distort_line_images(
        output_dir,
        output_dir,
        random.Random(42),
        create_scanned_book_pipeline,
        distorted_subdir_name="train",
    )
