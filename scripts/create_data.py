import random
from pathlib import Path

from text_generation import create_datafiles, create_random_dark_color, create_random_light_color

if __name__ == "__main__":
    text_file_path = Path(__file__).parent / "example_text.txt"
    font_path = "fonts/Ubuntu_Sans/UbuntuSans-VariableFont_wdth,wght.ttf"
    font_paths = [Path(font_path)]
    text_lines = text_file_path.read_text(encoding="utf-8").splitlines()
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    font_sizes = [16, 40]
    rng = random.Random(42)
    # light mode has dark text (fist in the tuple) and light background (second in the tuple)
    light_mode_color_pairs = [
        (create_random_dark_color(rng), create_random_light_color(rng)) for _ in range(10)
    ]
    # dark mode has light text (fist in the tuple) and dark background (second in the tuple)
    dark_mode_color_pairs = [
        (create_random_light_color(rng), create_random_dark_color(rng)) for _ in range(10)
    ]
    color_pairs = light_mode_color_pairs + dark_mode_color_pairs

    top_margins = [10, 20]
    bottom_margins = [10, 20]
    left_margins = [10, 20]
    right_margins = [10, 20]

    rng = random.Random(42)
    create_datafiles(
        text_lines=text_lines,
        output_dir=output_dir,
        font_paths=font_paths,
        font_sizes=font_sizes,
        color_pairs=color_pairs,
        top_margins=top_margins,
        bottom_margins=bottom_margins,
        left_margins=left_margins,
        right_margins=right_margins,
        rng=rng,
    )
    # TODO: maybe the generator for text should also provide a "textline_id" to keep track of multiple versions based on the same text?
