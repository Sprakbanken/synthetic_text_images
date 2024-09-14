import random
import shutil
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
from PIL import ImageFont

from text_generation.create import create_line_images
from text_generation.image import distort_line_images
from text_generation.pipeline import create_scanned_book_pipeline
from text_generation.text_processing import (
    TargetLengthDecider,
    add_transformed_text_lines,
    chunkify_words,
)


@dataclass
class FontInfo:
    path: Path
    name: str
    family: str

    def load_font(self, size: int):
        return ImageFont.truetype(self.path, size)


@dataclass
class DatasetFileInfo:
    name: str
    path: Path
    language: str
    metadata: dict[str, str]

    def load_text_lines(self) -> list[str]:
        return self.path.read_text().splitlines()

    def iter_splits(
        self, test_fraction: float, val_fraction: float, rng: random.Random
    ) -> Generator[tuple[Literal["train", "test", "val"], list[str]], None, None]:
        raw_text_lines = self.load_text_lines()
        rng.shuffle(raw_text_lines)
        n_total = len(raw_text_lines)
        n_test = int(n_total * test_fraction)
        n_val = int(n_total * val_fraction)

        yield "test", raw_text_lines[:n_test]
        yield "val", raw_text_lines[n_test : n_test + n_val]
        yield "train", raw_text_lines[n_test + n_val :]


@dataclass
class DatasetInputInfo:
    name: str
    lines: list[str]
    metadata: dict[str, str]
    split: Literal["train", "test", "val"]
    language: str
    font_list: list[FontInfo]
    color_pairs: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]]
    top_margins: int
    bottom_margins: int
    left_margins: int
    right_margins: int

    def get_huggingface_path(self, output_path: Path) -> Path:
        return output_path / "huggingface" / self.metadata["file"]


def setup_dataset(dataset_info: DatasetInputInfo, output_path: Path, rng: random.Random) -> Path:
    text_lines = chunkify_words(
        dataset_info.lines, 5, 100, TargetLengthDecider(35, 10, 5, 100, rng)
    )

    all_text_lines = add_transformed_text_lines(text_lines, str.upper)
    create_line_images(
        all_text_lines,
        output_path,
        fonts=dataset_info.font_list,
        color_pairs=dataset_info.color_pairs,
        top_margins=dataset_info.top_margins,
        bottom_margins=dataset_info.bottom_margins,
        left_margins=dataset_info.left_margins,
        right_margins=dataset_info.right_margins,
        rng=rng,
    )
    distort_line_images(
        line_images_dir=output_path,
        output_dir=output_path,
        rng=rng,
        pipeline_creator=create_scanned_book_pipeline,
        distorted_subdir_name=dataset_info.split,
    )

    return output_path


def combine_datasets(dataset_dirs: list[Path], output_path: Path):
    """Combine multiple datasets into one."""
    dataset_metadata = []
    for dataset_dir in dataset_dirs:
        shutil.copytree(dataset_dir, output_path, dirs_exist_ok=True)
        dataset_metadata.append(pd.read_csv(dataset_dir / "metadata.csv"))

    combined_metadata = pd.concat(dataset_metadata).reset_index(drop=True)
    combined_metadata.to_csv(output_path / "metadata.csv")


def setup_and_combine_datasets(
    datasets_info: Iterable[DatasetInputInfo], output_path: Path, rng: random.Random
):
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        datasets = [
            setup_dataset(dataset_info, tmpdir / dataset_info.name, rng)
            for dataset_info in datasets_info
        ]

        combine_datasets(datasets, output_path)


dataset_files = [
    DatasetFileInfo(
        name=f"lipsum{i}",
        path=Path(__file__).parent.parent.parent
        / f"input/lipsum_example_data/lipsum{i}/lipsum{i}.txt",
        metadata={"corpus_source": "something"},
        language=f"l{i}",
    )
    for i in (1, 2)
]

font_lists = {
    "train": [
        FontInfo(
            "/hdd/home/mariero/synthetic_text_images/fonts/Ubuntu_Sans/static/UbuntuSans_Condensed-Bold.ttf",
            "Ubuntu Sans Condensed Bold",
            "Ubuntu Sans",
        ).load_font(64)
    ],
    "val": [
        FontInfo(
            "/hdd/home/mariero/synthetic_text_images/fonts/Ubuntu_Sans/static/UbuntuSans-Light.ttf",
            "Ubuntu Sans Light",
            "Ubuntu Sans",
        ).load_font(64)
    ],
    "test": [
        FontInfo(
            "/hdd/home/mariero/synthetic_text_images/fonts/fanwood-master/fanwood-master/webfonts/fanwood_italic-webfont.ttf",
            "Fanwood Italic",
            "Fanwood",
        ).load_font(64)
    ],
}

rng = random.Random(42)

raw_data = [
    DatasetInputInfo(
        name=f"{file_info.name}_{split}",
        lines=lines,
        metadata=file_info.metadata,
        language=file_info.language,
        split=split,
        font_list=font_lists[split],
        top_margins=[20],
        bottom_margins=[20],
        left_margins=[20],
        right_margins=[20],
        color_pairs=[((0, 0, 0), (255, 255, 255)), ((255, 255, 255), (0, 0, 0))],
    )
    for file_info in dataset_files
    for split, lines in file_info.iter_splits(test_fraction=0.2, val_fraction=0.1, rng=rng)
]

setup_and_combine_datasets(raw_data, Path("output"), rng)
