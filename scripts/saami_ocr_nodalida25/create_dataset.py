import itertools
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["GOTO_NUM_THREADS"] = "1"
os.environ["OPENCV_FOR_THREADS_NUM"] = "1"
os.environ["OPENCV_FOR_OPENMP_DYNAMIC_DISABLE"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import logging
import random
import re
import shutil
import subprocess
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import jinja2
import pandas as pd
from lxml import etree
from tqdm import tqdm

from synthetic_ocr_data.augraphy_utils import create_scanned_book_pipeline
from synthetic_ocr_data.color import get_dark_mode_color_pair, get_light_mode_color_pair
from synthetic_ocr_data.fonts import FontInfo
from synthetic_ocr_data.image_creation import create_line_images
from synthetic_ocr_data.image_processing import distort_line_images
from synthetic_ocr_data.text_processing import (
    TargetLengthDecider,
    add_transformed_text_lines,
    chunkify_words,
    clean,
)
from synthetic_ocr_data.utils import Language, GitInfo

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, required=True)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--partition", type=int, default=0)
parser.add_argument("--num_partitions", type=int, default=1)
args = parser.parse_args()

assert args.output_dir.exists()
language = Language[args.language.upper()]
language_code = language._name_.lower()


@dataclass
class LineDirectoryDatasetInfo:
    name: str
    path: Path
    language: str
    metadata: dict[str, str]
    num_partitions: int
    partition_id: int

    def iter_files(self) -> Generator[list[str], None, None]:
        pattern = "converted/**/*.lines.txt"
        files = sorted(self.path.glob(pattern))
        for file in tqdm(files, desc=f"Loading text lines"):
            yield file.read_text().splitlines()

    def load_files(self) -> list[str]:
        return list(self.iter_files())[self.partition_id :: self.num_partitions]

    def join_files(self, segments: list[list[str]]) -> str:
        return clean(" ".join(itertools.chain.from_iterable(segments)))

    def iter_splits(
        self, test_fraction: float, val_fraction: float, rng: random.Random
    ) -> Generator[tuple[Literal["train", "test", "val"], list[str]], None, None]:
        files = self.load_files()
        rng.shuffle(files)
        n_total = len(files)
        n_test = int(n_total * test_fraction)
        n_val = int(n_total * val_fraction)

        yield "test", self.join_files(files[:n_test]).split()
        yield "val", self.join_files(files[n_test : n_test + n_val]).split()
        yield "train", self.join_files(files[n_test + n_val :]).split()


@dataclass
class DatasetInputInfo:
    name: str
    words: list[str]
    metadata: dict[str, str]
    split: Literal["train", "test", "val"]
    language: str
    font_list: list[FontInfo]
    size_range: range
    color_pairs: Sequence[tuple[tuple[int, int, int], tuple[int, int, int]]]
    top_margins: int
    bottom_margins: int
    left_margins: int
    right_margins: int
    max_lines: int = 0
    uppercase_fraction: float = 0.1

    def get_huggingface_path(self, output_path: Path) -> Path:
        return output_path / "huggingface" / self.metadata["file"]


def setup_dataset(dataset_info: DatasetInputInfo, output_path: Path, rng: random.Random) -> Path:
    logger.info("Creating word chunks")
    # These values are chosen because when we run the chunk-algorithm on the training data, then the histogram
    # of the chunk lengths is quite similar to the histogram of the line lengths in the training data. A main
    # difference is that the training data contain a separate peak of many very short lines. Since this chunk
    # algorithm is unimodal, it will not add that extra peak, but instead have a similar peak as the training
    # data at approximately 40 characters.
    text_lines = list(chunkify_words(
        dataset_info.words, 5, 100, TargetLengthDecider(75, 40, 5, 100, rng)
    ))
    rng.shuffle(text_lines)

    num_upper = dataset_info.uppercase_fraction * dataset_info.max_lines
    # 2 * num_upper because we keep both the transformed and untransformed data
    num_untransformed = int(dataset_info.max_lines - 2 * num_upper)

    logger.info("Created word chunks")
    untransformed_text_lines = add_transformed_text_lines(text_lines[:num_untransformed])
    transformed_text_lines = add_transformed_text_lines(text_lines[num_untransformed:dataset_info.max_lines], str.upper)

    logger.info("Creating line images")
    create_line_images(
        tqdm(itertools.chain(untransformed_text_lines, transformed_text_lines), desc="Creating line images", total=dataset_info.max_lines),
        output_path,
        fonts=dataset_info.font_list,
        color_pairs=dataset_info.color_pairs,
        top_margins=dataset_info.top_margins,
        bottom_margins=dataset_info.bottom_margins,
        left_margins=dataset_info.left_margins,
        right_margins=dataset_info.right_margins,
        size_range=dataset_info.size_range,
        rng=rng,
    )
    logger.info("Created line images, creating distorted images")
    distort_line_images(
        line_images_dir=output_path,
        output_dir=output_path,
        rng=rng,
        pipeline_creator=create_scanned_book_pipeline,
        distorted_subdir_name=dataset_info.split,
    )
    logger.info("Created distorted images")

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


file_info = LineDirectoryDatasetInfo(
    name=f"corpus_{language_code}",
    path=Path(__file__).parent.parent.parent / f"input/saami_ocr_nodalida25/lines/corpus-{language_code}",
    metadata={"corpus_source": f"https://github.com/giellalt/corpus-{language_code}"},
    language=language_code,
    num_partitions=args.num_partitions,
    partition_id=args.partition,
)

font_dir = Path(__file__).parent.parent.parent / "fonts/saami_ocr_nodalida25"
font_list = pd.read_csv(font_dir / "full_font_info.csv")


font_lists = {
    "train": [
        FontInfo.from_record(font_dir, record)
        for record in font_list.query("split == 'train'").to_dict(orient="records")
    ],
    "val": [
        FontInfo.from_record(font_dir, record)
        for record in font_list.query("split == 'val'").to_dict(orient="records")
    ],
    "test": [
        FontInfo.from_record(font_dir, record)
        for record in font_list.query("split == 'test'").to_dict(orient="records")
    ],
}
rng = random.Random(42)

test_fraction = 0.2
val_fraction = 0.1
train_fraction = 1 - test_fraction - val_fraction

num_lines = 100_000
num_lines_per_split = {
    "train": int(num_lines * train_fraction / args.num_partitions),
    "test": int(num_lines * test_fraction / args.num_partitions),
    "val": int(num_lines * val_fraction / args.num_partitions),
}

transformed_fraction = 0.1


raw_data = [
    DatasetInputInfo(
        name=f"{file_info.name}_{split}",
        words=words,
        metadata=file_info.metadata,
        language=file_info.language,
        split=split,
        font_list=font_lists[split],
        size_range=range(40, 100),
        top_margins=[50],
        bottom_margins=[50],
        left_margins=[100],
        right_margins=[100],
        color_pairs=[
            ((0, 0, 0), (255, 255, 255)),
            ((255, 255, 255), (0, 0, 0)),
            *[get_dark_mode_color_pair(rng) for _ in range(10)],
            *[get_light_mode_color_pair(rng) for _ in range(20)],
        ],
        max_lines=num_lines_per_split[split],
    )
    for split, words in file_info.iter_splits(test_fraction=test_fraction, val_fraction=val_fraction, rng=rng)
]


# Force all threads to run on the same CPU (doesn't matter much for speed since it barely runs on multiple cores)
subprocess.run(["taskset", "-pc", str(args.partition), str(os.getpid())], check=True)
out_dir = args.output_dir / language._name_.lower() / f"output_{args.partition}_max_{args.num_partitions-1}"
out_dir.mkdir(parents=True, exist_ok=True)
setup_and_combine_datasets(
    raw_data,
    out_dir,
    rng
)

metadata = pd.read_csv(out_dir / "metadata.csv")
num_lines = metadata.shape[0]
train_lines = metadata.query("file_name.str.startswith('train')").shape[0]
val_lines = metadata.query("file_name.str.startswith('val')").shape[0]
test_lines = metadata.query("file_name.str.startswith('test')").shape[0]
git_info = GitInfo.from_language_code(language_code)

jinja2.Environment(
    loader=jinja2.FileSystemLoader(Path(__file__).parent)
).get_template("dataset_readme.md.j2").stream(
    language=language._value_,
    corpus_hash=git_info.submodule_commit,
    train_perc=100 * (1 - test_fraction - val_fraction),
    train_lines=train_lines,
    val_perc=100 * val_fraction,
    val_lines=val_lines,
    test_prec=100 * test_fraction,
    test_lines=test_lines,
    repo_hash=git_info.commit,
    language_code=language_code,
).dump(out_dir / "README.md")
