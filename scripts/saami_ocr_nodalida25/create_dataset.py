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
from collections.abc import Generator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
from lxml import etree
from tqdm import tqdm

from text_generation.augraphy_utils import create_scanned_book_pipeline
from text_generation.color import get_dark_mode_color_pair, get_light_mode_color_pair
from text_generation.fonts import FontInfo
from text_generation.image_creation import create_line_images
from text_generation.image_processing import distort_line_images
from text_generation.text_processing import (
    TargetLengthDecider,
    add_transformed_text_lines,
    chunkify_words,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--partition", type=int, default=0)
parser.add_argument("--num_partitions", type=int, default=1)
args = parser.parse_args()


@dataclass
class DatasetFileInfo:
    name: str
    path: Path
    language: str
    metadata: dict[str, str]
    num_partitions: int = 1
    partition_id: int = 0

    def load_segments(self) -> list[str]:
        return self.path.read_text().splitlines()[self.partition_id :: self.num_partitions]

    def join_segments(self, segments: list[str]) -> str:
        return " ".join(segments)

    def iter_splits(
        self, test_fraction: float, val_fraction: float, rng: random.Random
    ) -> Generator[tuple[Literal["train", "test", "val"], list[str]], None, None]:
        segments = self.load_segments()
        rng.shuffle(segments)
        n_total = len(segments)
        n_test = int(n_total * test_fraction)
        n_val = int(n_total * val_fraction)

        yield "test", self.join_segments(segments[:n_test]).split()
        yield "val", self.join_segments(segments[n_test : n_test + n_val]).split()
        yield "train", self.join_segments(segments[n_test + n_val :]).split()


class SikorFileInfo(DatasetFileInfo):
    def handle_punctuation(self, segment: str) -> str:
        """Strip space before punctuation"""
        punctuation = ".,:;!?"
        return re.sub(f" ([{punctuation}])", r"\g<1>", segment)

    def join_segments(self, segments: list[str]) -> str:
        return self.handle_punctuation(super().join_segments(segments))

    def load_segments(self) -> list[str]:
        """Load the sentences from the SIKOR XML file as segments"""
        tree = etree.parse(self.path)
        root = tree.getroot()

        segments = [" ".join(sentence.xpath(".//w/@word")) for sentence in root.xpath("//sentence")]
        return segments[self.num_partitions :: self.partition_id]


class LineDirectoryDatasetInfo(DatasetFileInfo):
    def iter_files(self) -> Generator[list[str], None, None]:
        pattern = "converted/**/*.lines.txt"
        files = sorted(self.path.glob(pattern))
        for file in tqdm(files, desc=f"Loading text lines"):
            yield file.read_text().splitlines()

    def load_segments(self) -> list[str]:
        return list(self.iter_files())

    def join_segments(self, segments: list[list[str]]) -> str:
        return " ".join(itertools.chain.from_iterable(segments))


@dataclass
class DatasetInputInfo:
    name: str
    text_segments: list[str]
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

    def get_huggingface_path(self, output_path: Path) -> Path:
        return output_path / "huggingface" / self.metadata["file"]


def setup_dataset(dataset_info: DatasetInputInfo, output_path: Path, rng: random.Random) -> Path:
    logger.info("Creating word chunks")
    # These values are chosen because when we run the chunk-algorithm on the training data, then the histogram
    # of the chunk lengths is quite similar to the histogram of the line lengths in the training data. A main
    # difference is that the training data contain a separate peak of many very short lines. Since this chunk
    # algorithm is unimodal, it will not add that extra peak, but instead have a similar peak as the training
    # data at approximately 40 characters.
    text_lines = chunkify_words(
        dataset_info.text_segments, 5, 100, TargetLengthDecider(75, 40, 5, 100, random.Random())
    )
    logger.info("Created word chunks")

    all_text_lines = add_transformed_text_lines(tqdm(list(text_lines)), str.upper)

    logger.info("Creating line images")
    create_line_images(
        all_text_lines,
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


dataset_files = [
    DatasetFileInfo(
        name=f"lipsum{i}",
        path=Path(__file__).parent.parent.parent
        / f"input/lipsum_example_data/lipsum{i}/lipsum{i}.txt",
        metadata={"corpus_source": "something"},
        language=f"l{i}",
    )
    for i in (1,)  # 2)
]

dataset_files = [
    SikorFileInfo(
        name="SIKOR_free_sme_20151010",
        path=Path(__file__).parent.parent.parent
        / "input/SIKOR/SIKOR_sme_20151010/SIKOR_free_sme_20151010.xml",
        metadata={"corpus_source": ""},
        language="sme",
        num_partitions=args.num_partitions,
        partition_id=args.partition,
    )
]

dataset_files = [
    LineDirectoryDatasetInfo(
        name="corpus_smj",
        path=Path(__file__).parent.parent.parent / "input/saami_ocr_nodalida25/lines/corpus-smj",
        metadata={"corpus_source": "something"},
        language="sme",
        num_partitions=args.num_partitions,
        partition_id=args.partition,
    )
]

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

raw_data = [
    DatasetInputInfo(
        name=f"{file_info.name}_{split}",
        text_segments=segments,
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
    )
    for file_info in dataset_files
    for split, segments in file_info.iter_splits(test_fraction=0.2, val_fraction=0.1, rng=rng)
]


# Force all threads to run on the same CPU (doesn't matter much for speed since it barely runs on multiple cores)
subprocess.run(["taskset", "-pc", str(args.partition), str(os.getpid())], check=True)
setup_and_combine_datasets(raw_data, Path(f"output_{args.partition}_test"), rng)
