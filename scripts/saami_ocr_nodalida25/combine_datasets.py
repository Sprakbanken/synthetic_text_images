import argparse
import jinja2
import pandas as pd
from pathlib import Path
import logging
import shutil

from synthetic_ocr_data.utils import GitInfo, Language

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)

args = parser.parse_args()
dataset_metadata_files = sorted(args.input.glob("*/metadata.csv"))
dataset_directories = [metadata_file.parent for metadata_file in dataset_metadata_files]
logging.info("Found %d datasets", len(dataset_directories))

output_directory = args.output
if output_directory.exists():
    raise ValueError(f"Output directory {output_directory} already exists.")

metadata = pd.concat([
    pd.read_csv(metadata_file)
    for metadata_file in dataset_metadata_files
])
logger.info("Found metadata for %d images", len(metadata))
print((metadata).shape)

train_lines = metadata.query("file_name.str.startswith('train')").shape[0]
val_lines = metadata.query("file_name.str.startswith('val')").shape[0]
test_lines = metadata.query("file_name.str.startswith('test')").shape[0]
train_perc = round(100 * train_lines / (train_lines + val_lines + test_lines))
val_perc = round(100 * val_lines / (train_lines + val_lines + test_lines))
test_perc = round(100 * test_lines / (train_lines + val_lines + test_lines))

logger.info("Train: %d (%d%%), Val: %d (%d%%), Test: %d (%d%%)", train_lines, train_perc, val_lines, val_perc, test_lines, test_perc)

output_directory.mkdir(parents=True)
metadata.drop(columns="Unnamed: 0").to_csv(output_directory / "metadata.csv", index=False)

for dataset_directory in dataset_directories:
    for subdirectory in dataset_directory.glob("*/"):
        logger.info("Copying %s to %s", subdirectory, output_directory / subdirectory.name)
        shutil.copytree(subdirectory, output_directory / subdirectory.name, dirs_exist_ok=True)

found_test_lines = len(list((output_directory / "test").iterdir()))
if found_test_lines != test_lines:
    raise ValueError(f"Expected {test_lines} files in test directory, got {found_test_lines}")

found_train_lines = len(list((output_directory / "train").iterdir()))
if found_train_lines != train_lines:
    raise ValueError(f"Expected {train_lines} files in train directory, got {found_train_lines}")

found_val_lines = len(list((output_directory / "val").iterdir()))
if found_val_lines != val_lines:
    raise ValueError(f"Expected {val_lines} files in val directory, got {found_val_lines}")


num_lines = metadata.shape[0]
train_lines = metadata.query("file_name.str.startswith('train')").shape[0]
val_lines = metadata.query("file_name.str.startswith('val')").shape[0]
test_lines = metadata.query("file_name.str.startswith('test')").shape[0]
train_perc = round(100 * train_lines / num_lines)
val_perc = round(100 * val_lines / num_lines)
test_perc = round(100 * test_lines / num_lines)

language_code = args.input.name
git_info = GitInfo.from_language_code(language_code)

readme = (
    jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))
    .get_template("dataset_readme.md.j2")
    .render(
        language=git_info.language._value_,
        corpus_hash=git_info.submodule_commit,
        train_perc=train_perc,
        train_lines=train_lines,
        val_perc=val_perc,
        val_lines=val_lines,
        test_perc=test_perc,
        test_lines=test_lines,
        repo_hash=git_info.commit,
        language_code=language_code,
    )
)

(output_directory / "README.md").write_text(readme)