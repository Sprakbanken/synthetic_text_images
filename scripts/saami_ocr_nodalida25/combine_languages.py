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
language_names = ["sma", "sme", "smj", "smn"]
dataset_metadata_files = [args.input / d / "metadata.csv" for d in language_names]
dataset_directories = [args.input / d for d in language_names]

output_directory = args.output
if output_directory.exists():
    raise ValueError(f"Output directory {output_directory} already exists.")

# Add language information to dataset
metadata = pd.concat([
    pd.read_csv(metadata_file).assign(language_code=metadata_file.parent.name)
    for metadata_file in dataset_metadata_files
])
assert set(metadata["language_code"]) == {"sma", "sme", "smj", "smn"}

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
metadata.drop(columns="Unnamed: 0", errors="ignore").to_csv(output_directory / "metadata.csv", index=False)


def get_line_count(language_code, split):
    return metadata.query(f"language_code == '{language_code}' and file_name.str.startswith('{split}')").shape[0]

for language in language_names:
    for split in ["train", "val", "test"]:
        logger.info("Found %d lines for the %s split of %s", get_line_count(language, split), split, language)
        

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

git_hashes = {
    language_code: GitInfo.from_language_code(language_code).submodule_commit
    for language_code in ["sma", "sme", "smj", "smn"]
}
commit = GitInfo.from_language_code("sma").commit
    
readme = (
    jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))
    .get_template("./templates/multilanguage_dataset_readme.md.j2")
    .render(
        train_perc=train_perc,
        train_lines=train_lines,
        val_perc=val_perc,
        val_lines=val_lines,
        test_perc=test_perc,
        test_lines=test_lines,
        sma_hash=git_hashes["sma"],
        sme_hash=git_hashes["sme"],
        smj_hash=git_hashes["smj"],
        smn_hash=git_hashes["smn"],
        sma_train_lines=get_line_count("sma", "train"),
        sma_val_lines=get_line_count("sma", "val"),
        sma_test_lines=get_line_count("sma", "test"),
        sme_train_lines=get_line_count("sme", "train"),
        sme_val_lines=get_line_count("sme", "val"),
        sme_test_lines=get_line_count("sme", "test"),
        smj_train_lines=get_line_count("smj", "train"),
        smj_val_lines=get_line_count("smj", "val"),
        smj_test_lines=get_line_count("smj", "test"),
        smn_train_lines=get_line_count("smn", "train"),
        smn_val_lines=get_line_count("smn", "val"),
        smn_test_lines=get_line_count("smn", "test"),
        repo_hash=commit,
    )
)

(output_directory / "README.md").write_text(readme)