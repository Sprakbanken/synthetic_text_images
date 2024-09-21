from functools import partial
from pathlib import Path

import pandas as pd
import urllib3
import urllib3.util

from text_generation.fonts import check_font_support

# Define paths
font_directory = Path(__file__).parent.parent.parent / "fonts/saami_ocr_nodalida25"
character_information_path = Path(__file__).parent.parent.parent / "data/samiske_bokstaver.csv"

# Read character information
character_information = pd.read_csv(character_information_path)
saami_letters = "".join(character_information["bokstav"].to_list())

# Evaluate font support
eval_info = []
for font_subdir in font_directory.iterdir():
    for font_path in font_subdir.glob("**/*.ttf"):
        eval_info.append(
            {
                "font_path": font_path,
                "font_name": font_path.stem,
                "font_family": font_subdir.stem,
                "supports_all_characters": check_font_support(saami_letters, font_path),
            }
        )

# Create DataFrame and print results
eval_info_df = pd.DataFrame(eval_info)

# Read font information
font_information = pd.read_csv(font_directory / "full_font_info.csv")
if "supports_all_characters" in font_information.columns:
    font_information = font_information.drop(columns=["supports_all_characters"])


# Define font directory names mapping
def format_directory_name(directory_name: str, directory_name_map: dict[str, str]) -> str:
    """Format directory name. Use mapping if available."""
    dirname = directory_name_map.get(directory_name, str(directory_name).replace(" ", "_"))
    assert Path(font_directory / dirname).exists(), dirname
    return dirname


def change_url(url_string: str, preview_text: str) -> str:
    """Update the URL to include preview text."""
    url = urllib3.util.parse_url(url_string)
    url = urllib3.util.url.Url(
        scheme=url.scheme,
        auth=url.auth,
        host=url.host,
        port=url.port,
        path=url.path,
        query=f"preview.text={preview_text}",
        fragment=url.fragment,
    )
    return str(url)


# Clean up font information
font_information = font_information.dropna(subset=["name"])
font_information["directory"] = font_information["name"].map(
    partial(
        format_directory_name,
        directory_name_map={"Playwrite Cuba": "Playwrite_CU", "Noto Mono": "Noto_Sans_Mono"},
    ),
)
font_information["source"] = font_information["source"].map(
    partial(change_url, preview_text=saami_letters)
)

# Check font support and add to font information
(
    font_information.join(
        eval_info_df.groupby("font_family")["supports_all_characters"].min(), on="directory"
    )
    .sort_values("supports_all_characters", ascending=False)
    .to_csv(font_directory / "full_font_info.csv", index=False)
)