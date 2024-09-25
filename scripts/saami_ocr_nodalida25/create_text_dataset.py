import subprocess
from pathlib import Path

from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent.parent
data_dir = root_dir / "input/saami_ocr_nodalida25"
raw_dir = data_dir / "raw"
out_dir = data_dir / "lines"

for xml_file in tqdm(list(raw_dir.glob("corpus-sm*/converted/**/*.xml"))):
    file_path = xml_file.relative_to(raw_dir)
    new_file = out_dir / file_path.with_suffix(".lines.txt")

    text = subprocess.run(["ccat", str(xml_file), "-a"], capture_output=True, text=True).stdout
    processed_text = "\n".join(line.rstrip("¶").strip() for line in text.splitlines())

    new_file.parent.mkdir(parents=True, exist_ok=True)
    new_file.write_text(processed_text)
