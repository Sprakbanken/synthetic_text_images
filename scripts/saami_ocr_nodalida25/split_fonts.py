from pathlib import Path

import pandas as pd

font_directory = Path(__file__).parent.parent.parent / "fonts/saami_ocr_nodalida25"


df = pd.read_csv(font_directory / "full_font_info.csv")

# To make sure that we don't have fonts from the same family in multiple splits,
# we make sure that all font families with more than one font are only in the training set.
family_counts = df["family"].value_counts()
train_only_families = set(family_counts[family_counts > 1].index)

selected_fonts = {"train": set(), "val": set(), "test": set()}
for i, query in enumerate(
    ["pixel", "monospace", "script", "serif", "display", "~serif", "blackletter"]
):
    test_font, val_font = (
        df.query(
            "(family not in @train_only_families) & (name not in (@train | @val | @test))",
            local_dict=selected_fonts | {"train_only_families": train_only_families},
        )
        .query(query)
        .sample(2, random_state=i, replace=False)["name"]
    )
    selected_fonts["test"].add(test_font)
    selected_fonts["val"].add(val_font)

test = df.query("name in @test", local_dict=selected_fonts).copy()
val = df.query("name in @val", local_dict=selected_fonts).copy()
train = df.query("name not in (@val | @test)", local_dict=selected_fonts).copy()

test["split"] = "test"
val["split"] = "val"
train["split"] = "train"

pd.concat([train, val, test]).to_csv(font_directory / "full_font_info.csv", index=False)
