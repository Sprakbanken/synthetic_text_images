import pytest

import synthetic_ocr_data.utils


@pytest.mark.parametrize(
    "language_code, expected_language",
    [
        ("sme", synthetic_ocr_data.utils.Language.SME),
        ("sma", synthetic_ocr_data.utils.Language.SMA),
        ("smj", synthetic_ocr_data.utils.Language.SMJ),
        ("smn", synthetic_ocr_data.utils.Language.SMN),
    ],
)
def test_with_language(language_code, expected_language):
    git_info = synthetic_ocr_data.utils.GitInfo.from_language_code(language_code)
    assert git_info.language == expected_language
    assert len(git_info.commit) == 40
    assert len(git_info.submodule_commit) == 40
    assert git_info.commit != git_info.submodule_commit
    assert git_info.submodule_repo == f"https://github.com/giellalt/corpus-{language_code}"
