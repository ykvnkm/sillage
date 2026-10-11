"""Every odour descriptor the model can output has a Russian name on the page.

The API speaks the source data's English vocabulary; the web interface is Russian.
A descriptor added to the data without a translation would surface as a lone
English word in a Russian sentence, so the build fails instead.
"""

from __future__ import annotations

import re
from pathlib import Path

from sillage.data.schemas import OPENPOM_DESCRIPTORS

DESCRIPTORS_JS = (
    Path(__file__).resolve().parents[1] / "src/sillage/interfaces/web/static/descriptors.js"
)


def translations() -> dict[str, str]:
    return dict(re.findall(r'^\s*"([^"]+)": "([^"]+)",$', DESCRIPTORS_JS.read_text("utf-8"), re.M))


def test_every_descriptor_has_a_russian_name() -> None:
    missing = sorted(set(OPENPOM_DESCRIPTORS) - set(translations()))

    assert missing == []


def test_no_translation_for_a_descriptor_that_does_not_exist() -> None:
    assert sorted(set(translations()) - set(OPENPOM_DESCRIPTORS)) == []


def test_translations_are_cyrillic_and_distinct() -> None:
    names = translations()

    assert all(re.search("[а-яё]", name) for name in names.values())
    duplicates = {n for n in names.values() if list(names.values()).count(n) > 1}
    assert duplicates == set()
