"""Odour families: 138 descriptors folded into eight groups a person can hold in mind.

Eight and not more, for a reason outside chemistry: the odour map shows families as
highlighted groups, and no more than eight categories can share a chart before
colour stops telling them apart. Descriptors that fit no family ("aldehydic",
"solvent", "waxy") and "odorless" go to OTHER, which is never highlighted.

A molecule's family is decided from what experts recorded for it (see
:func:`dominant_family`), not from the model's prediction. The map places molecules
by the model's view of their smell and colours them by the experts' -- so where the
two disagree, the map shows it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from enum import StrEnum


class Family(StrEnum):
    FRUITY = "fruity"
    CITRUS = "citrus"
    FLORAL = "floral"
    GREEN = "green"
    WOODY = "woody"
    GOURMAND = "gourmand"
    SPICY = "spicy"
    SAVORY = "savory"
    OTHER = "other"


FAMILY_NAMES_RU: dict[Family, str] = {
    Family.FRUITY: "фруктовые",
    Family.CITRUS: "цитрусовые",
    Family.FLORAL: "цветочные",
    Family.GREEN: "зелёные и свежие",
    Family.WOODY: "древесные и амбровые",
    Family.GOURMAND: "сладкие и гурманские",
    Family.SPICY: "пряные и дымные",
    Family.SAVORY: "пикантные и землистые",
    Family.OTHER: "прочие",
}

_MEMBERS: dict[Family, tuple[str, ...]] = {
    Family.FRUITY: (
        "fruity", "apple", "apricot", "banana", "berry", "black currant", "cherry",
        "grape", "juicy", "melon", "peach", "pear", "pineapple", "plum", "raspberry",
        "strawberry", "tropical", "ripe", "fruit skin",
    ),
    Family.CITRUS: ("citrus", "lemon", "orange", "grapefruit", "bergamot"),
    Family.FLORAL: (
        "floral", "rose", "jasmin", "lily", "muguet", "hyacinth", "geranium", "violet",
        "orris", "orangeflower", "hawthorn", "lavender", "chamomile",
    ),
    Family.GREEN: (
        "green", "grassy", "leafy", "herbal", "hay", "tea", "weedy", "cucumber",
        "aromatic", "terpenic", "fresh", "clean", "mint", "cooling", "camphoreous",
        "ozone",
    ),
    Family.WOODY: (
        "woody", "cedar", "sandalwood", "vetiver", "pine", "dry", "cortex", "amber",
        "musk", "animal", "leathery", "powdery", "tobacco",
    ),
    Family.GOURMAND: (
        "sweet", "vanilla", "caramellic", "honey", "chocolate", "cocoa", "coffee",
        "buttery", "creamy", "milky", "dairy", "coumarinic", "lactonic", "almond",
        "hazelnut", "nutty", "coconut", "malty", "popcorn",
    ),
    Family.SPICY: (
        "spicy", "cinnamon", "clove", "anisic", "warm", "balsamic", "smoky", "burnt",
        "roasted", "phenolic", "medicinal",
    ),
    Family.SAVORY: (
        "sulfurous", "alliaceous", "garlic", "onion", "cabbage", "meaty", "beefy",
        "savory", "fishy", "cheesy", "cooked", "potato", "radish", "vegetable",
        "tomato", "celery", "gassy", "sweaty", "earthy", "musty", "mushroom", "metallic",
    ),
    Family.OTHER: (
        "alcoholic", "aldehydic", "ketonic", "solvent", "ethereal", "pungent", "sharp",
        "sour", "bitter", "fermented", "winey", "brandy", "rummy", "cognac", "waxy",
        "oily", "fatty", "soapy", "natural", "odorless",
    ),
}  # fmt: skip

FAMILY_OF: dict[str, Family] = {d: family for family, ds in _MEMBERS.items() for d in ds}


def dominant_family(descriptors: Iterable[str], frequency: Mapping[str, float]) -> Family:
    """The family a set of descriptors leans to, weighting rare descriptors higher.

    Plain counting would let "sweet" (on 30% of molecules) outvote "vanilla" (on 2%)
    for vanillin's neighbours. Each descriptor instead votes with its information
    content, ``-log(frequency)``: the rare, specific word decides. OTHER wins only
    when nothing else is present.
    """
    scores: dict[Family, float] = {}
    for descriptor in descriptors:
        family = FAMILY_OF.get(descriptor, Family.OTHER)
        weight = -math.log(max(frequency.get(descriptor, 1.0), 1e-6))
        scores[family] = scores.get(family, 0.0) + weight

    named = {f: s for f, s in scores.items() if f is not Family.OTHER}
    if not named:
        return Family.OTHER
    return max(named, key=lambda f: (named[f], -list(Family).index(f)))
