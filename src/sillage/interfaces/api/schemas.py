"""The public JSON contract, shared by the HTTP API and ``sillage-predict --json``.

Domain objects are not serialised directly. A response schema is a promise to
callers, and the domain must stay free to change shape without breaking it -- and
it must not be possible to leak a field by adding it to a dataclass. Notably,
:attr:`DescriptorScore.score` never appears here: an uncalibrated score is a
ranking position, and the only number a caller receives is ``probability``.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field

from sillage.application.predict_odor import DEFAULT_NEIGHBOURS, MAX_INPUT_LENGTH
from sillage.domain import (
    FAMILY_NAMES_RU,
    FAMILY_OF,
    DescriptorScore,
    Family,
    MapPoint,
    ModelDescription,
    MoleculeNames,
    NameSuggestion,
    OdorReport,
    Refusal,
)

MAX_NEIGHBOURS = 20


class PredictRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        validation_alias=AliasChoices("query", "smiles"),
        examples=["ванилин", "hedione", "COc1cc(C=O)ccc1O"],
        description=(
            "A molecule name (Russian or English) or a SMILES string. Stereochemistry is "
            "accepted and dropped. `smiles` is accepted as the field name too."
        ),
    )
    neighbours: int = Field(default=DEFAULT_NEIGHBOURS, ge=0, le=MAX_NEIGHBOURS)


class MoleculeOut(BaseModel):
    smiles: str = Field(description="Canonical SMILES without stereochemistry.")
    input_smiles: str
    formula: str
    weight: float = Field(description="Average molecular weight, Da.")
    stereo_discarded: bool


class DescriptorOut(BaseModel):
    descriptor: str
    confidence: str = Field(
        description="likely, probable, possible, unlikely, negligible, unverified"
    )
    probability: float | None = Field(
        description="Calibrated probability; null for descriptors too rare to calibrate."
    )
    base_rate: float = Field(description="Share of training molecules with this descriptor.")


class KnownOut(BaseModel):
    descriptors: list[str]
    seen_in_training: bool = Field(
        description="True when the model was fitted on this molecule: agreement is recall."
    )


class NamesOut(BaseModel):
    display: str | None = Field(description="Russian name when known, English otherwise.")
    ru: str | None
    en: str | None
    cid: int | None = Field(description="PubChem compound id.")
    pubchem_url: str | None


class PositionOut(BaseModel):
    x: float
    y: float


class FamilyOut(BaseModel):
    id: str
    name: str = Field(description="Russian name of the odour family.")
    count: int


class MapPointsOut(BaseModel):
    """Columns rather than one object per point: 4777 points, a third of the bytes."""

    x: list[float]
    y: list[float]
    family: list[int] = Field(description="Index into `families`.")
    smiles: list[str]
    name: list[str | None]


class MapOut(BaseModel):
    families: list[FamilyOut]
    descriptor_family: dict[str, int] = Field(
        description="Odour descriptor -> index into `families`."
    )
    points: MapPointsOut


class NeighbourOut(BaseModel):
    smiles: str
    similarity: float = Field(description="Tanimoto on ECFP4. Orientation, not confidence.")
    descriptors: list[str]
    names: NamesOut | None = None


class SuggestionOut(BaseModel):
    name: str = Field(description="The name to show.")
    matched: str = Field(description="The name that matched the typed text.")
    smiles: str


class ReportOut(BaseModel):
    refused: bool = False
    model: str
    query_kind: str = Field(description="name or smiles: what the input was read as.")
    matched_name: str | None = Field(
        default=None, description="The name the input matched, as the directory spells it."
    )
    names: NamesOut | None
    molecule: MoleculeOut
    predicted: list[DescriptorOut] = Field(
        description="Calibrated descriptors at 'possible' or above, most probable first."
    )
    unverified: list[DescriptorOut] = Field(
        description="Uncalibrated descriptors the model leans towards. No probabilities."
    )
    known: KnownOut | None
    neighbours: list[NeighbourOut]
    caveats: list[str]
    map_position: PositionOut | None = Field(
        default=None, description="Where the molecule lands on the odour map (GET /map)."
    )


class RefusalOut(BaseModel):
    refused: bool = True
    reason: str = Field(description="Stable code: invalid_input, unseen_element, ...")
    detail: str
    molecule: MoleculeOut | None = None


class ModelOut(BaseModel):
    name: str
    scope: str
    trained_on: int
    descriptors: list[str]
    unverified: list[str]
    metrics: dict[str, float]
    floor: dict[str, float]


def report_out(report: OdorReport) -> ReportOut:
    return ReportOut(
        model=report.model_name,
        query_kind=report.query_kind,
        matched_name=report.matched_name,
        names=names_out(report.names),
        molecule=MoleculeOut.model_validate(report.molecule, from_attributes=True),
        predicted=[_descriptor(s) for s in report.profile.confident()],
        unverified=[_descriptor(s) for s in report.profile.unverified()],
        known=(
            None
            if report.known is None
            else KnownOut(
                descriptors=list(report.known.descriptors),
                seen_in_training=report.known.seen_in_training,
            )
        ),
        neighbours=[
            NeighbourOut(
                smiles=n.smiles,
                similarity=n.similarity,
                descriptors=list(n.descriptors),
                names=names_out(n.names),
            )
            for n in report.neighbours
        ],
        caveats=[str(c) for c in report.caveats],
        map_position=(
            None
            if report.map_position is None
            else PositionOut(x=round(report.map_position.x, 4), y=round(report.map_position.y, 4))
        ),
    )


def refusal_out(refusal: Refusal) -> RefusalOut:
    return RefusalOut(
        reason=str(refusal.reason),
        detail=refusal.detail,
        molecule=(
            None
            if refusal.molecule is None
            else MoleculeOut.model_validate(refusal.molecule, from_attributes=True)
        ),
    )


def result_out(result: OdorReport | Refusal) -> ReportOut | RefusalOut:
    return refusal_out(result) if isinstance(result, Refusal) else report_out(result)


def model_out(description: ModelDescription) -> ModelOut:
    return ModelOut(
        name=description.name,
        scope=description.scope,
        trained_on=description.trained_on,
        descriptors=list(description.descriptors),
        unverified=list(description.unverified),
        metrics=description.metrics,
        floor=description.floor,
    )


def map_out(points: tuple[MapPoint, ...], name_of) -> MapOut:
    """The whole map in columns. *name_of* gives a display name for a SMILES, or None."""
    order = list(Family)
    return MapOut(
        families=[
            FamilyOut(
                id=f.value,
                name=FAMILY_NAMES_RU[f],
                count=sum(1 for p in points if p.family is f),
            )
            for f in order
        ],
        descriptor_family={d: order.index(f) for d, f in sorted(FAMILY_OF.items())},
        points=MapPointsOut(
            x=[round(p.position.x, 4) for p in points],
            y=[round(p.position.y, 4) for p in points],
            family=[order.index(p.family) for p in points],
            smiles=[p.smiles for p in points],
            name=[name_of(p.smiles) for p in points],
        ),
    )


def names_out(names: MoleculeNames | None) -> NamesOut | None:
    if names is None:
        return None
    return NamesOut(
        display=names.display,
        ru=names.ru,
        en=names.en,
        cid=names.cid,
        pubchem_url=(
            f"https://pubchem.ncbi.nlm.nih.gov/compound/{names.cid}" if names.cid else None
        ),
    )


def suggestion_out(suggestion: NameSuggestion) -> SuggestionOut:
    return SuggestionOut(
        name=suggestion.display, matched=suggestion.matched, smiles=suggestion.smiles
    )


def _descriptor(score: DescriptorScore) -> DescriptorOut:
    return DescriptorOut(
        descriptor=score.name,
        confidence=str(score.confidence),
        probability=None if score.probability is None else round(score.probability, 3),
        base_rate=round(score.base_rate, 4),
    )
