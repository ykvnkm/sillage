"""The public JSON contract, shared by the HTTP API and ``sillage-predict --json``.

Domain objects are not serialised directly. A response schema is a promise to
callers, and the domain must stay free to change shape without breaking it -- and
it must not be possible to leak a field by adding it to a dataclass. Notably,
:attr:`DescriptorScore.score` never appears here: an uncalibrated score is a
ranking position, and the only number a caller receives is ``probability``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sillage.application.predict_odor import DEFAULT_NEIGHBOURS, MAX_INPUT_LENGTH
from sillage.domain import (
    DescriptorScore,
    ModelDescription,
    OdorReport,
    Refusal,
)

MAX_NEIGHBOURS = 20


class PredictRequest(BaseModel):
    smiles: str = Field(
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        examples=["COc1cc(C=O)ccc1O"],
        description="The molecule, as a SMILES string. Stereochemistry is accepted and dropped.",
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


class NeighbourOut(BaseModel):
    smiles: str
    similarity: float = Field(description="Tanimoto on ECFP4. Orientation, not confidence.")
    descriptors: list[str]


class ReportOut(BaseModel):
    refused: bool = False
    model: str
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
            NeighbourOut(smiles=n.smiles, similarity=n.similarity, descriptors=list(n.descriptors))
            for n in report.neighbours
        ],
        caveats=[str(c) for c in report.caveats],
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


def _descriptor(score: DescriptorScore) -> DescriptorOut:
    return DescriptorOut(
        descriptor=score.name,
        confidence=str(score.confidence),
        probability=None if score.probability is None else round(score.probability, 3),
        base_rate=round(score.base_rate, 4),
    )
