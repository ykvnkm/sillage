"""The applicability domain must agree with RDKit and must actually refuse things."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from sillage.features import morgan_fingerprints, parse_smiles
from sillage.models.applicability import ApplicabilityDomain, ChemicalScope, tanimoto_similarity

FAMILIAR = [
    "CCO",
    "CCCO",
    "CCCCO",
    "CCCCCO",
    "c1ccccc1C",
    "c1ccccc1CC",
    "c1ccccc1CCC",
    "O=Cc1ccc(O)cc1",
    "O=Cc1ccccc1",
    "CC(=O)OCC",
]
ALIEN = "O=C([O-])[O-].[Mg+2].[OH-]"  # basic magnesium carbonate, the case from step 1.6


def fingerprints(smiles: list[str]) -> np.ndarray:
    return morgan_fingerprints(parse_smiles(smiles)).values


# --- the similarity itself ---------------------------------------------------


def test_tanimoto_matches_rdkit() -> None:
    """Our matrix version must be RDKit's number, not merely close to it."""
    molecules = parse_smiles(FAMILIAR)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    native = [generator.GetFingerprint(molecule) for molecule in molecules]

    expected = np.array(
        [DataStructs.BulkTanimotoSimilarity(reference, native) for reference in native]
    )
    actual = tanimoto_similarity(fingerprints(FAMILIAR), fingerprints(FAMILIAR))

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_a_molecule_is_identical_to_itself() -> None:
    matrix = fingerprints(["O=Cc1ccc(O)cc1"])

    assert tanimoto_similarity(matrix, matrix)[0, 0] == pytest.approx(1.0)


def test_similarity_is_symmetric() -> None:
    matrix = fingerprints(FAMILIAR)

    similarity = tanimoto_similarity(matrix, matrix)

    np.testing.assert_allclose(similarity, similarity.T, atol=1e-6)


def test_mismatched_widths_are_rejected() -> None:
    with pytest.raises(ValueError, match="widths differ"):
        tanimoto_similarity(np.zeros((2, 8)), np.zeros((2, 16)))


def test_a_fingerprint_with_no_bits_is_similar_to_nothing() -> None:
    """No shared evidence means no claim of similarity, not a division by zero."""
    empty = np.zeros((1, 16))

    assert tanimoto_similarity(empty, np.ones((3, 16))).max() == 0.0


# --- the domain --------------------------------------------------------------


def test_the_alien_molecule_is_less_familiar_than_the_familiar_ones() -> None:
    """The concrete failure from step 1.6, turned into an assertion."""
    domain = ApplicabilityDomain().fit(fingerprints(FAMILIAR))

    familiar = domain.max_similarity(fingerprints(FAMILIAR)).min()
    alien = domain.max_similarity(fingerprints([ALIEN]))[0]

    assert alien < familiar


def test_calibration_keeps_the_requested_share_inside() -> None:
    domain = ApplicabilityDomain().fit(fingerprints(FAMILIAR))
    held_out = fingerprints(FAMILIAR)

    domain.calibrate(held_out, coverage=0.9)

    assert domain.is_inside(held_out).mean() >= 0.9


def test_full_coverage_admits_everything_it_was_calibrated_on() -> None:
    matrix = fingerprints(FAMILIAR)
    domain = ApplicabilityDomain().fit(matrix).calibrate(matrix, coverage=1.0)

    assert domain.is_inside(matrix).all()


def test_a_stricter_threshold_never_admits_more() -> None:
    """Coverage is a dial, and it must turn in one direction only."""
    matrix = fingerprints(FAMILIAR)
    domain = ApplicabilityDomain().fit(matrix)

    loose = domain.calibrate(matrix, coverage=0.95).is_inside(matrix).sum()
    strict = domain.calibrate(matrix, coverage=0.60).is_inside(matrix).sum()

    assert strict <= loose


def test_impossible_coverage_is_rejected() -> None:
    with pytest.raises(ValueError, match="Coverage"):
        ApplicabilityDomain().fit(fingerprints(FAMILIAR)).calibrate(
            fingerprints(FAMILIAR), coverage=1.5
        )


def test_using_the_domain_before_fitting_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ApplicabilityDomain().max_similarity(np.zeros((1, 2048)))


def test_using_the_domain_before_calibrating_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="calibrate"):
        ApplicabilityDomain().fit(fingerprints(FAMILIAR)).is_inside(fingerprints([ALIEN]))


def test_an_explicit_threshold_needs_no_calibration() -> None:
    domain = ApplicabilityDomain(threshold=0.99).fit(fingerprints(FAMILIAR))

    assert domain.is_inside(fingerprints(FAMILIAR)).all()
    assert not domain.is_inside(fingerprints([ALIEN]))[0]


def test_empty_input_returns_no_scores() -> None:
    domain = ApplicabilityDomain().fit(fingerprints(FAMILIAR))

    assert domain.max_similarity(np.zeros((0, 2048))).shape == (0,)


# --- declared scope ----------------------------------------------------------


def test_an_ordinary_odorant_is_in_scope() -> None:
    assert ChemicalScope().reason_for_rejection(parse_smiles(["O=Cc1ccc(O)cc1"])[0]) is None


def test_the_salt_from_step_1_6_is_out_of_scope() -> None:
    """Basic magnesium carbonate: the worst molecule in the test set, refused by rule."""
    reason = ChemicalScope().reason_for_rejection(parse_smiles([ALIEN])[0])

    assert reason is not None
    assert "components" in reason


def test_an_inorganic_molecule_is_refused_for_having_no_carbon() -> None:
    assert "inorganic" in ChemicalScope().reason_for_rejection(parse_smiles(["[O]=[Mg]"])[0])


def test_a_charged_species_is_refused() -> None:
    scope = ChemicalScope(require_single_component=False)

    assert "ionic" in scope.reason_for_rejection(parse_smiles(["CC(=O)[O-]"])[0])


def test_a_molecule_too_heavy_to_evaporate_is_refused() -> None:
    triglyceride = "CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCC"

    reason = ChemicalScope().reason_for_rejection(parse_smiles([triglyceride])[0])

    assert "molecular weight" in reason


def test_contains_returns_one_flag_per_molecule() -> None:
    molecules = parse_smiles(["CCO", ALIEN, "O=Cc1ccc(O)cc1"])

    mask = ChemicalScope().contains(molecules)

    assert mask.tolist() == [True, False, True]


def test_the_bounds_are_configurable() -> None:
    """The scope is a stated policy, so it must be possible to state a different one."""
    heavy = parse_smiles(["CCCCCCCC(=O)OCC(COC(=O)CCCCCCC)OC(=O)CCCCCCC"])

    assert not ChemicalScope().contains(heavy)[0]
    assert ChemicalScope(max_weight=1000).contains(heavy)[0]
