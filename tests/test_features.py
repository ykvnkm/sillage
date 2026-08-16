"""Properties the featurizers must hold, especially the one that prevents leakage."""

from __future__ import annotations

import numpy as np
import pytest

from sillage.features import (
    DESCRIPTOR_NAMES,
    EXCLUDED_DESCRIPTORS,
    FeatureMatrix,
    InvalidSmilesError,
    morgan_fingerprints,
    parse_smiles,
    rdkit_descriptors,
)

ETHANOL = "CCO"
BENZENE = "c1ccccc1"
VANILLIN = "O=Cc1ccc(O)c(OC)c1"
POTASSIUM_CHLORIDE = "[Cl-].[K+]"


# --- parsing -----------------------------------------------------------------


def test_valid_smiles_parse() -> None:
    assert len(parse_smiles([ETHANOL, BENZENE, VANILLIN])) == 3


def test_salt_with_several_components_parses_as_one_entry() -> None:
    """A dot in SMILES separates components; the entry is still a single row."""
    assert len(parse_smiles([POTASSIUM_CHLORIDE])) == 1


def test_invalid_smiles_raises_and_points_at_the_row() -> None:
    """Silently dropping a molecule would make the metrics describe a smaller set."""
    with pytest.raises(InvalidSmilesError, match="row 1"):
        parse_smiles([ETHANOL, "not-a-molecule", BENZENE])


# --- the property that makes pre-split featurization safe --------------------


@pytest.mark.parametrize("featurize", [morgan_fingerprints, rdkit_descriptors])
def test_features_do_not_depend_on_the_other_molecules_in_the_batch(featurize) -> None:
    """Each row is a pure function of its own molecule.

    This is what allows featurizing the whole dataset before splitting it. Any
    step that broke this property -- column-mean imputation, scaling, variance
    filtering -- would leak test-set information into the training features.
    """
    batch = featurize(parse_smiles([ETHANOL, BENZENE, VANILLIN]))
    alone = featurize(parse_smiles([BENZENE]))

    np.testing.assert_array_equal(batch.values[1], alone.values[0])


@pytest.mark.parametrize("featurize", [morgan_fingerprints, rdkit_descriptors])
def test_featurization_is_deterministic(featurize) -> None:
    molecules = parse_smiles([ETHANOL, VANILLIN])

    np.testing.assert_array_equal(featurize(molecules).values, featurize(molecules).values)


# --- fingerprints ------------------------------------------------------------


def test_morgan_has_the_requested_width_and_is_sparse() -> None:
    matrix = morgan_fingerprints(parse_smiles([ETHANOL, BENZENE, VANILLIN]), n_bits=2048)

    assert matrix.shape == (3, 2048)
    assert set(np.unique(matrix.values)) <= {0.0, 1.0}
    assert matrix.values.sum() < 3 * 2048 * 0.05


def test_identical_molecules_get_identical_fingerprints() -> None:
    matrix = morgan_fingerprints(parse_smiles([ETHANOL, ETHANOL]))

    np.testing.assert_array_equal(matrix.values[0], matrix.values[1])


def test_different_molecules_get_different_fingerprints() -> None:
    matrix = morgan_fingerprints(parse_smiles([ETHANOL, VANILLIN]))

    assert not np.array_equal(matrix.values[0], matrix.values[1])


def test_larger_radius_sees_more_substructures() -> None:
    """Radius controls how far around each atom fragments extend."""
    molecules = parse_smiles([VANILLIN])

    narrow = morgan_fingerprints(molecules, radius=1).values.sum()
    wide = morgan_fingerprints(molecules, radius=3).values.sum()

    assert wide > narrow


def test_count_fingerprints_record_repetition() -> None:
    """Hexane repeats the same CH2 fragment; a binary vector cannot say so."""
    molecules = parse_smiles(["CCCCCC"])

    assert morgan_fingerprints(molecules, counts=True).values.max() > 1


# --- descriptors -------------------------------------------------------------


def test_descriptor_block_matches_its_names() -> None:
    matrix = rdkit_descriptors(parse_smiles([ETHANOL, BENZENE]))

    assert matrix.shape == (2, len(DESCRIPTOR_NAMES))


def test_descriptor_set_is_pinned() -> None:
    """An RDKit upgrade that adds or removes descriptors changes the feature space.

    Failing here is the intended behaviour: the number must be updated knowingly,
    and any model trained before the change has to be retrained, not compared.
    """
    assert len(DESCRIPTOR_NAMES) == 216
    assert "Ipc" in EXCLUDED_DESCRIPTORS
    assert "Ipc" not in DESCRIPTOR_NAMES


def test_known_molecular_weight() -> None:
    """Ethanol is 46.07 g/mol. If this drifts, the descriptor block is misaligned."""
    matrix = rdkit_descriptors(parse_smiles([ETHANOL]))

    weight = matrix.values[0, DESCRIPTOR_NAMES.index("MolWt")]

    assert weight == pytest.approx(46.07, abs=0.05)


def test_salts_produce_finite_features() -> None:
    """Gasteiger charges are undefined for ions; the matrix must stay usable anyway."""
    matrix = rdkit_descriptors(parse_smiles([POTASSIUM_CHLORIDE]))

    assert np.isfinite(matrix.values).all()


def test_no_infinities_survive() -> None:
    matrix = rdkit_descriptors(parse_smiles([ETHANOL, BENZENE, VANILLIN, POTASSIUM_CHLORIDE]))

    assert np.isfinite(matrix.values).all()


# --- the container -----------------------------------------------------------


def test_empty_input_keeps_the_column_layout() -> None:
    assert morgan_fingerprints([]).shape == (0, 2048)
    assert rdkit_descriptors([]).shape == (0, len(DESCRIPTOR_NAMES))


def test_matrix_rejects_a_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="disagree"):
        FeatureMatrix(values=np.zeros((2, 3)), names=("a", "b"))


def test_hstack_keeps_names_aligned_with_columns() -> None:
    molecules = parse_smiles([ETHANOL, BENZENE])

    combined = morgan_fingerprints(molecules, n_bits=64).hstack(rdkit_descriptors(molecules))

    assert combined.shape == (2, 64 + len(DESCRIPTOR_NAMES))
    assert combined.names[63] == "morgan_bit_63"
    assert combined.names[64] == DESCRIPTOR_NAMES[0]
