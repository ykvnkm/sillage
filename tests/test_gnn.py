"""The message-passing network (journal E-020) on a handful of hand-made molecules."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

torch = pytest.importorskip("torch")

from sillage.features.graphs import molecular_graph  # noqa: E402
from sillage.research import gnn  # noqa: E402

SMILES = ("CCO", "CC(=O)O", "c1ccccc1O", "CC1=CC[C@@H](CC1=O)C(C)=C", "C")


def graphs():
    return [molecular_graph(Chem.MolFromSmiles(s)) for s in SMILES]


def test_batching_keeps_molecules_apart() -> None:
    """A molecule's prediction must not depend on which others share its batch."""
    torch.manual_seed(0)
    model = gnn.DMPNN(3, hidden=16, depth=3).eval()
    g = graphs()

    with torch.no_grad():
        together = model(gnn.collate(g))
        alone = torch.cat([model(gnn.collate([x])) for x in g])

    assert torch.allclose(together, alone, atol=1e-5)


def test_reverse_of_every_directed_bond_points_back() -> None:
    batch = gnn.collate(graphs())

    assert torch.equal(batch.source[batch.reverse], batch.target)
    assert torch.equal(batch.target[batch.reverse], batch.source)


def test_loss_ignores_unknown_cells_and_takes_soft_targets() -> None:
    logits = torch.zeros(2, 2)  # p = 0.5 everywhere
    targets = torch.tensor([[0.5, float("nan")], [float("nan"), 0.5]])

    loss = gnn.masked_soft_bce(logits, targets)

    assert loss.item() == pytest.approx(np.log(2), abs=1e-6)  # the minimum for a 0.5 target


def test_enantiomers_can_be_told_apart() -> None:
    """A label that only one enantiomer has is learnable from the graph alone."""
    torch.manual_seed(0)
    pair = [molecular_graph(Chem.MolFromSmiles(s)) for s in ("C[C@H](N)O", "C[C@@H](N)O")]
    batch = gnn.collate(pair)
    targets = torch.tensor([[1.0], [0.0]])
    model = gnn.DMPNN(1, hidden=16, depth=2, dropout=0.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(100):
        loss = gnn.masked_soft_bce(model(batch), targets)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    p = torch.sigmoid(model(batch)).detach()
    assert p[0, 0] > 0.9 and p[1, 0] < 0.1


def test_numpy_forward_pass_matches_torch(tmp_path) -> None:
    """The service runs the NumPy port; it must give what training measured."""
    from sillage.models.graph_network import GraphEnsemble

    torch.manual_seed(1)
    nets = [gnn.DMPNN(4, hidden=16, depth=d).eval() for d in (2, 4)]
    g = graphs()
    with torch.no_grad():
        expected = np.mean([torch.sigmoid(net(gnn.collate(g))).numpy() for net in nets], axis=0)
        expected_embedding = nets[0].embed(gnn.collate(g[:1])).numpy()[0]

    ensemble = GraphEnsemble.from_state_dicts([n.state_dict() for n in nets], [2, 4])
    ensemble.save(tmp_path / "graph.npz")
    loaded = GraphEnsemble.load(tmp_path / "graph.npz")

    got = np.stack([loaded.predict(x) for x in g])
    np.testing.assert_allclose(got, expected, atol=1e-5)
    np.testing.assert_allclose(loaded.members[0].embed(g[0]), expected_embedding, atol=1e-5)
