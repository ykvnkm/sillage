"""The trained graph network's forward pass in NumPy, for the service (journal E-020).

Training needs torch; answering one request does not. The forward pass of a D-MPNN
is a few matrix products and sums over bonds, so the service runs it here and the
Docker image stays without torch's few hundred megabytes. Weights are an ``.npz`` of
plain arrays, loaded with ``allow_pickle=False``: unlike the joblib files, opening one
cannot run code.

:mod:`sillage.research.gnn` is the reference; ``tests/test_gnn.py`` holds the two to
the same output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sillage.features.graphs import MolecularGraph

PARAMETERS = (
    "bond_in.weight",
    "bond_step.weight",
    "atom_out.weight",
    "atom_out.bias",
    "hidden_layer.weight",
    "hidden_layer.bias",
    "output.weight",
    "output.bias",
)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


@dataclass(frozen=True, slots=True)
class GraphNetwork:
    """One trained D-MPNN: its weights as the torch module named them, and its depth."""

    weights: dict[str, np.ndarray]
    depth: int

    def embed(self, graph: MolecularGraph) -> np.ndarray:
        w = self.weights
        # Each bond in both directions, interleaved as in training: a->b, b->a, ...
        source = graph.bonds.reshape(-1)
        target = graph.bonds[:, ::-1].reshape(-1)
        reverse = np.arange(source.size) ^ 1
        edges = np.hstack([graph.atoms[source], np.repeat(graph.bond_features, 2, axis=0)])

        initial = edges @ w["bond_in.weight"].T
        state = _relu(initial)
        n_atoms = graph.n_atoms
        for _ in range(self.depth - 1):
            arriving = np.zeros((n_atoms, state.shape[1]), dtype=state.dtype)
            np.add.at(arriving, target, state)
            message = arriving[source] - state[reverse]
            state = _relu(initial + message @ w["bond_step.weight"].T)

        arriving = np.zeros((n_atoms, state.shape[1]), dtype=state.dtype)
        np.add.at(arriving, target, state)
        atom = _relu(
            np.hstack([graph.atoms, arriving]) @ w["atom_out.weight"].T + w["atom_out.bias"]
        )
        molecule = atom.mean(axis=0)
        return _relu(molecule @ w["hidden_layer.weight"].T + w["hidden_layer.bias"])

    def predict(self, graph: MolecularGraph) -> np.ndarray:
        logits = self.embed(graph) @ self.weights["output.weight"].T + self.weights["output.bias"]
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True, slots=True)
class GraphEnsemble:
    """Several networks trained from different seeds; their probabilities averaged."""

    members: tuple[GraphNetwork, ...]

    def predict(self, graph: MolecularGraph) -> np.ndarray:
        return np.mean([m.predict(graph) for m in self.members], axis=0)

    def embed(self, graph: MolecularGraph) -> np.ndarray:
        return np.concatenate([m.embed(graph) for m in self.members])

    def save(self, path: Path) -> None:
        arrays = {"depths": np.array([m.depth for m in self.members])}
        for i, member in enumerate(self.members):
            for name in PARAMETERS:
                arrays[f"{i}/{name}"] = member.weights[name]
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> GraphEnsemble:
        with np.load(path, allow_pickle=False) as data:
            depths = data["depths"]
            members = tuple(
                GraphNetwork({name: data[f"{i}/{name}"] for name in PARAMETERS}, int(depth))
                for i, depth in enumerate(depths)
            )
        return cls(members)

    @classmethod
    def from_state_dicts(cls, states: Sequence[dict], depths: Sequence[int]) -> GraphEnsemble:
        """From torch ``state_dict``s (tensors) or dicts of arrays."""
        members = []
        for state, depth in zip(states, depths, strict=True):
            weights = {}
            for name in PARAMETERS:
                value = state[name]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                weights[name] = np.asarray(value)
            members.append(GraphNetwork(weights, int(depth)))
        return cls(tuple(members))
