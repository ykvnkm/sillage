"""A directed message-passing network over molecular graphs (journal E-020).

The architecture is Chemprop's D-MPNN (Yang et al. 2019): hidden states live on
directed bonds, and a bond's message excludes what it itself sent the step before, so
information does not bounce straight back along the bond it came from. A few rounds of
this give each atom a view of its neighbourhood -- the learned analogue of a Morgan
fingerprint's radius, except nothing is hashed and the stereo labels on atoms and bonds
(see :mod:`sillage.features.graphs`) take part in every message.

Atoms are averaged into one molecule vector. Optionally the RDKit descriptors are
appended to it -- Chemprop's "hybrid" mode -- because a mean over atoms knows the
molecule's composition but not its size, and size is much of what makes a smell.

Training sees the same targets as the linear baseline (E-019): a label a collection
could not express is masked out of the loss, and a disputed one is a soft 0.5, which
binary cross-entropy takes as it is.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from sillage.features.graphs import ATOM_FEATURES, BOND_FEATURES, MolecularGraph
from sillage.research import train_v2


@dataclass(frozen=True, slots=True)
class Batch:
    """Several graphs as one disconnected graph, with directed bonds in both directions."""

    atoms: torch.Tensor  # (atoms, ATOM_FEATURES)
    edges: torch.Tensor  # (directed bonds, ATOM_FEATURES + BOND_FEATURES)
    source: torch.Tensor  # atom each directed bond leaves
    target: torch.Tensor  # atom it arrives at
    reverse: torch.Tensor  # index of the same bond in the opposite direction
    molecule: torch.Tensor  # molecule each atom belongs to
    n_molecules: int
    extra: torch.Tensor | None  # (molecules, extra features) or None


def collate(
    graphs: Sequence[MolecularGraph], extra: np.ndarray | None = None, device="cpu"
) -> Batch:
    atoms, edges, source, target, molecule = [], [], [], [], []
    offset = 0
    for index, graph in enumerate(graphs):
        a, b = graph.bonds[:, 0] + offset, graph.bonds[:, 1] + offset
        # Interleaved a->b, b->a, so a bond's reverse is its index with the last bit flipped.
        source.append(np.stack([a, b], axis=1).reshape(-1))
        target.append(np.stack([b, a], axis=1).reshape(-1))
        bond = np.repeat(graph.bond_features, 2, axis=0)
        edges.append(np.hstack([graph.atoms[source[-1] - offset], bond]))
        atoms.append(graph.atoms)
        molecule.append(np.full(graph.n_atoms, index))
        offset += graph.n_atoms
    source_all = np.concatenate(source)
    tensor = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)  # noqa: E731
    return Batch(
        atoms=tensor(np.concatenate(atoms)),
        edges=tensor(np.concatenate(edges).reshape(-1, ATOM_FEATURES + BOND_FEATURES)),
        source=tensor(source_all, torch.long),
        target=tensor(np.concatenate(target), torch.long),
        reverse=tensor(np.arange(source_all.size) ^ 1, torch.long),
        molecule=tensor(np.concatenate(molecule), torch.long),
        n_molecules=len(graphs),
        extra=None if extra is None else tensor(extra),
    )


class DMPNN(nn.Module):
    def __init__(
        self,
        n_labels: int,
        *,
        hidden: int = 300,
        depth: int = 4,
        dropout: float = 0.1,
        n_extra: int = 0,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.bond_in = nn.Linear(ATOM_FEATURES + BOND_FEATURES, hidden, bias=False)
        self.bond_step = nn.Linear(hidden, hidden, bias=False)
        self.atom_out = nn.Linear(ATOM_FEATURES + hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.hidden_layer = nn.Linear(hidden + n_extra, hidden)
        self.output = nn.Linear(hidden, n_labels)

    def embed(self, batch: Batch) -> torch.Tensor:
        """The molecule's representation just before the output layer."""
        initial = self.bond_in(batch.edges)
        state = functional.relu(initial)
        n_atoms = batch.atoms.shape[0]
        for _ in range(self.depth - 1):
            arriving = torch.zeros(n_atoms, state.shape[1], device=state.device)
            arriving.index_add_(0, batch.target, state)
            # What reaches the bond's source atom, minus what this bond's reverse sent.
            message = arriving[batch.source] - state[batch.reverse]
            state = self.dropout(functional.relu(initial + self.bond_step(message)))

        arriving = torch.zeros(n_atoms, state.shape[1], device=state.device)
        arriving.index_add_(0, batch.target, state)
        atom = self.dropout(functional.relu(self.atom_out(torch.cat([batch.atoms, arriving], 1))))

        total = torch.zeros(batch.n_molecules, atom.shape[1], device=atom.device)
        total.index_add_(0, batch.molecule, atom)
        counts = torch.bincount(batch.molecule, minlength=batch.n_molecules).clamp(min=1)
        molecule = total / counts[:, None]
        if batch.extra is not None:
            molecule = torch.cat([molecule, batch.extra], 1)
        return functional.relu(self.hidden_layer(self.dropout(molecule)))

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.output(self.dropout(self.embed(batch)))


def masked_soft_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over known cells only; a 0.5 target is learnt as it stands."""
    known = ~torch.isnan(targets)
    loss = functional.binary_cross_entropy_with_logits(
        logits, torch.nan_to_num(targets), reduction="none"
    )
    return (loss * known).sum() / known.sum().clamp(min=1)


@dataclass
class TrainingConfig:
    hidden: int = 300
    depth: int = 4
    dropout: float = 0.1
    epochs: int = 60
    batch_size: int = 50
    learning_rate: float = 1e-3
    patience: int = 15
    device: str = "cpu"


@dataclass
class Trained:
    model: DMPNN
    best_epoch: int
    best_val_ap: float
    history: list[float]


def train(
    graphs: Sequence[MolecularGraph],
    targets: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    val_truth: np.ndarray,
    *,
    extra: np.ndarray | None = None,
    config: TrainingConfig | None = None,
    seed: int = 0,
) -> Trained:
    """Fit on *train_rows*; keep the epoch with the best macro AP on *val_rows*."""
    config = config or TrainingConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DMPNN(
        targets.shape[1],
        hidden=config.hidden,
        depth=config.depth,
        dropout=config.dropout,
        n_extra=0 if extra is None else extra.shape[1],
    ).to(config.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    steps = config.epochs * int(np.ceil(train_rows.size / config.batch_size))
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=config.learning_rate, total_steps=steps, pct_start=0.1
    )

    best, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_rows)
        for start in range(0, order.size, config.batch_size):
            rows = order[start : start + config.batch_size]
            batch = _batch(graphs, extra, rows, config.device)
            y = torch.as_tensor(targets[rows], dtype=torch.float32, device=config.device)
            loss = masked_soft_bce(model(batch), y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()

        val_ap = train_v2.score(predict(model, graphs, val_rows, extra=extra), val_truth)[
            "macro_ap"
        ]
        history.append(val_ap)
        if val_ap > best:
            best, best_epoch, best_state = val_ap, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= config.patience:
            break

    model.load_state_dict(best_state)
    return Trained(model=model, best_epoch=best_epoch, best_val_ap=best, history=history)


@torch.no_grad()
def predict(
    model: DMPNN,
    graphs: Sequence[MolecularGraph],
    rows: np.ndarray,
    *,
    extra: np.ndarray | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    out = []
    for start in range(0, rows.size, batch_size):
        chunk = rows[start : start + batch_size]
        out.append(torch.sigmoid(model(_batch(graphs, extra, chunk, device))).cpu().numpy())
    return np.concatenate(out)


def _batch(graphs, extra, rows, device) -> Batch:
    return collate([graphs[i] for i in rows], None if extra is None else extra[rows], device)
