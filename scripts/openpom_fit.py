# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "openpom==1.0.0",
#   "torch==2.2.2",
#   "torchdata==0.7.1",
#   "dgl==2.2.0",
#   "numpy<2",
#   "pyyaml",
#   "pydantic",
# ]
# ///
"""Train OpenPOM's network on a split handed over by scripts/run_openpom.py.

Runs in an environment of its own (the block above; ``uv run --script`` builds it):
OpenPOM needs DeepChem, DGL and a torch that DGL was built for, none of which belong
in this project's lock file. Not called by hand.

The recipe is OpenPOM's examples/ensemble_benchmark.ipynb, unchanged: the model's
hyper-parameters, 62 epochs, the learning-rate decay, class-imbalance weights from the
training part, an ensemble as the plain mean of the members' predictions.
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import deepchem as dc  # noqa: E402
import torch  # noqa: E402
from openpom.feat.graph_featurizer import GraphConvConstants, GraphFeaturizer  # noqa: E402
from openpom.models.mpnn_pom import MPNNPOMModel  # noqa: E402
from openpom.utils.data_utils import get_class_imbalance_ratio  # noqa: E402


def main(source: str, out: str, members: int, epochs: int) -> int:
    handed = np.load(source, allow_pickle=False)
    featurizer = GraphFeaturizer()
    train = dc.data.NumpyDataset(featurizer.featurize(handed["train_smiles"]), handed["train_y"])
    test = dc.data.NumpyDataset(featurizer.featurize(handed["test_smiles"]))
    ratios = get_class_imbalance_ratio(train)
    n_tasks = handed["train_y"].shape[1]

    predictions = []
    for member in range(members):
        torch.manual_seed(member)
        np.random.seed(member)
        model = MPNNPOMModel(
            n_tasks=n_tasks,
            batch_size=128,
            learning_rate=dc.models.optimizers.ExponentialDecay(
                initial_rate=0.001, decay_rate=0.5, decay_steps=32 * 20, staircase=True
            ),
            class_imbalance_ratio=ratios,
            loss_aggr_type="sum",
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type="sum",
            mode="classification",
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type="set2set",
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation="relu",
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name="adam",
            log_frequency=32,
            model_dir=tempfile.mkdtemp(),
            device_name="cpu",
        )
        started = time.perf_counter()
        loss = model.fit(train, nb_epoch=epochs, max_checkpoints_to_keep=1, deterministic=False)
        predictions.append(np.asarray(model.predict(test)).reshape(len(test), n_tasks))
        # Saved after every member: a crash on the ninth must not cost the first eight.
        np.savez_compressed(out, members=np.stack(predictions))
        took = time.perf_counter() - started
        print(f"member {member + 1}/{members}: loss {loss:.4f}, {took:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
