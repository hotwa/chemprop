import numpy as np
import torch

from chemprop.data.molgraph import MolGraph
from chemprop.data.collate import BatchMolGraph
from chemprop.nn.message_passing.base import BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from chemprop.models.model import MPNN


def _make_tiny_batch():
    # Two atoms (V=2), one undirected bond => two directed edges (E_dir=2)
    d_v = 5
    d_e = 3
    V = np.random.randn(2, d_v).astype(np.float32)
    # two directed edges: 0->1 and 1->0
    E = np.random.randn(2, d_e).astype(np.float32)
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)  # shape 2xE
    rev_edge_index = np.array([1, 0], dtype=np.int64)
    mg = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
    bmg = BatchMolGraph([mg])
    return bmg


def test_bond_message_passing_returns_edge_features():
    bmg = _make_tiny_batch()

    d_v = bmg.V.shape[1]
    d_e = bmg.E.shape[1]
    d_h = 16

    mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=d_h, depth=2)

    # 1) default behavior: only node/atom features
    H_atom_only = mp(bmg)
    assert isinstance(H_atom_only, torch.Tensor)
    assert H_atom_only.shape[0] == bmg.V.shape[0]

    # 2) new behavior: also return directed edge hidden states
    H_atom, H_edge = mp(bmg, return_edge_features=True)
    assert isinstance(H_atom, torch.Tensor) and isinstance(H_edge, torch.Tensor)
    assert H_atom.shape[0] == bmg.V.shape[0]
    assert H_edge.shape[0] == bmg.edge_index.shape[1]
    assert H_edge.shape[1] == d_h


def test_model_encode_multi_shapes():
    bmg = _make_tiny_batch()

    d_v = bmg.V.shape[1]
    d_e = bmg.E.shape[1]
    d_h = 16

    mp = BondMessagePassing(d_v=d_v, d_e=d_e, d_h=d_h, depth=2)
    agg = MeanAggregation()
    predictor = RegressionFFN(n_tasks=1, input_dim=mp.output_dim, hidden_dim=32, n_layers=1)
    model = MPNN(message_passing=mp, agg=agg, predictor=predictor)
    model.eval()

    h_g, H_atom, H_edge = model.encode_multi(bmg, None, None, return_masks=False)
    assert h_g.shape[0] == 1  # single graph
    assert H_atom.shape[0] == bmg.V.shape[0]
    # H_edge is present for bond-based message passing
    assert H_edge is not None and H_edge.shape[0] == bmg.edge_index.shape[1]

