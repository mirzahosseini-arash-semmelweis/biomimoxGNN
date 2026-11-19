# biomimox_models.py
# Model definitions shared by the single-task (arm 5) and multitask (arm 6) runs.
#
# Architecture: a dual encoder, which prevents information leaks.
#
#        3-D branch            2-D branch                 globals
#   DimeNet++ / SchNet     GINE over REAL bonds      RDKit + HOT flags + Percepta
#   (z, pos)  -> hidden    (x, edge_attr) -> hidden   -> scaled vector
#            \                   |                        /
#             \                  |                       /
#              ----------  concat + MLP head  ----------
#                                |
#                      n_tasks logits (1 for single-task)
#
#  * the 3-D branch sees geometry over a conformer ensemble
#  * the 2-D branch consumes Data.x and Data.edge_attr
#  * set BACKBONE="none" to get a pure 2-D model, or USE_2D_BRANCH=False to get
#    the original geometry-only behaviour
#
# Backbone options:
#   "dimenet++"  default; most expressive, slowest
#   "schnet"     noticeably faster, use when DimeNet++ is too slow
#   "none"       2-D only (GINE); seconds per fold

from __future__ import annotations

import inspect
import os
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn.models import DimeNetPlusPlus

try:
    from torch_geometric.nn.models import SchNet
    _HAS_SCHNET = True
except Exception:  # pragma: no cover
    SchNet = None
    _HAS_SCHNET = False

import biomimox_common as C


# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


# -----------------------------------------------------------------------------
# 3-D backbones
# -----------------------------------------------------------------------------
DPP_INT_EMB_SIZE = int(os.environ.get("BIOMIMOX_DPP_INT_EMB", "64"))
DPP_BASIS_EMB_SIZE = int(os.environ.get("BIOMIMOX_DPP_BASIS_EMB", "8"))


def build_dimenetpp(hidden: int,
                    out_channels: int,
                    cutoff: float,
                    num_blocks: int = 4,
                    num_spherical: int = 7,
                    num_radial: int = 6,
                    max_neighbors: int = 32) -> nn.Module:
    """DimeNet++ with the published embedding sizes.

    int_emb_size and basis_emb_size are NOT tied to `hidden`. They size the
    per-triplet work inside each interaction block, and the triplet count is
    large: a 45-atom molecule (H's included) yields ~24 000 triplets at
    cutoff 5 A / 32 neighbours, so a batch of 16 carries a few hundred
    thousand. `lin_sbf2` maps basis_emb_size -> int_emb_size once per triplet,
    so scaling those with `hidden` (basis_emb_size = hidden//2 and
    int_emb_size = hidden, as an earlier version did) multiplied that matmul by
    4x at hidden=64 and 16x at hidden=128 for roughly a 10% change in parameter
    count -- all cost, no capacity. The reference values are 64 and 8.
    """
    params = set(inspect.signature(DimeNetPlusPlus).parameters.keys())
    kwargs: Dict[str, object] = {}
    if "hidden_channels" in params:
        kwargs["hidden_channels"] = hidden
    if "out_channels" in params:
        kwargs["out_channels"] = out_channels
    if "int_emb_size" in params:
        kwargs["int_emb_size"] = min(DPP_INT_EMB_SIZE, hidden)
    if "basis_emb_size" in params:
        kwargs["basis_emb_size"] = DPP_BASIS_EMB_SIZE
    if "out_emb_channels" in params:
        kwargs["out_emb_channels"] = out_channels
    if "num_blocks" in params:
        kwargs["num_blocks"] = num_blocks
    if "num_spherical" in params:
        kwargs["num_spherical"] = num_spherical
    if "num_radial" in params:
        kwargs["num_radial"] = num_radial
    if "cutoff" in params:
        kwargs["cutoff"] = cutoff
    if "max_num_neighbors" in params:
        kwargs["max_num_neighbors"] = max_neighbors
    if "envelope_exponent" in params:
        kwargs["envelope_exponent"] = 5
    if "num_before_skip" in params:
        kwargs["num_before_skip"] = 1
    if "num_after_skip" in params:
        kwargs["num_after_skip"] = 2
    if "num_output_layers" in params:
        kwargs["num_output_layers"] = 3
    return DimeNetPlusPlus(**kwargs)


def build_schnet(hidden: int,
                 out_channels: int,
                 cutoff: float,
                 num_interactions: int = 4,
                 num_gaussians: int = 50,
                 max_neighbors: int = 32) -> nn.Module:
    if not _HAS_SCHNET:
        raise ImportError("SchNet is not available in this torch_geometric build")
    params = set(inspect.signature(SchNet).parameters.keys())
    kwargs: Dict[str, object] = {}
    if "hidden_channels" in params:
        kwargs["hidden_channels"] = hidden
    if "num_filters" in params:
        kwargs["num_filters"] = hidden
    if "num_interactions" in params:
        kwargs["num_interactions"] = num_interactions
    if "num_gaussians" in params:
        kwargs["num_gaussians"] = num_gaussians
    if "cutoff" in params:
        kwargs["cutoff"] = cutoff
    if "max_num_neighbors" in params:
        kwargs["max_num_neighbors"] = max_neighbors
    if "readout" in params:
        kwargs["readout"] = "add"
    if "dipole" in params:
        kwargs["dipole"] = False
    model = SchNet(**kwargs)

    # SchNet's head ends in Linear(hidden//2 -> 1) before the graph readout, so
    # out of the box it returns a scalar per graph. Widen it so the branch can
    # be concatenated with the 2-D branch and the globals.
    if hasattr(model, "lin2") and isinstance(model.lin2, nn.Linear):
        model.lin2 = nn.Linear(model.lin2.in_features, int(out_channels))
    return model


# -----------------------------------------------------------------------------
# 2-D branch over the real bond graph
# -----------------------------------------------------------------------------
class BondGraphEncoder(nn.Module):
    """GINE message passing on the molecular bond graph.
    """

    def __init__(self,
                 atom_in: int = C.ATOM_FEATURE_DIM,
                 bond_in: int = C.BOND_FEATURE_DIM,
                 hidden: int = 64,
                 num_layers: int = 3,
                 activation: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(atom_in, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(int(num_layers)):
            mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden),
                get_activation(activation),
                nn.Linear(2 * hidden, hidden),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True, edge_dim=bond_in))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, hidden)
        self.out_dim = hidden

    def forward(self, data: Data) -> torch.Tensor:
        h = self.embed(data.x)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if edge_attr is None or edge_attr.numel() == 0:
            edge_attr = torch.zeros((edge_index.size(1), C.BOND_FEATURE_DIM),
                                    device=h.device, dtype=h.dtype)
        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = self.act(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new                      # residual: helps at n ~ 100
        pooled = torch.cat([global_add_pool(h, data.batch),
                            global_mean_pool(h, data.batch)], dim=1)
        return self.out(pooled)


# -----------------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------------
class BiomimoxNet(nn.Module):
    def __init__(self,
                 global_in: int,
                 n_tasks: int = 1,
                 backbone: str = "dimenet++",
                 hidden: int = 64,
                 activation: str = "relu",
                 dropout: float = 0.1,
                 cutoff: float = 5.0,
                 max_neighbors: int = 32,
                 dpp_blocks: int = 4,
                 dpp_spherical: int = 7,
                 dpp_radial: int = 6,
                 schnet_interactions: int = 4,
                 schnet_gaussians: int = 50,
                 use_2d_branch: bool = True,
                 gine_layers: int = 3):
        super().__init__()
        self.backbone_name = str(backbone).lower()
        self.n_tasks = int(n_tasks)
        self.use_2d_branch = bool(use_2d_branch)

        feat_dim = 0

        if self.backbone_name in ("dimenet++", "dimenetpp", "dpp"):
            self.backbone = build_dimenetpp(
                hidden=hidden, out_channels=hidden, cutoff=cutoff,
                num_blocks=dpp_blocks, num_spherical=dpp_spherical,
                num_radial=dpp_radial, max_neighbors=max_neighbors,
            )
            feat_dim += hidden
        elif self.backbone_name == "schnet":
            self.backbone = build_schnet(
                hidden=hidden, out_channels=hidden, cutoff=cutoff,
                num_interactions=schnet_interactions,
                num_gaussians=schnet_gaussians, max_neighbors=max_neighbors,
            )
            feat_dim += hidden
        elif self.backbone_name in ("none", "2d", "gine"):
            self.backbone = None
            if not self.use_2d_branch:
                raise ValueError("backbone='none' requires use_2d_branch=True")
        else:
            raise ValueError("backbone must be 'dimenet++', 'schnet' or 'none'")

        if self.use_2d_branch:
            self.encoder_2d = BondGraphEncoder(
                atom_in=C.ATOM_FEATURE_DIM, bond_in=C.BOND_FEATURE_DIM,
                hidden=hidden, num_layers=gine_layers,
                activation=activation, dropout=dropout,
            )
            feat_dim += self.encoder_2d.out_dim
        else:
            self.encoder_2d = None

        self.global_norm = nn.LayerNorm(int(global_in))
        feat_dim += int(global_in)

        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_tasks),
        )
        self.feat_dim = feat_dim

    def encode(self, data: Data) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if self.backbone is not None:
            fargs = set(inspect.signature(self.backbone.forward).parameters.keys())
            kwargs: Dict[str, torch.Tensor] = {}
            if "z" in fargs:
                kwargs["z"] = data.z
            if "pos" in fargs:
                kwargs["pos"] = data.pos
            if "batch" in fargs:
                kwargs["batch"] = data.batch
            # Deliberately NOT passing a precomputed radius graph: the 3-D
            # backbones build their own neighbour lists from `pos`, and the
            # bond graph on data.edge_index belongs to the 2-D branch.
            h3d = self.backbone(**kwargs)
            h3d = torch.nan_to_num(h3d, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(h3d)

        if self.encoder_2d is not None:
            parts.append(self.encoder_2d(data))

        g = data.g
        n_graphs = int(data.batch.max().item()) + 1 if data.batch is not None else 1
        if g.dim() == 1:
            g = g.view(n_graphs, -1)
        elif g.size(0) != n_graphs:
            g = g.view(n_graphs, -1)
        parts.append(self.global_norm(g))

        return torch.cat(parts, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data))


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
class MaskedBCEWithLogitsLoss(nn.Module):
    """BCE that ignores missing labels, with an optional per-task pos_weight.

    For the multitask run this is what keeps a molecule with a label for
    N_dealk but not for S_ox from contributing noise to the S_ox head.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        if pos_weight is None:
            self.pos_weight = None
        else:
            self.register_buffer("pos_weight_buf", pos_weight.view(1, -1))
            self.pos_weight = True

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(targets)
        pw = self.pos_weight_buf.to(logits.device) if self.pos_weight else None
        per_elem = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, weight=None, pos_weight=pw, reduction="none"
        )
        per_elem = per_elem * mask
        denom = mask.sum().clamp_min(1.0)
        return per_elem.sum() / denom


class MaskedFocalLoss(nn.Module):
    """Masked focal loss."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha_buf", alpha.view(1, -1))
            self.alpha = True

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(targets)
        p = torch.sigmoid(logits)
        eps = 1e-8
        ce = -(targets * torch.log(p.clamp_min(eps))
               + (1 - targets) * torch.log((1 - p).clamp_min(eps)))
        pt = p * targets + (1 - p) * (1 - targets)
        loss = (1 - pt).pow(self.gamma) * ce
        if self.alpha:
            a = self.alpha_buf.to(logits.device)
            loss = loss * (a * targets + (1 - a) * (1 - targets))
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)


def compute_pos_weight(y: "object", n_tasks: int = 1) -> torch.Tensor:
    """neg/pos per task, computed on training labels only."""
    import numpy as np
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = []
    for t in range(arr.shape[1]):
        col = arr[:, t]
        col = col[np.isfinite(col)]
        pos = max(float((col == 1).sum()), 1.0)
        neg = max(float((col == 0).sum()), 1.0)
        out.append(neg / pos)
    w = torch.tensor(out, dtype=torch.float32)
    if w.numel() != n_tasks:
        w = w.expand(n_tasks).clone()
    return w


def make_loss(loss_name: str,
              y_train: "object",
              n_tasks: int,
              device: torch.device) -> nn.Module:
    import numpy as np
    if str(loss_name).lower() == "focal":
        arr = np.asarray(y_train, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        alphas = []
        for t in range(arr.shape[1]):
            col = arr[:, t]
            col = col[np.isfinite(col)]
            prev = float(col.mean()) if col.size else 0.5
            prev = min(max(prev, 1e-3), 1 - 1e-3)
            inv = 1.0 / prev
            alphas.append(float(inv / (inv + 1.0)))
        return MaskedFocalLoss(alpha=torch.tensor(alphas, dtype=torch.float32),
                               gamma=2.0).to(device)
    return MaskedBCEWithLogitsLoss(
        pos_weight=compute_pos_weight(y_train, n_tasks=n_tasks)
    ).to(device)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
