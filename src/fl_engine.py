"""
Federated Learning Engine for IIoT Anomaly Detection
Implements:
- FedAvg: standard federated averaging
- FedProx: with proximal regularization for non-IID
- Top-K Compression: gradient sparsification for communication efficiency
- Robust Aggregators: Coordinate-wise Median, Trimmed Mean, Krum, Multi-Krum
- FLTrust-style: trust score weighting
Communication cost tracking included.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Optional, Callable
from torch.utils.data import DataLoader, TensorDataset

from src.attacks import FLAttack


# ─────────────────────────── Communication Tracking ──────────────────────────

class CommTracker:
    """Tracks bytes transmitted per round."""

    def __init__(self, bits_per_param: int = 32):
        self.bits_per_param = bits_per_param
        self.bytes_per_round = []

    def count_update_bytes(self, delta: dict, topk_ratio: float = 1.0) -> int:
        """Count bytes in a model update (considering compression)."""
        total_params = sum(v.numel() for v in delta.values())
        active_params = math.ceil(total_params * topk_ratio)
        bytes_sent = (active_params * self.bits_per_param) // 8
        return bytes_sent

    def record_round(self, n_clients: int, delta: dict, topk_ratio: float = 1.0):
        total = n_clients * self.count_update_bytes(delta, topk_ratio)
        self.bytes_per_round.append(total)

    def total_bytes(self) -> int:
        return sum(self.bytes_per_round)

    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.bytes_per_round)


# ─────────────────────────── Top-K Compressor ────────────────────────────────

class TopKCompressor:
    """
    Top-K sparsification for communication efficiency.
    Keeps only the top-k% largest (by magnitude) parameters.
    Error-feedback residual is tracked per client.
    Based on: Lin et al. (Deep Gradient Compression) and IIoT FL work, 2020.
    """

    def __init__(self, k_ratio: float = 0.1):
        assert 0.0 < k_ratio <= 1.0
        self.k_ratio = k_ratio
        self.residuals: Dict[int, dict] = {}  # error-feedback per client

    def compress(self, delta: dict, client_id: int) -> dict:
        """Apply top-k sparsification with error-feedback."""
        # Add residual from previous round
        if client_id not in self.residuals:
            self.residuals[client_id] = {k: torch.zeros_like(v) for k, v in delta.items()}

        compressed = {}
        new_residuals = {}

        for key, val in delta.items():
            # Add error-feedback residual
            val_corrected = val + self.residuals[client_id][key]

            flat = val_corrected.flatten()
            k = max(1, int(flat.numel() * self.k_ratio))
            _, top_indices = torch.topk(flat.abs(), k)

            mask = torch.zeros_like(flat)
            mask[top_indices] = 1.0

            compressed_flat = flat * mask
            compressed[key] = compressed_flat.reshape(val.shape)

            # Error-feedback: residual = full - compressed
            new_residuals[key] = (val_corrected - compressed[key]).detach()

        self.residuals[client_id] = new_residuals
        return compressed

    def compression_ratio(self) -> float:
        return self.k_ratio


# ─────────────────────────── Robust Aggregators ──────────────────────────────

def fedavg_aggregate(updates: List[dict], weights: Optional[List[float]] = None) -> dict:
    """Standard FedAvg: weighted average of updates."""
    if weights is None:
        weights = [1.0 / len(updates)] * len(updates)
    else:
        s = sum(weights)
        weights = [w / s for w in weights]

    agg = {}
    for key in updates[0].keys():
        agg[key] = sum(w * u[key] for w, u in zip(weights, updates))
    return agg


def coordinate_median_aggregate(updates: List[dict]) -> dict:
    """Coordinate-wise median aggregation. Byzantine-robust."""
    agg = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key] for u in updates], dim=0)
        agg[key] = stacked.median(dim=0).values
    return agg


def trimmed_mean_aggregate(updates: List[dict], trim_ratio: float = 0.1) -> dict:
    """
    Trimmed mean: remove top and bottom trim_ratio fraction per coordinate.
    Based on: Yin et al., ICML 2018.
    """
    n = len(updates)
    n_trim = max(1, int(n * trim_ratio))
    agg = {}
    for key in updates[0].keys():
        stacked = torch.stack([u[key] for u in updates], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        trimmed = sorted_vals[n_trim: n - n_trim]
        agg[key] = trimmed.mean(dim=0)
    return agg


def krum_aggregate(updates: List[dict], n_attackers: int = 1, multi_k: int = 1) -> dict:
    """
    Krum / Multi-Krum aggregation rule.
    Selects the update(s) with minimum sum of distances to n-f-2 nearest neighbors.
    Based on: Blanchard et al., NeurIPS 2017.
    """
    n = len(updates)

    # Flatten updates for distance computation
    flat_updates = []
    for u in updates:
        flat = torch.cat([v.flatten() for v in u.values()])
        flat_updates.append(flat)

    # Pairwise L2 distances
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = (flat_updates[i] - flat_updates[j]).norm()

    # Krum score: sum of n-f-2 smallest distances
    k_neighbors = n - n_attackers - 2
    k_neighbors = max(1, k_neighbors)
    scores = []
    for i in range(n):
        d_sorted = dists[i].sort().values
        score = d_sorted[1: k_neighbors + 1].sum()  # skip self (0)
        scores.append(score.item())

    # Select top multi_k updates
    selected_indices = np.argsort(scores)[:multi_k]
    selected_updates = [updates[i] for i in selected_indices]

    return fedavg_aggregate(selected_updates)


def rfa_aggregate(updates: List[dict], n_iters: int = 5) -> dict:
    """
    Robust Federated Aggregation via approximate geometric median (Weiszfeld).
    Based on: Pillutla et al., RFA 2019/2022.
    """
    # Initialize with mean
    agg = fedavg_aggregate(updates)

    for _ in range(n_iters):
        # Compute weights inversely proportional to distance from current estimate
        weights = []
        for u in updates:
            diff = {k: u[k] - agg[k] for k in agg.keys()}
            norm = sum(v.norm().item() ** 2 for v in diff.values()) ** 0.5
            weights.append(1.0 / max(norm, 1e-6))

        total = sum(weights)
        weights = [w / total for w in weights]

        # Recompute weighted mean
        new_agg = {}
        for key in updates[0].keys():
            new_agg[key] = sum(w * u[key] for w, u in zip(weights, updates))
        agg = new_agg

    return agg


def fltrust_aggregate(updates: List[dict], server_update: dict) -> dict:
    """
    FLTrust: weight updates by cosine similarity with server's root model update.
    Based on: Cao et al., NDSS 2021.
    """
    server_flat = torch.cat([v.flatten() for v in server_update.values()])
    server_norm = server_flat.norm()

    agg = None
    total_weight = 0.0

    for u in updates:
        flat = torch.cat([v.flatten() for v in u.values()])
        # Cosine similarity (trust score)
        ts = F.cosine_similarity(flat.unsqueeze(0), server_flat.unsqueeze(0)).item()
        ts = max(0.0, ts)  # ReLU: negative = no trust

        # Normalize client update to server norm
        client_norm = flat.norm()
        if client_norm > 0:
            scale = server_norm / client_norm
        else:
            scale = 0.0

        total_weight += ts
        if agg is None:
            agg = {k: ts * scale * v for k, v in u.items()}
        else:
            for k in agg.keys():
                agg[k] += ts * scale * u[k]

    if total_weight < 1e-8:
        return fedavg_aggregate(updates)

    for k in agg.keys():
        agg[k] /= total_weight

    return agg


def get_aggregator(name: str) -> Callable:
    """Return aggregation function by name."""
    aggs = {
        "fedavg": fedavg_aggregate,
        "median": coordinate_median_aggregate,
        "trimmed_mean": lambda u: trimmed_mean_aggregate(u, trim_ratio=0.1),
        "krum": lambda u: krum_aggregate(u, n_attackers=1, multi_k=1),
        "rfa": rfa_aggregate,
    }
    if name not in aggs:
        raise ValueError(f"Unknown aggregator: {name}. Choose from {list(aggs.keys())}")
    return aggs[name]


# ─────────────────────────── Client Local Training ───────────────────────────

def local_train_fedavg(
    model: nn.Module,
    X_train: np.ndarray,
    n_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple:
    """Standard FedAvg local training. Returns (updated_model, delta, train_losses)."""
    model = model.to(device)
    model.train()

    global_params = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    for epoch in range(n_epochs):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(tensor))

    # Compute delta = updated - global (always keep on CPU for aggregation)
    updated_params = model.state_dict()
    delta = {k: updated_params[k].float().cpu() - global_params[k].float().cpu() for k in global_params}

    return model, delta, losses


def local_train_fedprox(
    model: nn.Module,
    global_model: nn.Module,
    X_train: np.ndarray,
    n_epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    mu: float = 0.01,   # proximal regularization coefficient
    device: str = "cpu",
) -> tuple:
    """
    FedProx local training with proximal term: ||w - w_global||^2.
    Based on: Li et al., MLSys 2020.
    """
    model = model.to(device)
    global_model = global_model.to(device)
    model.train()

    global_params = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})
    global_weights = {k: v.clone().to(device) for k, v in global_model.state_dict().items()}

    tensor = torch.tensor(X_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    losses = []
    for epoch in range(n_epochs):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon = model(xb)
            # Reconstruction loss
            loss = F.mse_loss(recon, xb)
            # Proximal term
            prox = 0.0
            for k, p in model.named_parameters():
                if k in global_weights:
                    prox += (p - global_weights[k]).norm() ** 2
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)
        losses.append(ep_loss / len(tensor))

    updated_params = model.state_dict()
    delta = {k: updated_params[k].float().cpu() - global_params[k].float().cpu() for k in global_params}
    return model, delta, losses


# ─────────────────────────── FL Round ────────────────────────────────────────

def fl_round(
    global_model: nn.Module,
    client_data: list,
    selected_clients: list,
    aggregator_name: str = "fedavg",
    local_epochs: int = 3,
    lr: float = 1e-3,
    mu: float = 0.0,  # >0 for FedProx
    compressor: Optional[TopKCompressor] = None,
    attacker_assignments: Optional[dict] = None,
    device: str = "cpu",
    comm_tracker: Optional[CommTracker] = None,
) -> tuple:
    """
    Execute one federated learning round.
    Returns: (new_global_model, round_losses, bytes_this_round)
    """
    updates = []
    all_losses = []
    weights = []

    for cid in selected_clients:
        # Copy global model for client
        client_model = copy.deepcopy(global_model)
        X_train = client_data[cid]["X_train"]
        n_samples = len(X_train)

        # Local training
        if mu > 0:
            client_model, delta, losses = local_train_fedprox(
                client_model, global_model, X_train,
                n_epochs=local_epochs, lr=lr, mu=mu, device=device
            )
        else:
            client_model, delta, losses = local_train_fedavg(
                client_model, X_train,
                n_epochs=local_epochs, lr=lr, device=device
            )

        # Apply attack if this client is an attacker
        if attacker_assignments and attacker_assignments.get(cid) is not None:
            delta = attacker_assignments[cid].poison_update(delta)

        # Compression
        if compressor is not None:
            delta = compressor.compress(delta, client_id=cid)

        updates.append(delta)
        all_losses.extend(losses)
        weights.append(n_samples)

    # Aggregate
    aggregator = get_aggregator(aggregator_name)
    if aggregator_name == "fedavg":
        agg_delta = aggregator(updates, weights=[w / sum(weights) for w in weights])
    else:
        agg_delta = aggregator(updates)

    # Apply aggregated update to global model
    new_state = copy.deepcopy(global_model.state_dict())
    model_device = next(global_model.parameters()).device
    for k in new_state:
        if k in agg_delta:
            new_state[k] = new_state[k].float() + agg_delta[k].float().to(model_device)
    global_model.load_state_dict(new_state)

    # Track communication
    topk_ratio = compressor.compression_ratio() if compressor else 1.0
    bytes_this_round = 0
    if comm_tracker is not None:
        for u in updates:
            bytes_this_round += comm_tracker.count_update_bytes(u, topk_ratio)
        comm_tracker.bytes_per_round.append(bytes_this_round)

    return global_model, np.mean(all_losses) if all_losses else 0.0, bytes_this_round


# ─────────────────────────── Full FL Training Loop ───────────────────────────

def federated_training(
    model: nn.Module,
    client_data: list,
    n_rounds: int = 50,
    n_clients_per_round: int = 3,
    aggregator_name: str = "fedavg",
    local_epochs: int = 3,
    lr: float = 1e-3,
    mu: float = 0.0,
    topk_ratio: Optional[float] = None,
    attack_fraction: float = 0.0,
    attack_type: str = "none",
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full federated training with configurable aggregator, compression, and attacks.
    Returns dict with training history.
    """
    from src.attacks import assign_attackers

    rng = np.random.RandomState(seed)
    n_clients = len(client_data)

    compressor = TopKCompressor(k_ratio=topk_ratio) if topk_ratio and topk_ratio < 1.0 else None
    comm_tracker = CommTracker()
    attacker_assignments = None

    if attack_fraction > 0 and attack_type != "none":
        attacker_assignments = assign_attackers(n_clients, attack_fraction, attack_type, seed)

    history = {
        "round_losses": [],
        "bytes_per_round": [],
        "cumulative_bytes": [],
    }

    for r in range(n_rounds):
        # Select clients for this round
        selected = rng.choice(n_clients, size=min(n_clients_per_round, n_clients), replace=False).tolist()

        model, avg_loss, bytes_r = fl_round(
            global_model=model,
            client_data=client_data,
            selected_clients=selected,
            aggregator_name=aggregator_name,
            local_epochs=local_epochs,
            lr=lr,
            mu=mu,
            compressor=compressor,
            attacker_assignments=attacker_assignments,
            device=device,
            comm_tracker=comm_tracker,
        )

        history["round_losses"].append(avg_loss)
        history["bytes_per_round"].append(bytes_r)
        history["cumulative_bytes"].append(sum(history["bytes_per_round"]))

        if verbose and (r % 10 == 0 or r == n_rounds - 1):
            mb = history["cumulative_bytes"][-1] / 1e6
            print(
                f"  Round {r+1:3d}/{n_rounds}: loss={avg_loss:.6f}, "
                f"bytes={bytes_r/1e3:.1f}KB, total={mb:.2f}MB"
            )

    return history
