"""
Adversarial Attacks for Federated Learning
Implements model poisoning, label flipping, and on-off attack strategies.
"""

import numpy as np
import copy
import torch
import torch.nn as nn
from typing import List


# ─────────────────────────── Attack Base ─────────────────────────────────────

class FLAttack:
    """Base class for federated learning attacks."""

    def __init__(self, attack_strength: float = 1.0, seed: int = 42):
        self.attack_strength = attack_strength
        self.rng = np.random.RandomState(seed)

    def poison_update(self, delta: dict) -> dict:
        """Modify the model update dict. Override in subclasses."""
        return delta


# ─────────────────────────── Model Poisoning Attack ──────────────────────────

class ModelPoisoningAttack(FLAttack):
    """
    Gaussian noise model poisoning attack.
    Replaces gradient update with scaled random noise to degrade global model.
    Based on: Fang et al. (Local model poisoning attacks), USENIX Security 2020.
    """

    def __init__(self, attack_strength: float = 5.0, seed: int = 42):
        super().__init__(attack_strength, seed)
        self.name = "model_poisoning"

    def poison_update(self, delta: dict) -> dict:
        """Replace update with large random noise scaled by attack_strength."""
        poisoned = {}
        for k, v in delta.items():
            noise = torch.randn_like(v) * self.attack_strength
            poisoned[k] = noise
        return poisoned


class ScaledPoisoningAttack(FLAttack):
    """
    Scaled model poisoning: amplify gradient in opposite direction.
    Simulates 'model replacement' style attack.
    Based on: Bhagoji et al. (Adversarial Lens), ICML 2019; Bagdasaryan et al. 2020.
    """

    def __init__(self, scale: float = -5.0, seed: int = 42):
        super().__init__(abs(scale), seed)
        self.scale = scale
        self.name = "scaled_poisoning"

    def poison_update(self, delta: dict) -> dict:
        """Negate and scale the update to push model in wrong direction."""
        poisoned = {}
        for k, v in delta.items():
            poisoned[k] = v * self.scale
        return poisoned


# ─────────────────────────── On-Off Attack ───────────────────────────────────

class OnOffAttack(FLAttack):
    """
    On-Off attack: client alternates between honest and malicious behavior
    to evade detection by anomaly-based defenses.
    Based on: Attacks against FL Defense Systems, JMLR 2023.
    """

    def __init__(
        self,
        base_attack: FLAttack,
        honest_rounds: int = 3,
        attack_rounds: int = 2,
        seed: int = 42,
    ):
        super().__init__(base_attack.attack_strength, seed)
        self.base_attack = base_attack
        self.honest_rounds = honest_rounds
        self.attack_rounds = attack_rounds
        self.round_counter = 0
        self.name = "on_off"

    def is_attacking(self) -> bool:
        cycle = self.honest_rounds + self.attack_rounds
        pos = self.round_counter % cycle
        return pos >= self.honest_rounds

    def poison_update(self, delta: dict) -> dict:
        self.round_counter += 1
        if self.is_attacking():
            return self.base_attack.poison_update(delta)
        return delta  # honest round


# ─────────────────────────── Label Flipping Attack ───────────────────────────

class LabelFlippingAttack(FLAttack):
    """
    Label flipping for semi-supervised/supervised settings.
    Flips anomaly labels to normal during local training.
    Implemented as data poisoning at the client level.
    Based on: on-off / label-flipping variations in FL attack literature.
    """

    def __init__(self, flip_ratio: float = 1.0, seed: int = 42):
        super().__init__(1.0, seed)
        self.flip_ratio = flip_ratio
        self.name = "label_flipping"

    def corrupt_data(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Flip anomaly labels to 0 (normal) for a fraction of anomalous samples."""
        y_corrupted = y.copy()
        anomaly_indices = np.where(y == 1)[0]
        n_flip = int(len(anomaly_indices) * self.flip_ratio)
        if n_flip > 0:
            flip_indices = self.rng.choice(anomaly_indices, size=n_flip, replace=False)
            y_corrupted[flip_indices] = 0
        return X, y_corrupted

    def poison_update(self, delta: dict) -> dict:
        """For autoencoder (unsupervised) setting, add noise to simulate label flipping effect."""
        poisoned = {}
        for k, v in delta.items():
            noise = torch.randn_like(v) * 0.5
            poisoned[k] = v + noise
        return poisoned


# ─────────────────────────── Attack Factory ──────────────────────────────────

def create_attack(attack_type: str, **kwargs) -> FLAttack:
    """Factory to create attack instances."""
    if attack_type == "none":
        return None
    elif attack_type == "model_poisoning":
        return ModelPoisoningAttack(
            attack_strength=kwargs.get("attack_strength", 5.0),
            seed=kwargs.get("seed", 42),
        )
    elif attack_type == "scaled_poisoning":
        return ScaledPoisoningAttack(
            scale=kwargs.get("scale", -5.0),
            seed=kwargs.get("seed", 42),
        )
    elif attack_type == "on_off":
        base = ModelPoisoningAttack(
            attack_strength=kwargs.get("attack_strength", 5.0),
            seed=kwargs.get("seed", 42),
        )
        return OnOffAttack(
            base,
            honest_rounds=kwargs.get("honest_rounds", 3),
            attack_rounds=kwargs.get("attack_rounds", 2),
        )
    elif attack_type == "label_flipping":
        return LabelFlippingAttack(
            flip_ratio=kwargs.get("flip_ratio", 1.0),
            seed=kwargs.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


def assign_attackers(n_clients: int, attack_fraction: float, attack_type: str, seed: int = 42) -> dict:
    """
    Assign attackers to a subset of clients.
    Returns dict: {client_id: FLAttack or None}
    """
    rng = np.random.RandomState(seed)
    n_attackers = max(0, int(n_clients * attack_fraction))
    attacker_ids = set(rng.choice(n_clients, size=n_attackers, replace=False).tolist())

    assignments = {}
    for cid in range(n_clients):
        if cid in attacker_ids:
            assignments[cid] = create_attack(attack_type, seed=seed + cid)
        else:
            assignments[cid] = None

    if n_attackers > 0:
        print(f"  Attackers ({attack_type}): clients {sorted(attacker_ids)} ({n_attackers}/{n_clients})")
    return assignments
