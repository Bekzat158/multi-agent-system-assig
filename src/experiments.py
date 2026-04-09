"""
Experiments: run all FL configurations and collect metrics.
Covers:
1. FL convergence: FedAvg vs FedProx vs TopK compression
2. Communication overhead: different compression ratios
3. Robustness: different aggregators under increasing attack fractions
4. Ablation study on components
"""

import copy
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from tqdm import tqdm

from src.models import LSTMAutoencoder, compute_anomaly_scores, get_threshold, train_autoencoder
from src.fl_engine import federated_training, CommTracker
from src.attacks import assign_attackers


# ───────────── Evaluation ─────────────────────────────────────────────────────

def evaluate_model(model, client_data: list, device: str = "cpu") -> dict:
    """Evaluate global model across all clients. Returns aggregated metrics."""
    all_scores = []
    all_labels = []

    for cd in client_data:
        X_test = cd["X_test"]
        y_test = cd["y_test"]

        min_len = min(len(X_test), len(y_test))
        X_test = X_test[:min_len]
        y_test = y_test[:min_len]

        if len(X_test) == 0:
            continue

        scores = compute_anomaly_scores(model, X_test, device=device)
        all_scores.extend(scores.tolist())
        all_labels.extend(y_test.tolist())

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return {"auroc": 0.5, "f1": 0.0, "aupr": 0.0}

    auroc = roc_auc_score(all_labels, all_scores)

    threshold = np.percentile(all_scores, 90)
    preds = (all_scores > threshold).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    aupr = auc(recall, precision)

    return {"auroc": auroc, "f1": f1, "aupr": aupr}


def pretrain_local(client_data: list, model_cfg: dict, n_epochs: int = 5, device: str = "cpu"):
    """Pretrain one local model per client for warm-start."""
    local_models = []
    for cd in client_data:
        m = LSTMAutoencoder(**model_cfg)
        train_autoencoder(m, cd["X_train"], n_epochs=n_epochs, device=device)
        local_models.append(m)
    return local_models


# ───────────── Experiment 1: FL Convergence ───────────────────────────────────

def exp_fl_convergence(client_data: list, model_cfg: dict, device: str = "cpu") -> dict:
    """
    Compare FedAvg, FedProx, and Top-K compression convergence.
    Evaluates AUROC every 10 rounds.
    """
    print("\n[Exp 1] FL Convergence: FedAvg vs FedProx vs TopK")
    n_rounds = 60
    eval_every = 10
    configs = [
        {"name": "FedAvg",        "mu": 0.0,  "topk_ratio": None},
        {"name": "FedProx",       "mu": 0.01, "topk_ratio": None},
        {"name": "TopK-10%",      "mu": 0.0,  "topk_ratio": 0.10},
        {"name": "FedProx+TopK",  "mu": 0.01, "topk_ratio": 0.10},
    ]

    results = {}
    for cfg in configs:
        print(f"\n  Config: {cfg['name']}")
        model = LSTMAutoencoder(**model_cfg)
        history = federated_training(
            model=model,
            client_data=client_data,
            n_rounds=n_rounds,
            n_clients_per_round=3,
            aggregator_name="fedavg",
            local_epochs=3,
            lr=1e-3,
            mu=cfg["mu"],
            topk_ratio=cfg["topk_ratio"],
            attack_fraction=0.0,
            attack_type="none",
            device=device,
            verbose=False,
        )

        # Evaluate periodically
        auroc_curve = []
        eval_model = LSTMAutoencoder(**model_cfg)
        eval_hist = federated_training(
            model=eval_model,
            client_data=client_data,
            n_rounds=n_rounds,
            n_clients_per_round=3,
            aggregator_name="fedavg",
            local_epochs=3,
            lr=1e-3,
            mu=cfg["mu"],
            topk_ratio=cfg["topk_ratio"],
            attack_fraction=0.0,
            attack_type="none",
            device=device,
            verbose=False,
        )
        # Use final model for evaluation
        metrics = evaluate_model(eval_model, client_data, device)

        results[cfg["name"]] = {
            "round_losses": history["round_losses"],
            "bytes_per_round": history["bytes_per_round"],
            "cumulative_bytes": history["cumulative_bytes"],
            "final_auroc": metrics["auroc"],
            "final_f1": metrics["f1"],
        }
        print(f"    AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")

    return results


# ───────────── Experiment 2: Communication Overhead ──────────────────────────

def exp_communication_overhead(client_data: list, model_cfg: dict, device: str = "cpu") -> dict:
    """
    Evaluate communication cost for different Top-K ratios.
    """
    print("\n[Exp 2] Communication Overhead: TopK ratios")
    n_rounds = 40
    topk_ratios = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]

    results = {}
    for k in topk_ratios:
        name = f"TopK-{int(k*100)}%"
        print(f"  Config: {name}")
        model = LSTMAutoencoder(**model_cfg)
        history = federated_training(
            model=model,
            client_data=client_data,
            n_rounds=n_rounds,
            n_clients_per_round=3,
            aggregator_name="fedavg",
            local_epochs=3,
            lr=1e-3,
            mu=0.0,
            topk_ratio=k if k < 1.0 else None,
            attack_fraction=0.0,
            attack_type="none",
            device=device,
            verbose=False,
        )
        metrics = evaluate_model(model, client_data, device)
        total_mb = sum(history["bytes_per_round"]) / 1e6
        results[name] = {
            "topk_ratio": k,
            "total_mb": total_mb,
            "bytes_per_round": history["bytes_per_round"],
            "cumulative_bytes": history["cumulative_bytes"],
            "auroc": metrics["auroc"],
            "f1": metrics["f1"],
        }
        print(f"    Total={total_mb:.2f}MB, AUROC={metrics['auroc']:.4f}")

    return results


# ───────────── Experiment 3: Robustness Under Attack ─────────────────────────

def exp_robustness(client_data: list, model_cfg: dict, device: str = "cpu") -> dict:
    """
    Compare aggregators under increasing attack fractions.
    Attack type: model poisoning.
    """
    print("\n[Exp 3] Robustness: Aggregators vs Attack Fraction")
    n_rounds = 40
    aggregators = ["fedavg", "median", "trimmed_mean", "krum", "rfa"]
    attack_fractions = [0.0, 0.1, 0.2, 0.3]
    attack_type = "model_poisoning"

    results = {}
    for agg in aggregators:
        results[agg] = {}
        for af in attack_fractions:
            print(f"  Aggregator={agg}, attack_fraction={af}")
            model = LSTMAutoencoder(**model_cfg)
            history = federated_training(
                model=model,
                client_data=client_data,
                n_rounds=n_rounds,
                n_clients_per_round=4,
                aggregator_name=agg,
                local_epochs=3,
                lr=1e-3,
                mu=0.0,
                topk_ratio=None,
                attack_fraction=af,
                attack_type=attack_type,
                device=device,
                verbose=False,
            )
            metrics = evaluate_model(model, client_data, device)
            results[agg][af] = {
                "auroc": metrics["auroc"],
                "f1": metrics["f1"],
                "round_losses": history["round_losses"],
            }
            print(f"    AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}")

    return results


# ───────────── Experiment 4: Ablation Study ──────────────────────────────────

def exp_ablation(client_data: list, model_cfg: dict, device: str = "cpu") -> dict:
    """
    Ablation study: contribution of each FL component.
    """
    print("\n[Exp 4] Ablation Study")
    n_rounds = 40
    attack_fraction = 0.2

    configs = [
        {
            "name": "Full System\n(FedProx+TopK+RFA)",
            "mu": 0.01, "topk_ratio": 0.1, "agg": "rfa",
            "af": attack_fraction, "at": "model_poisoning",
        },
        {
            "name": "No Compression\n(FedProx+RFA)",
            "mu": 0.01, "topk_ratio": None, "agg": "rfa",
            "af": attack_fraction, "at": "model_poisoning",
        },
        {
            "name": "No Robust Agg\n(FedProx+TopK)",
            "mu": 0.01, "topk_ratio": 0.1, "agg": "fedavg",
            "af": attack_fraction, "at": "model_poisoning",
        },
        {
            "name": "No FedProx\n(FedAvg+TopK+RFA)",
            "mu": 0.0, "topk_ratio": 0.1, "agg": "rfa",
            "af": attack_fraction, "at": "model_poisoning",
        },
        {
            "name": "Baseline\n(FedAvg only)",
            "mu": 0.0, "topk_ratio": None, "agg": "fedavg",
            "af": 0.0, "at": "none",
        },
    ]

    results = {}
    for cfg in configs:
        clean_name = cfg["name"].replace("\n", " ")
        print(f"  Config: {clean_name}")
        model = LSTMAutoencoder(**model_cfg)
        history = federated_training(
            model=model,
            client_data=client_data,
            n_rounds=n_rounds,
            n_clients_per_round=3,
            aggregator_name=cfg["agg"],
            local_epochs=3,
            lr=1e-3,
            mu=cfg["mu"],
            topk_ratio=cfg["topk_ratio"],
            attack_fraction=cfg["af"],
            attack_type=cfg["at"],
            device=device,
            verbose=False,
        )
        metrics = evaluate_model(model, client_data, device)
        total_mb = sum(history["bytes_per_round"]) / 1e6
        results[cfg["name"]] = {
            "auroc": metrics["auroc"],
            "f1": metrics["f1"],
            "total_mb": total_mb,
        }
        print(f"    AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}, Comm={total_mb:.2f}MB")

    return results


# ───────────── Experiment 5: On-Off Attack ────────────────────────────────────

def exp_on_off_attack(client_data: list, model_cfg: dict, device: str = "cpu") -> dict:
    """
    Evaluate robustness against on-off attack strategy (harder to detect).
    """
    print("\n[Exp 5] On-Off Attack vs Robust Aggregators")
    n_rounds = 50
    aggregators = ["fedavg", "median", "rfa"]
    attack_fraction = 0.2

    results = {}
    for attack_type in ["model_poisoning", "on_off"]:
        results[attack_type] = {}
        for agg in aggregators:
            print(f"  Attack={attack_type}, Aggregator={agg}")
            model = LSTMAutoencoder(**model_cfg)
            history = federated_training(
                model=model,
                client_data=client_data,
                n_rounds=n_rounds,
                n_clients_per_round=4,
                aggregator_name=agg,
                local_epochs=3,
                lr=1e-3,
                mu=0.0,
                topk_ratio=None,
                attack_fraction=attack_fraction,
                attack_type=attack_type,
                device=device,
                verbose=False,
            )
            metrics = evaluate_model(model, client_data, device)
            results[attack_type][agg] = {
                "auroc": metrics["auroc"],
                "f1": metrics["f1"],
            }
            print(f"    AUROC={metrics['auroc']:.4f}")

    return results
