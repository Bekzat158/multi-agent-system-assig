"""
Main pipeline: run all experiments and generate all figures.
Usage: uv run python run_experiments.py
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch

from src.data_gen import generate_all_clients
from src.models import LSTMAutoencoder, train_autoencoder
from src.experiments import (
    exp_fl_convergence,
    exp_communication_overhead,
    exp_robustness,
    exp_ablation,
    exp_on_off_attack,
    evaluate_model,
)
from src.visualization import generate_all_figures


def save_results(results: dict, path: str):
    """Save experiment results as JSON (convert numpy types)."""
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(convert(results), f, indent=2)
    print(f"  Results saved: {path}")


def main():
    t0 = time.time()
    print("=" * 70)
    print("Communication-Efficient & Robust Federated Anomaly Detection")
    print("IIoT Edge Networks — ML Experiment Pipeline")
    print("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────────
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {DEVICE}")

    N_CLIENTS = 5
    N_SAMPLES = 4000
    N_FEATURES = 6
    WINDOW_SIZE = 30
    ANOMALY_RATIO = 0.06

    MODEL_CFG = {
        "n_features": N_FEATURES,
        "window_size": WINDOW_SIZE,
        "hidden_size": 48,
        "latent_size": 12,
        "num_layers": 1,
    }

    # ── Step 1: Generate Dataset ─────────────────────────────────────────────
    print("\n[Step 1/6] Generating Synthetic IIoT Dataset...")
    client_data = generate_all_clients(
        n_clients=N_CLIENTS,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        anomaly_ratio=ANOMALY_RATIO,
        window_size=WINDOW_SIZE,
        save_dir="data",
    )

    # ── Step 2: Pretrain Baseline Model ──────────────────────────────────────
    print("\n[Step 2/6] Pretraining baseline model (local, client 0)...")
    baseline_model = LSTMAutoencoder(**MODEL_CFG)
    train_autoencoder(
        baseline_model, client_data[0]["X_train"],
        n_epochs=8, device=DEVICE, verbose=True
    )
    baseline_metrics = evaluate_model(baseline_model, client_data, DEVICE)
    print(f"  Baseline (local only): AUROC={baseline_metrics['auroc']:.4f}, F1={baseline_metrics['f1']:.4f}")

    # ── Step 3: Run Experiments ───────────────────────────────────────────────
    print("\n[Step 3/6] Running FL Experiments...")

    conv_results = exp_fl_convergence(client_data, MODEL_CFG, device=DEVICE)
    comm_results = exp_communication_overhead(client_data, MODEL_CFG, device=DEVICE)
    rob_results = exp_robustness(client_data, MODEL_CFG, device=DEVICE)
    ablation_results = exp_ablation(client_data, MODEL_CFG, device=DEVICE)
    on_off_results = exp_on_off_attack(client_data, MODEL_CFG, device=DEVICE)

    # ── Step 4: Train Best Model for Visualization ───────────────────────────
    print("\n[Step 4/6] Training best model for visualization (FedProx + RFA)...")
    from src.fl_engine import federated_training
    best_model = LSTMAutoencoder(**MODEL_CFG)
    federated_training(
        model=best_model,
        client_data=client_data,
        n_rounds=40,
        n_clients_per_round=3,
        aggregator_name="rfa",
        local_epochs=3,
        lr=1e-3,
        mu=0.01,
        topk_ratio=0.1,
        attack_fraction=0.0,
        attack_type="none",
        device=DEVICE,
        verbose=True,
    )
    best_metrics = evaluate_model(best_model, client_data, DEVICE)
    print(f"  Best FL model: AUROC={best_metrics['auroc']:.4f}, F1={best_metrics['f1']:.4f}")

    # ── Step 5: Save Results ─────────────────────────────────────────────────
    print("\n[Step 5/6] Saving results...")
    all_results = {
        "baseline": baseline_metrics,
        "best_fl": best_metrics,
        "convergence": conv_results,
        "communication": comm_results,
        "robustness": rob_results,
        "ablation": ablation_results,
        "on_off": on_off_results,
        "config": {
            "n_clients": N_CLIENTS,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "window_size": WINDOW_SIZE,
            "model_cfg": MODEL_CFG,
        },
    }
    save_results(all_results, "results/experiment_results.json")

    # ── Step 6: Generate Figures ─────────────────────────────────────────────
    print("\n[Step 6/6] Generating figures...")
    generate_all_figures(
        client_data=client_data,
        conv_results=conv_results,
        comm_results=comm_results,
        rob_results=rob_results,
        ablation_results=ablation_results,
        on_off_results=on_off_results,
        best_model=best_model,
        device=DEVICE,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Dataset:  data/")
    print(f"  Figures:  figures/  ({len(os.listdir('figures'))} files)")
    print(f"  Results:  results/experiment_results.json")
    print(f"  Best FL AUROC: {best_metrics['auroc']:.4f}")
    print(f"  Best FL F1:    {best_metrics['f1']:.4f}")
    print(f"{'='*70}")

    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'Config':<25} {'AUROC':>8} {'F1':>8} {'Comm(MB)':>10}")
    print("-" * 55)
    for name, res in conv_results.items():
        clean = name.replace("\n", " ")
        mb = sum(res["bytes_per_round"]) / 1e6
        print(f"{clean:<25} {res['final_auroc']:>8.4f} {res['final_f1']:>8.4f} {mb:>10.2f}")


if __name__ == "__main__":
    main()
