"""
Visualization Module: generate all figures for the research article.
Saves all plots to figures/ directory with publication-quality styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# ─────────────────────────── Style Setup ─────────────────────────────────────

def setup_style():
    """Apply consistent publication-quality style."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fa",
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

COLORS = {
    "FedAvg":         "#2196F3",
    "FedProx":        "#4CAF50",
    "TopK-10%":       "#FF9800",
    "FedProx+TopK":   "#9C27B0",
    "fedavg":         "#2196F3",
    "median":         "#4CAF50",
    "trimmed_mean":   "#FF9800",
    "krum":           "#9C27B0",
    "rfa":            "#F44336",
    "fltrust":        "#009688",
}

ATTACK_COLORS = {
    "model_poisoning": "#E53935",
    "on_off":          "#FF6F00",
}

SAVE_DIR = "figures"


def save(fig, name: str, bbox_inches: str = "tight"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, bbox_inches=bbox_inches, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────── Figure 1: Dataset Overview ──────────────────────

def plot_dataset_overview(client_data: list):
    """Visualize the synthetic IIoT dataset: non-IID distributions per client."""
    setup_style()
    n_clients = len(client_data)
    n_features = client_data[0]["raw"]["X"].shape[1]
    n_show = 300  # samples to show

    fig, axes = plt.subplots(n_clients, 1, figsize=(12, 3 * n_clients), sharex=False)

    client_colors = plt.cm.tab10(np.linspace(0, 0.8, n_clients))

    for cid, (cd, ax) in enumerate(zip(client_data, axes)):
        X = cd["raw"]["X"][:n_show, 0]
        labels = cd["raw"]["labels"][:n_show]
        t = np.arange(n_show)

        ax.plot(t, X, color=client_colors[cid], alpha=0.85, linewidth=1.5, label=f"Sensor 0")
        anom_mask = labels == 1
        ax.scatter(t[anom_mask], X[anom_mask], color="#E53935", s=15, zorder=5, label="Anomaly")
        ax.set_ylabel(f"Client {cid}\n(f={cd['params']['base_freq']:.4f})", fontsize=10)
        ax.set_xlim(0, n_show)
        if cid == 0:
            ax.set_title("Synthetic IIoT Dataset: Non-IID Time-Series per Client", fontsize=14)
        if cid == n_clients - 1:
            ax.set_xlabel("Time Step")
        ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    save(fig, "fig_dataset_overview.png")


# ─────────────────────────── Figure 2: FL Convergence ────────────────────────

def plot_fl_convergence(conv_results: dict):
    """Training loss curves and cumulative communication cost."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for name, res in conv_results.items():
        color = COLORS.get(name, "gray")
        rounds = range(1, len(res["round_losses"]) + 1)
        ax1.plot(rounds, res["round_losses"], color=color, label=name)

    ax1.set_xlabel("Round")
    ax1.set_ylabel("Avg. Reconstruction Loss (MSE)")
    ax1.set_title("FL Convergence: Training Loss per Round")
    ax1.legend()

    for name, res in conv_results.items():
        color = COLORS.get(name, "gray")
        cum_mb = np.array(res["cumulative_bytes"]) / 1e6
        ax2.plot(range(1, len(cum_mb) + 1), cum_mb, color=color, label=name)

    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Communication (MB)")
    ax2.set_title("Communication Cost over Rounds")
    ax2.legend()

    plt.tight_layout()
    save(fig, "fig_fl_convergence.png")


# ─────────────────────────── Figure 3: Communication Overhead ────────────────

def plot_communication_overhead(comm_results: dict):
    """Trade-off between compression ratio, total bytes, and AUROC."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    names = list(comm_results.keys())
    ratios = [comm_results[n]["topk_ratio"] * 100 for n in names]
    total_mb = [comm_results[n]["total_mb"] for n in names]
    auroc = [comm_results[n]["auroc"] for n in names]

    # Bar: total comm
    ax1 = axes[0]
    bars = ax1.bar(names, total_mb, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(names))), edgecolor="k", linewidth=0.5)
    ax1.set_ylabel("Total Communication (MB)")
    ax1.set_title("Communication Cost vs Compression Ratio")
    ax1.set_xticklabels(names, rotation=30, ha="right")
    for bar, mb in zip(bars, total_mb):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{mb:.1f}", ha="center", va="bottom", fontsize=9)

    # Scatter: AUROC vs compression ratio
    ax2 = axes[1]
    scatter = ax2.scatter(ratios, auroc, c=total_mb, cmap="RdYlGn", s=120, edgecolor="k", zorder=5)
    plt.colorbar(scatter, ax=ax2, label="Total MB")
    ax2.set_xlabel("Top-K Compression Ratio (%)")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Detection Quality vs Compression Ratio")
    for r, a, n in zip(ratios, auroc, names):
        ax2.annotate(n, (r, a), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    save(fig, "fig_communication_overhead.png")


# ─────────────────────────── Figure 4: Robustness Comparison ─────────────────

def plot_robustness_comparison(rob_results: dict):
    """AUROC of each aggregator under different attack fractions."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    attack_fractions = sorted(list(list(rob_results.values())[0].keys()))
    agg_names = list(rob_results.keys())

    # Line plot: AUROC vs attack fraction
    for agg in agg_names:
        color = COLORS.get(agg, "gray")
        aurocs = [rob_results[agg][af]["auroc"] for af in attack_fractions]
        ax1.plot([af * 100 for af in attack_fractions], aurocs,
                 marker="o", color=color, label=agg.upper())

    ax1.set_xlabel("Fraction of Malicious Clients (%)")
    ax1.set_ylabel("AUROC")
    ax1.set_title("Robustness: AUROC vs Attack Fraction")
    ax1.legend(loc="lower left")
    ax1.set_ylim(0.4, 1.0)
    ax1.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Random")

    # Heatmap: AUROC values
    matrix = []
    for agg in agg_names:
        row = [rob_results[agg][af]["auroc"] for af in attack_fractions]
        matrix.append(row)
    matrix = np.array(matrix)

    im = ax2.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    plt.colorbar(im, ax=ax2, label="AUROC")
    ax2.set_xticks(range(len(attack_fractions)))
    ax2.set_xticklabels([f"{int(af*100)}%" for af in attack_fractions])
    ax2.set_yticks(range(len(agg_names)))
    ax2.set_yticklabels([a.upper() for a in agg_names])
    ax2.set_xlabel("Attack Fraction")
    ax2.set_title("AUROC Heatmap (Aggregator vs Attack %)")

    for i in range(len(agg_names)):
        for j in range(len(attack_fractions)):
            ax2.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")

    plt.tight_layout()
    save(fig, "fig_robustness_comparison.png")


# ─────────────────────────── Figure 5: Anomaly Detection Example ─────────────

def plot_anomaly_detection(client_data: list, model, device: str = "cpu"):
    """Visualize anomaly scores and detected anomalies on test data."""
    from src.models import compute_anomaly_scores, get_threshold
    setup_style()

    fig, axes = plt.subplots(3, 1, figsize=(13, 9))
    showcase_client = 0
    cd = client_data[showcase_client]

    X_test = cd["X_test"]
    y_test = cd["y_test"]
    min_len = min(len(X_test), len(y_test))
    X_test = X_test[:min_len]
    y_test = y_test[:min_len]

    scores = compute_anomaly_scores(model, X_test, device=device)

    # --- Raw signal
    ax1 = axes[0]
    t = np.arange(min_len)
    ax1.plot(t, X_test[:, 0, 0], color="#1565C0", alpha=0.8, linewidth=1.0, label="Sensor signal")
    anom_mask = y_test == 1
    ax1.scatter(t[anom_mask], X_test[:, 0, 0][anom_mask], color="#E53935", s=8, zorder=5, label="True anomaly")
    ax1.set_ylabel("Sensor Value")
    ax1.set_title(f"Client {showcase_client}: Raw Sensor Signal (Test Set)")
    ax1.legend()

    # --- Reconstruction error
    ax2 = axes[1]
    ax2.plot(t, scores, color="#6A1B9A", alpha=0.85, linewidth=1.0, label="Reconstruction Error")
    threshold = get_threshold(scores, percentile=90)
    ax2.axhline(threshold, color="#E53935", linestyle="--", label=f"Threshold (90th pct)")
    ax2.fill_between(t, 0, scores, where=(scores > threshold), alpha=0.3, color="#E53935")
    ax2.set_ylabel("Anomaly Score (MSE)")
    ax2.set_title("LSTM-AE Reconstruction Error")
    ax2.legend()

    # --- Detection result
    ax3 = axes[2]
    preds = (scores > threshold).astype(int)
    tp = (preds == 1) & (y_test == 1)
    fp = (preds == 1) & (y_test == 0)
    fn = (preds == 0) & (y_test == 1)
    tn = (preds == 0) & (y_test == 0)

    ax3.scatter(t[tn], np.zeros(tn.sum()), color="#90CAF9", s=4, label="TN")
    ax3.scatter(t[tp], np.ones(tp.sum()), color="#43A047", s=10, label="TP", zorder=5)
    ax3.scatter(t[fp], np.ones(fp.sum()), color="#FF7043", s=10, label="FP", zorder=5)
    ax3.scatter(t[fn], np.zeros(fn.sum()), color="#E53935", s=10, label="FN", zorder=5)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Normal", "Anomaly"])
    ax3.set_xlabel("Time Step")
    ax3.set_title("Detection Results (TP/FP/FN/TN)")
    ax3.legend(loc="upper right", ncol=4)

    plt.tight_layout()
    save(fig, "fig_anomaly_detection.png")


# ─────────────────────────── Figure 6: Ablation Study ────────────────────────

def plot_ablation_study(ablation_results: dict):
    """Bar chart comparing ablation configurations."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    names = list(ablation_results.keys())
    auroc_vals = [ablation_results[n]["auroc"] for n in names]
    f1_vals = [ablation_results[n]["f1"] for n in names]
    mb_vals = [ablation_results[n]["total_mb"] for n in names]

    palette = plt.cm.Set2(np.linspace(0, 1, len(names)))

    def bar_plot(ax, vals, title, ylabel, fmt=".3f"):
        bars = ax.bar(range(len(names)), vals, color=palette, edgecolor="k", linewidth=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=9)

    bar_plot(axes[0], auroc_vals, "AUROC (higher = better)", "AUROC")
    bar_plot(axes[1], f1_vals, "F1 Score (higher = better)", "F1")
    bar_plot(axes[2], mb_vals, "Total Communication (lower = better)", "MB", fmt=".1f")

    plt.suptitle("Ablation Study: Contribution of Each FL Component", fontsize=14, y=1.02)
    plt.tight_layout()
    save(fig, "fig_ablation_study.png")


# ─────────────────────────── Figure 7: Architecture Diagram ──────────────────

def plot_architecture_diagram():
    """
    System architecture diagram showing the FL pipeline.
    Drawn programmatically using matplotlib patches and arrows.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    def box(ax, x, y, w, h, label, color="#BBDEFB", text_size=9, sublabel=""):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            linewidth=1.5, edgecolor="#37474F", facecolor=color
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
                label, ha="center", va="center", fontsize=text_size, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25, sublabel,
                    ha="center", va="center", fontsize=7.5, color="#546E7A")

    def arrow(ax, x1, y1, x2, y2, label="", color="#37474F"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=8, color="#546E7A")

    # --- Edge Clients (left side)
    client_colors = ["#E3F2FD", "#E8F5E9", "#FFF8E1", "#FCE4EC", "#F3E5F5"]
    client_labels = ["Client 0\n(Factory Line A)", "Client 1\n(SCADA Node)", "Client 2\n(Edge Gateway)",
                     "Client 3\n(Sensor Array)", "Client 4\n(PLC Unit)"]
    for i in range(5):
        y_pos = 6.8 - i * 1.3
        box(ax, 0.2, y_pos, 2.8, 0.9, client_labels[i], color=client_colors[i], text_size=8)

    # --- Local Model box
    box(ax, 3.5, 0.3, 2.5, 7.2, "", color="#E0F7FA")
    ax.text(4.75, 6.9, "Local Processing", ha="center", va="center", fontsize=10, fontweight="bold", color="#00838F")
    box(ax, 3.7, 5.6, 2.1, 0.8, "Windowing &\nNormalization", color="#B2EBF2", text_size=8)
    box(ax, 3.7, 4.3, 2.1, 0.9, "LSTM-AE\nLocal Training", color="#80DEEA", text_size=8)
    box(ax, 3.7, 3.1, 2.1, 0.8, "FedProx\nProximal Loss", color="#4DD0E1", text_size=8)
    box(ax, 3.7, 2.0, 2.1, 0.7, "Top-K\nCompressor", color="#26C6DA", text_size=8)
    box(ax, 3.7, 0.6, 2.1, 1.0, "Anomaly\nScore & Threshold", color="#00BCD4", text_size=8)

    # --- Server (center)
    box(ax, 6.8, 1.0, 3.0, 5.8, "", color="#FFF3E0")
    ax.text(8.3, 6.5, "Federated Server", ha="center", va="center", fontsize=10, fontweight="bold", color="#E65100")
    box(ax, 7.0, 5.0, 2.6, 0.9, "Client Selection\n(Partial Participation)", color="#FFE0B2", text_size=8)
    box(ax, 7.0, 3.7, 2.6, 0.9, "Robust Aggregation\nRFA / Median / Krum", color="#FFCC80", text_size=8)
    box(ax, 7.0, 2.5, 2.6, 0.9, "Byzantine Filter\n& Trust Score", color="#FFB74D", text_size=8)
    box(ax, 7.0, 1.2, 2.6, 0.9, "Global Model θ\nBroadcast", color="#FF9800", text_size=8)

    # --- Threat Model (right)
    box(ax, 10.6, 4.5, 3.0, 2.0, "Adversarial\nClients", color="#FFEBEE", text_size=8,
        sublabel="Model Poisoning\nLabel Flipping / On-Off")
    box(ax, 10.6, 2.0, 3.0, 1.8, "Non-IID\nHeterogeneity", color="#F3E5F5", text_size=8,
        sublabel="By-asset / Feature-shift\nTemporal Drift")
    box(ax, 10.6, 0.3, 3.0, 1.3, "Resource\nConstraints", color="#E8F5E9", text_size=8,
        sublabel="Bandwidth / CPU / Memory")

    # --- Arrows ---
    for i in range(5):
        y_pos = 6.8 - i * 1.3 + 0.45
        arrow(ax, 3.0, y_pos, 3.6, 5.0 if i == 0 else 4.3, color="#1565C0")
    arrow(ax, 5.8, 6.0, 6.8, 6.0, label="Compressed\nUpdate ΔC(Δk)", color="#00838F")
    arrow(ax, 5.8, 4.5, 6.8, 4.5, color="#00838F")
    arrow(ax, 6.8, 1.6, 5.8, 1.6, label="Global θ", color="#E65100")
    arrow(ax, 10.6, 5.5, 9.8, 4.2, color="#C62828")
    arrow(ax, 10.6, 2.9, 9.8, 3.0, color="#7B1FA2")

    ax.set_title(
        "System Architecture: Communication-Efficient & Robust Federated Anomaly Detection for IIoT",
        fontsize=12, fontweight="bold", pad=10
    )

    plt.tight_layout()
    save(fig, "fig_architecture_diagram.png")


# ─────────────────────────── Figure 8: On-Off Attack ─────────────────────────

def plot_on_off_attack(on_off_results: dict):
    """Compare aggregator performance against persistent vs on-off attacks."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    attack_types = ["model_poisoning", "on_off"]
    attack_labels = {"model_poisoning": "Persistent Poisoning", "on_off": "On-Off Attack"}
    agg_names = list(list(on_off_results.values())[0].keys())
    x = np.arange(len(agg_names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, [on_off_results["model_poisoning"][a]["auroc"] for a in agg_names],
                   width, label=attack_labels["model_poisoning"], color="#E53935", alpha=0.85, edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, [on_off_results["on_off"][a]["auroc"] for a in agg_names],
                   width, label=attack_labels["on_off"], color="#FF6F00", alpha=0.85, edgecolor="k", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("AUROC")
    ax.set_title("Defense Performance: Persistent vs On-Off Attack Strategy (20% Attackers)")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in agg_names])
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    save(fig, "fig_on_off_attack.png")


# ─────────────────────────── Generate All Figures ────────────────────────────

def generate_all_figures(
    client_data,
    conv_results,
    comm_results,
    rob_results,
    ablation_results,
    on_off_results,
    best_model,
    device: str = "cpu",
):
    """Generate and save all figures to figures/ directory."""
    print("\n=== Generating Figures ===")
    plot_architecture_diagram()
    plot_dataset_overview(client_data)
    plot_fl_convergence(conv_results)
    plot_communication_overhead(comm_results)
    plot_robustness_comparison(rob_results)
    plot_anomaly_detection(client_data, best_model, device=device)
    plot_ablation_study(ablation_results)
    plot_on_off_attack(on_off_results)
    print(f"\nAll figures saved to {SAVE_DIR}/")
