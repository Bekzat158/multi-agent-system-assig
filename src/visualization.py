"""
Visualization module for the article figure set.

All figures are redrawn with a consistent publication-oriented design system and
saved to figures/ as both PNG and PDF.
"""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SAVE_DIR = "figures"
PNG_DPI = 500

METHOD_COLORS = {
    "FedAvg": "#4C78A8",
    "FedProx": "#72B7B2",
    "TopK-10%": "#F58518",
    "FedProx+TopK": "#54A24B",
    "fedavg": "#4C78A8",
    "median": "#B279A2",
    "trimmed_mean": "#E45756",
    "krum": "#9D755D",
    "rfa": "#2F6B3B",
}

ATTACK_COLORS = {
    "model_poisoning": "#C44E52",
    "on_off": "#DD8452",
}

DETECTION_COLORS = {
    "signal": "#4C78A8",
    "score": "#7A5195",
    "threshold": "#D62728",
    "anomaly": "#D62728",
    "tp": "#2CA02C",
    "fp": "#FF7F0E",
    "fn": "#D62728",
    "tn": "#BDBDBD",
}


def setup_style():
    """Apply a compact, publication-safe style across all figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.6,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stix",
        }
    )
    sns.set_style("white")


def save(fig, name: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    base, _ = os.path.splitext(name)
    png_path = os.path.join(SAVE_DIR, f"{base}.png")
    pdf_path = os.path.join(SAVE_DIR, f"{base}.pdf")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def add_panel_label(ax, label: str):
    ax.text(
        -0.14,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def style_axis(
    ax, xlabel: str | None = None, ylabel: str | None = None, ygrid: bool = True
):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ygrid:
        ax.grid(axis="y")
    ax.tick_params(length=3.2, width=0.8)


def _sort_topk_names(names: list[str]) -> list[str]:
    def ratio(name: str) -> float:
        return float(name.split("-")[-1].replace("%", ""))

    return sorted(names, key=ratio, reverse=True)


def plot_dataset_overview(client_data: list):
    """Visualize the synthetic IIoT dataset with a compact shared design."""
    setup_style()
    n_clients = len(client_data)
    n_show = 240

    fig, axes = plt.subplots(n_clients, 1, figsize=(6.9, 7.4), sharex=True)
    client_colors = ["#4C78A8", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]

    line_handle = None
    anomaly_handle = None

    for cid, (cd, ax) in enumerate(zip(client_data, axes)):
        x = cd["raw"]["X"][:n_show, 0]
        labels = cd["raw"]["labels"][:n_show]
        t = np.arange(n_show)
        is_anomaly = labels == 1

        line = ax.plot(t, x, color=client_colors[cid], linewidth=1.0, alpha=0.95)[0]
        points = ax.scatter(
            t[is_anomaly],
            x[is_anomaly],
            color=DETECTION_COLORS["anomaly"],
            s=14,
            zorder=4,
        )

        line_handle = line_handle or line
        anomaly_handle = anomaly_handle or points

        ax.set_ylabel(f"C{cid}")
        ax.margins(x=0)
        ax.grid(axis="y")
        ax.spines["left"].set_visible(False)
        ax.axhline(np.mean(x), color="#E6E6E6", linewidth=0.6, zorder=0)

    style_axis(axes[-1], xlabel="Time step", ygrid=False)
    for ax in axes[:-1]:
        ax.set_xlabel("")

    add_panel_label(axes[0], "A")
    fig.legend(
        [line_handle, anomaly_handle],
        ["Sensor signal", "Injected anomaly"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
    )
    fig.subplots_adjust(top=0.93, left=0.10, right=0.99, bottom=0.08, hspace=0.18)
    save(fig, "fig_dataset_overview.png")


def plot_fl_convergence(conv_results: dict):
    """Training loss and cumulative communication cost."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.1, 3.1), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    handles = []
    labels = []

    for name in ["FedAvg", "FedProx", "TopK-10%", "FedProx+TopK"]:
        if name not in conv_results:
            continue
        result = conv_results[name]
        rounds = np.arange(1, len(result["round_losses"]) + 1)
        color = METHOD_COLORS[name]
        line = ax1.plot(rounds, result["round_losses"], color=color, linewidth=1.7)[0]
        ax2.plot(
            rounds,
            np.array(result["cumulative_bytes"]) / 1e6,
            color=color,
            linewidth=1.7,
        )
        handles.append(line)
        labels.append(name.replace("+", " + "))

    style_axis(ax1, xlabel="Round", ylabel="Training loss")
    style_axis(ax2, xlabel="Round", ylabel="Cumulative communication (MB)")
    ax1.set_xlim(1, max(len(v["round_losses"]) for v in conv_results.values()))
    ax2.set_xlim(1, max(len(v["cumulative_bytes"]) for v in conv_results.values()))
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.02),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.4,
    )
    fig.subplots_adjust(top=0.78, left=0.10, right=0.99, bottom=0.18, wspace=0.28)
    save(fig, "fig_fl_convergence.png")


def plot_communication_overhead(comm_results: dict):
    """Trade-off between compression ratio, total bytes, and AUROC."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 3.1))

    names = _sort_topk_names(list(comm_results.keys()))
    labels = [f"{int(comm_results[name]['topk_ratio'] * 100)}%" for name in names]
    total_mb = [comm_results[name]["total_mb"] for name in names]
    auroc = [comm_results[name]["auroc"] for name in names]
    x = np.arange(len(names))

    bar_colors = ["#F6C58B"] * len(names)
    if "10%" in labels:
        bar_colors[labels.index("10%")] = METHOD_COLORS["TopK-10%"]
    bars = ax1.bar(
        x, total_mb, width=0.64, color=bar_colors, edgecolor="#666666", linewidth=0.5
    )
    for bar, value in zip(bars, total_mb):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2.plot(
        x,
        auroc,
        color=METHOD_COLORS["TopK-10%"],
        marker="o",
        markersize=4.6,
        linewidth=1.5,
    )
    if "10%" in labels:
        idx = labels.index("10%")
        ax2.scatter(
            x[idx],
            auroc[idx],
            s=65,
            facecolor="white",
            edgecolor=METHOD_COLORS["TopK-10%"],
            linewidth=1.2,
            zorder=4,
        )
        ax2.annotate(
            "10%",
            (x[idx], auroc[idx]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    style_axis(ax1, xlabel="Top-k ratio", ylabel="Total communication (MB)")
    style_axis(ax2, xlabel="Top-k ratio", ylabel="AUROC")
    ax1.set_ylim(0, max(total_mb) * 1.16)
    ax2.set_ylim(min(auroc) - 0.01, max(auroc) + 0.01)
    add_panel_label(ax1, "A")
    add_panel_label(ax2, "B")
    fig.subplots_adjust(top=0.92, left=0.10, right=0.99, bottom=0.18, wspace=0.28)
    save(fig, "fig_communication_overhead.png")


def plot_robustness_comparison(rob_results: dict):
    """AUROC of each aggregator under different attack fractions."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.1, 3.2), gridspec_kw={"width_ratios": [1.12, 0.88]}
    )

    agg_order = ["fedavg", "median", "trimmed_mean", "krum", "rfa"]
    attack_fractions = [0.0, 0.1, 0.2, 0.3]
    x = np.array([0, 10, 20, 30])
    matrix = []
    handles = []
    labels = []

    for agg in agg_order:
        if agg not in rob_results:
            continue
        values = [rob_results[agg][str(frac)]["auroc"] for frac in attack_fractions]
        matrix.append(values)
        line = ax1.plot(
            x,
            values,
            marker="o",
            markersize=4.4,
            linewidth=1.6,
            color=METHOD_COLORS[agg],
        )[0]
        handles.append(line)
        labels.append(agg.replace("_", " ").title())

    ax1.axhline(0.5, color="#8C8C8C", linestyle=":", linewidth=0.9)
    style_axis(ax1, xlabel="Byzantine clients (%)", ylabel="AUROC")
    ax1.set_xticks(x)
    ax1.set_ylim(0.45, 0.84)
    add_panel_label(ax1, "A")

    heatmap_data = np.array(matrix)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.45,
        vmax=0.82,
        cbar_kws={"label": "AUROC"},
        linewidths=0.6,
        linecolor="white",
        xticklabels=["0%", "10%", "20%", "30%"],
        yticklabels=[
            name.title().replace("_", " ") for name in agg_order if name in rob_results
        ],
        ax=ax2,
        annot_kws={"size": 8},
    )
    ax2.set_xlabel("Byzantine clients (%)")
    ax2.set_ylabel("")
    ax2.tick_params(length=0)
    add_panel_label(ax2, "B")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.49, 1.03),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.1,
    )
    fig.subplots_adjust(top=0.78, left=0.10, right=0.98, bottom=0.18, wspace=0.28)
    save(fig, "fig_robustness_comparison.png")


def plot_anomaly_detection(client_data: list, model, device: str = "cpu"):
    """Visualize anomaly scores and detection outputs on test data."""
    from src.models import compute_anomaly_scores, get_threshold

    setup_style()
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(6.9, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 0.65]},
    )
    showcase_client = 0
    client = client_data[showcase_client]

    x_test = client["X_test"]
    y_test = client["y_test"]
    min_len = min(len(x_test), len(y_test), 260)
    x_test = x_test[:min_len]
    y_test = y_test[:min_len]
    t = np.arange(min_len)

    scores = compute_anomaly_scores(model, x_test, device=device)
    scores = scores[:min_len]
    threshold = get_threshold(scores, percentile=90)
    preds = (scores > threshold).astype(int)

    tp = (preds == 1) & (y_test == 1)
    fp = (preds == 1) & (y_test == 0)
    fn = (preds == 0) & (y_test == 1)

    axes[0].plot(t, x_test[:, 0, 0], color=DETECTION_COLORS["signal"], linewidth=1.0)
    axes[0].scatter(
        t[y_test == 1],
        x_test[:, 0, 0][y_test == 1],
        color=DETECTION_COLORS["anomaly"],
        s=13,
        zorder=4,
    )
    style_axis(axes[0], ylabel="Signal")
    add_panel_label(axes[0], "A")

    axes[1].plot(t, scores, color=DETECTION_COLORS["score"], linewidth=1.3)
    axes[1].axhline(
        threshold, color=DETECTION_COLORS["threshold"], linestyle="--", linewidth=1.0
    )
    axes[1].fill_between(
        t, threshold, scores, where=scores > threshold, color="#E7C8EA", alpha=0.7
    )
    style_axis(axes[1], ylabel="Reconstruction\nerror")
    add_panel_label(axes[1], "B")

    axes[2].eventplot(
        [np.where(tp)[0], np.where(fp)[0], np.where(fn)[0]],
        colors=[DETECTION_COLORS["tp"], DETECTION_COLORS["fp"], DETECTION_COLORS["fn"]],
        lineoffsets=[2, 1, 0],
        linelengths=0.7,
        linewidths=1.2,
        orientation="horizontal",
    )
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["FN", "FP", "TP"])
    style_axis(axes[2], xlabel="Time step", ylabel="Outcome", ygrid=False)
    axes[2].grid(False)
    add_panel_label(axes[2], "C")

    handles = [
        plt.Line2D([0], [0], color=DETECTION_COLORS["signal"], lw=1.2),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=DETECTION_COLORS["anomaly"],
            markersize=5.5,
        ),
        plt.Line2D([0], [0], color=DETECTION_COLORS["score"], lw=1.2),
        plt.Line2D(
            [0], [0], color=DETECTION_COLORS["threshold"], lw=1.0, linestyle="--"
        ),
    ]
    labels = ["Sensor signal", "True anomaly", "Reconstruction error", "Threshold"]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.01),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(top=0.84, left=0.10, right=0.99, bottom=0.10, hspace=0.18)
    save(fig, "fig_anomaly_detection.png")


def plot_ablation_study(ablation_results: dict):
    """Bar chart comparing ablation configurations."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.5))

    names = list(ablation_results.keys())
    short_names = ["Full", "No\nComp", "No\nRobust", "No\nProx", "Baseline"]
    auroc_vals = [ablation_results[name]["auroc"] for name in names]
    f1_vals = [ablation_results[name]["f1"] for name in names]
    mb_vals = [ablation_results[name]["total_mb"] for name in names]
    x = np.arange(len(names))

    palette = ["#54A24B", "#BDBDBD", "#E45756", "#C7C7C7", "#D4D4D4"]

    def bar_panel(ax, values, ylabel, ylim, fmt):
        bars = ax.bar(
            x, values, color=palette, edgecolor="#666666", linewidth=0.5, width=0.58
        )
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=0)
        ax.tick_params(axis="x", labelsize=7, pad=3)
        style_axis(ax, ylabel=ylabel)
        ax.set_ylim(*ylim)
        offset = (ylim[1] - ylim[0]) * 0.03
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                format(value, fmt),
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    bar_panel(axes[0], auroc_vals, "AUROC", (0.40, 0.76), ".2f")
    bar_panel(axes[1], f1_vals, "F1", (0.00, 0.33), ".2f")
    bar_panel(axes[2], mb_vals, "Communication (MB)", (0.00, 16.5), ".1f")

    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    add_panel_label(axes[2], "C")
    fig.subplots_adjust(top=0.90, left=0.08, right=0.99, bottom=0.26, wspace=0.30)
    save(fig, "fig_ablation_study.png")


def plot_architecture_diagram():
    """Integrated architecture figure showing the end-to-end pipeline."""
    setup_style()
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 16.1)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    module_bands = [
        (0.55, 3.45, "#EAF2FB", "M1\nLocal detector"),
        (3.45, 7.10, "#EEF7EC", "M2\nCommunication-efficient FL"),
        (7.10, 9.55, "#FFF5E8", "M3\nNon-IID\nstabilization"),
        (9.55, 14.70, "#F4EDF7", "M4\nRobust\naggregation"),
    ]
    for x0, x1, color, label in module_bands:
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x0, 1.55),
                x1 - x0,
                3.95,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            6.72,
            label,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )

    def node(
        x, y, w, h, label, face="#FFFFFF", edge="#5A5A5A", fontsize=8.9, weight="normal"
    ):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.05",
            linewidth=0.9,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weight,
        )

    def arrow(x1, y1, x2, y2, label=None, label_y_offset=0.22):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.1, color="#4D4D4D"),
        )
        if label:
            ax.text(
                (x1 + x2) / 2,
                y1 + label_y_offset,
                label,
                ha="center",
                va="bottom",
                fontsize=7.6,
                color="#5C5C5C",
            )

    node(0.78, 2.85, 1.18, 1.15, "K edge\nclients", face="#FFFFFF", weight="bold")
    node(2.12, 2.85, 1.28, 1.15, "Input IIoT\nwindows")
    node(3.72, 2.85, 1.34, 1.15, "Windowing\nNormalize")
    node(5.33, 2.85, 1.56, 1.15, "Local LSTM-AE\ntraining", face="#F8FBFE")
    node(7.15, 2.85, 1.30, 1.15, "Top-k + error\nfeedback", face="#FFF6EA")
    node(8.72, 2.85, 1.16, 1.15, "FedProx\nterm", face="#FFF1E1")
    node(10.18, 2.85, 1.26, 1.15, "Compressed\nuplink")
    node(11.70, 2.85, 1.72, 1.15, "RFA / Median /\nTrimmed Mean / Krum", face="#FAF6FC")
    node(13.72, 2.85, 1.28, 1.15, "Global\nmodel")

    node(13.68, 1.05, 1.35, 0.98, "Local\nthreshold")
    node(
        15.15,
        1.05,
        0.72,
        0.98,
        "AUROC\nF1\nComm.",
        face="#F7F7F7",
        fontsize=8.2,
        weight="bold",
    )

    arrow(1.96, 3.42, 2.12, 3.42)
    arrow(3.40, 3.42, 3.72, 3.42)
    arrow(5.06, 3.42, 5.33, 3.42)
    arrow(6.89, 3.42, 7.15, 3.42)
    arrow(8.45, 3.42, 8.72, 3.42)
    arrow(9.88, 3.42, 10.18, 3.42)
    arrow(11.44, 3.42, 11.70, 3.42)
    arrow(13.42, 3.42, 13.72, 3.42)
    arrow(14.35, 2.85, 14.35, 2.03, label="broadcast", label_y_offset=0.08)
    arrow(15.03, 1.54, 15.15, 1.54)

    node(8.56, 5.55, 1.62, 0.50, "non-IID\nlocal drift", face="#FFF5E8", fontsize=7.0)
    node(11.82, 5.55, 1.82, 0.50, "Byzantine\nupdates", face="#FBEAEC", fontsize=7.0)
    node(6.98, 0.90, 1.92, 0.64, "bandwidth constraint", face="#EEF7EC", fontsize=8.0)
    ax.annotate(
        "",
        xy=(9.34, 4.00),
        xytext=(9.38, 5.55),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#8C8C8C"),
    )
    ax.annotate(
        "",
        xy=(12.58, 4.00),
        xytext=(12.72, 5.55),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#8C8C8C"),
    )
    ax.annotate(
        "",
        xy=(7.72, 2.84),
        xytext=(7.72, 1.54),
        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#8C8C8C"),
    )

    save(fig, "fig_architecture_diagram.png")


def plot_on_off_attack(on_off_results: dict):
    """Compare aggregator performance against persistent vs on-off attacks."""
    setup_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.1))

    agg_order = ["fedavg", "median", "rfa"]
    labels = ["FedAvg", "Median", "RFA"]
    x = np.arange(len(agg_order))
    width = 0.34

    persistent = [on_off_results["model_poisoning"][agg]["auroc"] for agg in agg_order]
    on_off = [on_off_results["on_off"][agg]["auroc"] for agg in agg_order]

    bars1 = ax.bar(
        x - width / 2,
        persistent,
        width,
        color=ATTACK_COLORS["model_poisoning"],
        edgecolor="#666666",
        linewidth=0.5,
        label="Persistent poisoning",
    )
    bars2 = ax.bar(
        x + width / 2,
        on_off,
        width,
        color=ATTACK_COLORS["on_off"],
        edgecolor="#666666",
        linewidth=0.5,
        label="On-off attack",
    )

    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0.5, color="#8C8C8C", linestyle=":", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axis(ax, xlabel="Aggregation method", ylabel="AUROC")
    ax.set_ylim(0.45, 0.86)
    add_panel_label(ax, "A")
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 1.02), ncol=2, frameon=False)
    fig.subplots_adjust(top=0.83, left=0.12, right=0.99, bottom=0.20)
    save(fig, "fig_on_off_attack.png")


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
    """Generate and save all figure assets."""
    print("\n=== Generating Figures ===")
    plot_architecture_diagram()
    plot_dataset_overview(client_data)
    plot_fl_convergence(conv_results)
    plot_communication_overhead(comm_results)
    plot_anomaly_detection(client_data, best_model, device=device)
    plot_robustness_comparison(rob_results)
    plot_on_off_attack(on_off_results)
    plot_ablation_study(ablation_results)
    print(f"\nAll figures saved to {SAVE_DIR}/")
